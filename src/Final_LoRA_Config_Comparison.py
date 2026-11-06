#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — LoRA Konfigürasyon Karşılaştırması
# ============================================================================
# PDF §9 "Fine-Tuning Yaklaşım Türleri" kapsamında farklı LoRA hiperparametre
# konfigürasyonlarının domain performansına etkisini analiz eder.
#
# 3 Konfigürasyon — Qwen2.5-3B üzerinde (T4 bellekle uyumlu):
#   Config A: r=8,  alpha=16  — Hafif, hızlı, az parametre
#   Config B: r=16, alpha=32  — Dengeli (mevcut standart)
#   Config C: r=32, alpha=64  — Güçlü, yavaş, çok parametre
#
# Her config için aynı veri ve eğitim adımı → loss eğrisi + QA skoru karşılaştır.
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q transformers peft trl bitsandbytes accelerate datasets pandas matplotlib
# ============================================================================

import json, os, gc, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from trl import SFTTrainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def _resolve(fname, hardcoded=""):
    import glob as _glob
    candidates = [
        os.path.join(".", "data", fname),
        os.path.join("..", "data", fname),
        os.path.join(".", fname),
        os.path.join("..", fname),
        f"/kaggle/input/cfbook/{fname}",
        f"/kaggle/input/datasets/yunusemredal/cfbook/{fname}",
        f"/kaggle/working/{fname}",
        hardcoded,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    hits = _glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if hits:
        return hits[0]
    hits = _glob.glob(f"**/{fname}", recursive=True)
    if hits:
        return os.path.abspath(hits[0])
    return hardcoded or fname

def _resolve_dir(dname):
    for c in [f"./{dname}", f"../{dname}"]:
        if os.path.exists(c):
            return c
    return f"./{dname}"

# ── Konfigürasyon ─────────────────────────────────────────────────────────────
INPUT_JSON = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
BASE_DIR   = "./final_lora_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# 3 LoRA konfigürasyonu — sadece r, alpha, target_modules farklı
LORA_CONFIGS = {
    "config_A_r8": {
        "r":              8,
        "lora_alpha":     16,
        "lora_dropout":   0.05,
        "target_modules": ["q_proj", "v_proj"],       # Minimal — sadece Q ve V
        "label":          "r=8, α=16, target=[q,v]",
        "params_note":    "Az parametre, hızlı eğitim",
    },
    "config_B_r16": {
        "r":              16,
        "lora_alpha":     32,
        "lora_dropout":   0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Standart
        "label":          "r=16, α=32, target=[q,k,v,o]",
        "params_note":    "Dengeli — proje standardı",
    },
    "config_C_r32": {
        "r":              32,
        "lora_alpha":     64,
        "lora_dropout":   0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],     # Tam MLP dahil
        "label":          "r=32, α=64, target=[q,k,v,o,gate,up,down]",
        "params_note":    "Güçlü — MLP katmanları dahil",
    },
}

# Eğitim sabitleri (tüm configler için aynı)
MAX_STEPS       = 80      # Karşılaştırma için yeterli (tam eğitim değil)
EVAL_STEPS      = 20
LOGGING_STEPS   = 5
MAX_COUNTRIES   = 100     # Hızlı karşılaştırma için veri sınırı
LR              = 2e-4
BATCH           = 2
GRAD_ACCUM      = 4
MAX_SEQ_LEN     = 512

print("=" * 70)
print(" FINAL — LoRA KONFİGÜRASYON KARŞILAŞTIRMASI")
print(f" Model: {MODEL_ID}")
print(f" Konfigürasyonlar: {list(LORA_CONFIGS.keys())}")
print(f" Max Steps (her config): {MAX_STEPS}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ SETİ
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== VERI SETI HAZIRLIGI ===")

IMS, IME = "<|im_start|>", "<|im_end|>"
SYS_MSG = "You are a knowledgeable assistant about world countries."


def fmt_qwen(q, a):
    return (f"{IMS}system\n{SYS_MSG}{IME}\n"
            f"{IMS}user\n{q}{IME}\n"
            f"{IMS}assistant\n{a}{IME}")


TEMPLATES = {
    "Government: Capital - name":
        ("What is the capital of {c}?", "The capital of {c} is {v}."),
    "Geography: Location":
        ("Where is {c} located?", "{c} is located in {v}."),
    "Geography: Climate":
        ("What is the climate of {c}?", "The climate of {c}: {v}."),
    "People and Society: Population - total":
        ("What is the population of {c}?", "Population of {c}: {v}."),
    "People and Society: Languages":
        ("What languages are spoken in {c}?", "Languages in {c}: {v}."),
    "Economy: Economic overview":
        ("Give an overview of {c}'s economy.", "Economy of {c}: {v}."),
    "Government: Government type":
        ("What government does {c} have?", "{c} has a {v} government."),
    "Geography: Natural resources":
        ("What are the natural resources of {c}?", "Natural resources of {c}: {v}."),
}


def build_dataset(json_path, max_countries=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    countries = list(data.items())
    if max_countries:
        countries = countries[:max_countries]
    pairs = []
    for country, details in countries:
        for key, (q_t, a_t) in TEMPLATES.items():
            if key in details and len(str(details[key]).strip()) > 3:
                val = str(details[key])[:300]
                pairs.append({"text": fmt_qwen(q_t.format(c=country), a_t.format(c=country, v=val))})
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    return Dataset.from_list(pairs[:split]), Dataset.from_list(pairs[split:]), len(pairs)


train_ds, val_ds, total_pairs = build_dataset(INPUT_JSON, MAX_COUNTRIES)
print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Toplam: {total_pairs}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TEMEL YÜKLEMESİ (bir kez yükle, her config için yeniden kullan)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== BASE MODEL YUKLEME (4-bit) ===")

# bf16 destekleniyorsa (guncel torch'ta T4 dahil) bf16, yoksa fp16.
# bf16 yolu fp16 GradScaler'a ihtiyac duymaz -> "unscale bf16" hatasini tamamen onler.
_BF16_OK = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
_COMPUTE_DTYPE = torch.bfloat16 if _BF16_OK else torch.float16
print(f"[INFO] bf16 destegi: {_BF16_OK}  ->  compute dtype: {_COMPUTE_DTYPE}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=_COMPUTE_DTYPE,
)

tokenizer = None
base_model_loaded = False

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("[OK] Tokenizer yuklendi.")
    base_model_loaded = True
except Exception as e:
    print(f"[HATA] Tokenizer yuklenemedi: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. HER KONFİG İÇİN EĞİTİM
# ══════════════════════════════════════════════════════════════════════════════
comparison_results = []

for cfg_key, cfg in LORA_CONFIGS.items():
    print(f"\n{'═'*60}")
    print(f" KONFİGÜRASYON: {cfg_key}")
    print(f" {cfg['label']}")
    print(f" Not: {cfg['params_note']}")
    print(f"{'═'*60}")

    adapter_dir = os.path.join(BASE_DIR, cfg_key)
    os.makedirs(adapter_dir, exist_ok=True)

    if not base_model_loaded:
        print("[ATLANDI] Base model yuklenemedi.")
        comparison_results.append({
            "Config": cfg_key, "Label": cfg["label"],
            "r": cfg["r"], "alpha": cfg["lora_alpha"],
            "n_targets": len(cfg["target_modules"]),
            "Train_Loss": None, "Eval_Loss": None,
            "Trainable_Params": None, "Train_Time_min": None,
        })
        continue

    # Her config için modeli yeniden yukle (temiz baslangic)
    t_load = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto",
            torch_dtype=_COMPUTE_DTYPE,   # bnb compute dtype ile tutarli
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print(f"[HATA] Model yuklenemedi: {e}")
        comparison_results.append({
            "Config": cfg_key, "Label": cfg["label"],
            "r": cfg["r"], "alpha": cfg["lora_alpha"],
            "n_targets": len(cfg["target_modules"]),
            "Train_Loss": None, "Eval_Loss": None,
            "Trainable_Params": None, "Train_Time_min": None,
        })
        continue

    # LoRA uygula
    peft_cfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["target_modules"],
    )

    # Egitimi sayılabilir parametreleri hesapla
    peft_model = get_peft_model(model, peft_cfg)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in peft_model.parameters())
    trainable_pct    = 100 * trainable_params / total_params
    print(f"  Egitilecek Param : {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"  Toplam Param     : {total_params:,}")

    # Egitilebilir (LoRA) parametreleri fp32'ye sabitle:
    # peft, adaptorleri model config'inin bfloat16 dtype'iyle olusturabiliyor;
    # fp16 GradScaler bf16 gradyani unscale edemedigi icin (T4) hata veriyor.
    for _n, _p in peft_model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.float32:
            _p.data = _p.data.to(torch.float32)

    # Sürüm-uyumlu egitim argumanlari (eski/yeni trl + transformers)
    import inspect
    try:
        from trl import SFTConfig
        _HAS_SFTCONFIG = True
    except Exception:
        _HAS_SFTCONFIG = False

    _ArgsClass = SFTConfig if _HAS_SFTCONFIG else TrainingArguments
    _args_sig  = inspect.signature(_ArgsClass.__init__).parameters
    _ta = dict(
        output_dir=adapter_dir,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="paged_adamw_32bit",
        max_steps=MAX_STEPS,
        save_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LR,
        fp16=False,           # fp16 GradScaler bf16 gradyani unscale edemiyor
        bf16=_BF16_OK,        # bf16 yolu scaler kullanmaz -> hata cikmaz
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="cosine",
        eval_steps=EVAL_STEPS,
        save_total_limit=1,
        report_to="none",
    )
    _ta["eval_strategy" if "eval_strategy" in _args_sig else "evaluation_strategy"] = "steps"
    if _HAS_SFTCONFIG:
        if "dataset_text_field" in _args_sig: _ta["dataset_text_field"] = "text"
        if   "max_seq_length" in _args_sig:   _ta["max_seq_length"] = MAX_SEQ_LEN
        elif "max_length"     in _args_sig:   _ta["max_length"] = MAX_SEQ_LEN
    _ta = {k: v for k, v in _ta.items() if k in _args_sig}
    training_args = _ArgsClass(**_ta)

    _sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    _sk = dict(model=peft_model, train_dataset=train_ds, eval_dataset=val_ds,
               args=training_args)
    _sk["tokenizer" if "tokenizer" in _sft_sig else "processing_class"] = tokenizer
    if not _HAS_SFTCONFIG:
        if "dataset_text_field" in _sft_sig: _sk["dataset_text_field"] = "text"
        if "max_seq_length"     in _sft_sig: _sk["max_seq_length"] = MAX_SEQ_LEN
    trainer = SFTTrainer(**_sk)

    t0 = time.time()
    print(f"  [INFO] Egitim basliyor ({MAX_STEPS} adim)...")
    train_result = trainer.train()
    train_time   = time.time() - t0

    # Eval loss (son checkpoint'ten)
    eval_out   = trainer.evaluate()
    eval_loss  = eval_out.get("eval_loss", None)
    train_loss = train_result.training_loss

    print(f"  [SONUC] Train Loss : {train_loss:.4f}")
    eval_loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "N/A"
    print(f"           Eval Loss  : {eval_loss_str}")
    print(f"           Sure       : {train_time/60:.1f} dk")

    # Loss grafigi kaydet
    if hasattr(trainer.state, "log_history"):
        logs = trainer.state.log_history
        tr_l = [(l["step"], l["loss"]) for l in logs if "loss" in l]
        ev_l = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]
        fig, ax = plt.subplots(figsize=(8, 4))
        if tr_l:
            s, l = zip(*tr_l)
            ax.plot(s, l, label="Train", alpha=0.8)
        if ev_l:
            s, l = zip(*ev_l)
            ax.plot(s, l, label="Eval", marker="o", color="red")
        ax.set_title(f"Loss — {cfg['label']}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(adapter_dir, f"loss_{cfg_key}.png"), dpi=130)
        plt.close()

    # Adapter kaydet
    trainer.model.save_pretrained(adapter_dir)

    comparison_results.append({
        "Config":          cfg_key,
        "Label":           cfg["label"],
        "r":               cfg["r"],
        "alpha":           cfg["lora_alpha"],
        "n_targets":       len(cfg["target_modules"]),
        "Trainable_Params":trainable_params,
        "Trainable_Pct":   round(trainable_pct, 4),
        "Train_Loss":      round(train_loss, 4),
        "Eval_Loss":       round(eval_loss, 4) if eval_loss else None,
        "Train_Time_min":  round(train_time / 60, 1),
        "GPU_Hours":       round(train_time / 3600, 4),
        "Cost_USD":        round(train_time / 3600 * 0.35, 4),
    })

    # Bellek temizle
    del peft_model, model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 4. KARŞILAŞTIRMA TABLOSU + GÖRSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" LORA KONFİGÜRASYON KARŞILAŞTIRMA SONUÇLARI")
print("=" * 60)

df_comp = pd.DataFrame(comparison_results)
df_comp.to_csv(os.path.join(OUTPUT_DIR, "lora_config_comparison.csv"), index=False)
print(df_comp[["Config", "r", "alpha", "n_targets", "Trainable_Params",
               "Trainable_Pct", "Train_Loss", "Eval_Loss", "Train_Time_min"]].to_string(index=False))

df_valid = df_comp.dropna(subset=["Train_Loss"])
if not df_valid.empty:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Train ve Eval Loss
    x = range(len(df_valid))
    w = 0.35
    ax = axes[0, 0]
    ax.bar([i - w/2 for i in x], df_valid["Train_Loss"], w, label="Train Loss", color="#3498db")
    if df_valid["Eval_Loss"].notna().any():
        ax.bar([i + w/2 for i in x], df_valid["Eval_Loss"].fillna(0), w, label="Eval Loss", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(df_valid["Config"], rotation=15, fontsize=8)
    ax.set_title("Train vs Eval Loss")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Egitilecek Parametre Sayisi
    ax = axes[0, 1]
    bars = ax.bar(df_valid["Config"], df_valid["Trainable_Params"] / 1e6,
                  color=["#2ecc71", "#f39c12", "#9b59b6"])
    ax.set_title("Eğitilebilir Parametre (Milyon)")
    ax.set_ylabel("Parametre (M)")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    for bar, val in zip(bars, df_valid["Trainable_Params"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f"{val/1e6:.1f}M", ha="center", fontsize=8)

    # (3) Egitim Suresi
    ax = axes[1, 0]
    bars = ax.bar(df_valid["Config"], df_valid["Train_Time_min"],
                  color=["#1abc9c", "#e67e22", "#8e44ad"])
    ax.set_title("Eğitim Süresi (Dakika)")
    ax.set_ylabel("Dakika")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    for bar, val in zip(bars, df_valid["Train_Time_min"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", fontsize=9)

    # (4) r Degeri vs Train Loss (trade-off)
    ax = axes[1, 1]
    ax.scatter(df_valid["r"], df_valid["Train_Loss"],
               s=df_valid["n_targets"] * 50, c=["#3498db", "#e74c3c", "#2ecc71"],
               alpha=0.8, edgecolors="black", linewidths=1.5)
    for _, row in df_valid.iterrows():
        ax.annotate(row["Config"],
                    (row["r"], row["Train_Loss"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Train Loss")
    ax.set_title("r Değeri vs Train Loss (baloncuk = hedef katman sayısı)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("LoRA Konfigürasyon Karşılaştırması — Qwen2.5-3B", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lora_config_comparison.png"), dpi=150)
    print("\n[OK] lora_config_comparison.png kaydedildi")
    plt.close()

    # Birlesik loss grafigi
    loss_files = []
    for cfg_key in LORA_CONFIGS:
        path = os.path.join(BASE_DIR, cfg_key)
        for pf in os.listdir(path):
            if pf.endswith(".png") and "loss" in pf:
                loss_files.append((cfg_key, os.path.join(path, pf)))

    # Analiz notu
    best_cfg = df_valid.loc[df_valid["Train_Loss"].idxmin()]
    print(f"\n[ANALIZ] En Dusuk Train Loss  : {best_cfg['Config']} ({best_cfg['Train_Loss']:.4f})")
    print("[ANALIZ] Parametre-Loss Trade-off:")
    for _, row in df_valid.iterrows():
        print(f"  {row['Config']:18s}: Loss={row['Train_Loss']:.4f} | "
              f"Param={row['Trainable_Params']/1e6:.1f}M | "
              f"Sure={row['Train_Time_min']:.1f}dk")

print("\n" + "=" * 70)
print(" LoRA KONFİGÜRASYON KARŞILAŞTIRMASI TAMAMLANDI!")
print(f" Sonuc: {OUTPUT_DIR}/lora_config_comparison.csv")
print(" Raporda bu tabloyu 'en iyi LoRA konfigürasyonunu seçme' gerekçesi olarak kullanin.")
print("=" * 70)
