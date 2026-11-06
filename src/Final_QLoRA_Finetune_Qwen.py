#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — QLoRA Fine-Tuning: Qwen2.5-3B-Instruct
# ============================================================================
# Amac: Qwen2.5-3B modelini countries.json domain verisiyle fine-tune etmek.
# Cikti adapter yolu: ./final_finetuned/qwen/
# Bu yol Final_Model_Evaluator.py tarafindan otomatik okunur.
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q transformers peft trl bitsandbytes accelerate datasets pandas matplotlib
#
# Kaggle Secrets: HF_TOKEN tanimli olmalidir.
# Tanimli degilse: os.environ["HF_TOKEN"] = "hf_xxxx"
# ============================================================================

import json, os, gc, time, random
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from trl import SFTTrainer

# --- Secrets ---
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print("[INFO] HF_TOKEN alindi.")
except Exception:
    print("[UYARI] HF_TOKEN alinamadi. Manuel tanimlayin.")

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
INPUT_JSON  = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
ADAPTER_DIR = "./final_finetuned/qwen"    # Final_Model_Evaluator.py buraya bakar
MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

print("=" * 70)
print(" FINAL — QLoRA FINE-TUNING: Qwen2.5-3B-Instruct")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. EĞİTİM VERİ SETİ (≥500 QA Pair)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== VERI SETI HAZIRLIGI ===")

# Qwen ChatML template token sabitleri
IMS = "<|im_start|>"
IME = "<|im_end|>"
SYS_MSG = "You are a knowledgeable assistant about world countries and geography."


def fmt_qwen(question, answer):
    return (
        f"{IMS}system\n{SYS_MSG}{IME}\n"
        f"{IMS}user\n{question}{IME}\n"
        f"{IMS}assistant\n{answer}{IME}"
    )


TEMPLATES = {
    "Government: Capital - name":
        ("What is the capital city of {c}?",
         "The capital city of {c} is {v}."),
    "Geography: Location":
        ("Where is {c} located geographically?",
         "{c} is located in {v}."),
    "Geography: Climate":
        ("Describe the climate of {c}.",
         "The climate of {c} is characterized as: {v}."),
    "Geography: Terrain":
        ("What does the terrain of {c} look like?",
         "The terrain of {c}: {v}."),
    "Geography: Natural resources":
        ("What natural resources does {c} have?",
         "The natural resources of {c} include: {v}."),
    "People and Society: Population - total":
        ("What is the total population of {c}?",
         "The total population of {c} is approximately {v}."),
    "People and Society: Languages":
        ("What languages are spoken in {c}?",
         "The languages spoken in {c} include: {v}."),
    "People and Society: Religions":
        ("What are the main religions practiced in {c}?",
         "The main religions in {c} are: {v}."),
    "Economy: Economic overview":
        ("Give a brief economic overview of {c}.",
         "Economic overview of {c}: {v}."),
    "Economy: Agricultural products":
        ("What are the main agricultural products of {c}?",
         "The main agricultural products of {c} are: {v}."),
    "Government: Government type":
        ("What form of government does {c} have?",
         "{c} has the following form of government: {v}."),
    "Government: Independence":
        ("When did {c} gain its independence?",
         "Regarding independence of {c}: {v}."),
    "Economy: Industries":
        ("What are the main industries in {c}?",
         "The main industries of {c} are: {v}."),
    "People and Society: Ethnic groups":
        ("What ethnic groups live in {c}?",
         "The ethnic groups in {c}: {v}."),
}


def create_ft_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = []
    for country, details in data.items():
        for key, (q_tmpl, a_tmpl) in TEMPLATES.items():
            if key in details and len(str(details[key]).strip()) > 5:
                val = str(details[key])[:500]
                q = q_tmpl.format(c=country)
                a = a_tmpl.format(c=country, v=val)
                qa_pairs.append({"text": fmt_qwen(q, a)})

    print(f"[INFO] Toplam QA pair: {len(qa_pairs)}")
    assert len(qa_pairs) >= 500, f"[HATA] Yeterli QA pair yok: {len(qa_pairs)}"
    return qa_pairs


qa_pairs = create_ft_dataset(INPUT_JSON)
random.shuffle(qa_pairs)

split_idx  = int(len(qa_pairs) * 0.9)
train_data = Dataset.from_list(qa_pairs[:split_idx])
val_data   = Dataset.from_list(qa_pairs[split_idx:])

token_est = sum(len(p["text"].split()) for p in qa_pairs)
print(f"[INFO] Train: {len(train_data)}  |  Validation: {len(val_data)}")
print(f"[INFO] Tahmini toplam token: ~{token_est:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL + QLORA YAPILANDIRMASI
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== MODEL YUKLEME (4-bit QLoRA) ===")

# Qwen2.5-3B T4'te 4-bit quantization ile soguk yuklenebilir
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

train_start = time.time()
model = None
tokenizer = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
        torch_dtype=_COMPUTE_DTYPE,   # bnb compute dtype ile tutarli
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    print("[OK] Model 4-bit formata cekildi.")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.memory_allocated(i) / (1024 ** 3)
            print(f"  GPU {i}: {vram:.2f} GB VRAM")

except Exception as e:
    print(f"[HATA] Model yuklenemedi: {e}")

peft_config = LoraConfig(
    r=32,                # onceki: 16 -> domain bilgisini gercekten gommek icin artirildi
    lora_alpha=64,       # alpha = 2*r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Attention + MLP -> FT'nin base'den belirgin farkli olmasi icin MLP sart
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. EĞİTİM
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== EGITIM ===")

num_epochs  = 3
eff_batch   = 2 * 4
total_steps = (len(train_data) // eff_batch) * num_epochs
print(f"[INFO] Toplam adim: ~{total_steps}  ({num_epochs} epoch)")

# ── Sürüm-uyumlu egitim argumanlari (eski/yeni trl + transformers) ────────────
# Yeni transformers: evaluation_strategy -> eval_strategy
# Yeni trl: SFTConfig + dataset_text_field/max_seq_length args icinde,
#           SFTTrainer tokenizer -> processing_class
import inspect
try:
    from trl import SFTConfig
    _HAS_SFTCONFIG = True
except Exception:
    _HAS_SFTCONFIG = False

_ArgsClass = SFTConfig if _HAS_SFTCONFIG else TrainingArguments
_args_sig  = inspect.signature(_ArgsClass.__init__).parameters

_ta = dict(
    output_dir=ADAPTER_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,           # fp16 GradScaler bf16 gradyani unscale edemiyor
    bf16=_BF16_OK,        # bf16 yolu scaler kullanmaz -> hata cikmaz
    max_grad_norm=0.3,
    num_train_epochs=num_epochs,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    eval_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)
_ta["eval_strategy" if "eval_strategy" in _args_sig else "evaluation_strategy"] = "steps"
if _HAS_SFTCONFIG:                       # yeni trl: SFT alanlari args icinde
    if "dataset_text_field" in _args_sig: _ta["dataset_text_field"] = "text"
    if   "max_seq_length" in _args_sig:   _ta["max_seq_length"] = 512
    elif "max_length"     in _args_sig:   _ta["max_length"] = 512
_ta = {k: v for k, v in _ta.items() if k in _args_sig}   # imzada olmayanlari ele
training_args = _ArgsClass(**_ta)

if model is not None:
    print("[INFO] SFTTrainer olusturuluyor...")
    _sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    _sk = dict(model=model, train_dataset=train_data, eval_dataset=val_data,
               peft_config=peft_config, args=training_args)
    _sk["tokenizer" if "tokenizer" in _sft_sig else "processing_class"] = tokenizer
    if not _HAS_SFTCONFIG:               # eski trl: SFT alanlari trainer'a
        if "dataset_text_field" in _sft_sig: _sk["dataset_text_field"] = "text"
        if "max_seq_length"     in _sft_sig: _sk["max_seq_length"] = 512

    # ── Completion-only loss: loss SADECE cevap (assistant) tokenlarinda ─────────
    # Model, soruyu modellemek yerine CEVABI uretmeyi ogrenir -> FT etkisi artar.
    # NOT: Egitim loss'u duz/nan kalirsa template bulunamamistir;
    #      USE_COMPLETION_ONLY=False yapip standart egitime don.
    USE_COMPLETION_ONLY = True
    if USE_COMPLETION_ONLY and "data_collator" in _sft_sig:
        try:
            from trl import DataCollatorForCompletionOnlyLM
            _sk["data_collator"] = DataCollatorForCompletionOnlyLM(
                response_template="<|im_start|>assistant\n", tokenizer=tokenizer)
            print("[INFO] Completion-only loss aktif (sadece cevap tokenlari).")
        except Exception as _e:
            print(f"[UYARI] Completion-only kurulamadi, standart egitim: {_e}")

    trainer = SFTTrainer(**_sk)

    # Egitilebilir (LoRA) parametreleri fp32'ye sabitle:
    # peft, adaptorleri model config'inin bfloat16 dtype'iyle olusturabiliyor;
    # fp16 GradScaler bf16 gradyani unscale edemedigi icin (T4) hata veriyor.
    for _n, _p in trainer.model.named_parameters():
        if _p.requires_grad and _p.dtype != torch.float32:
            _p.data = _p.data.to(torch.float32)

    print("[INFO] Egitim basliyor...")
    train_result = trainer.train()
    train_time   = time.time() - train_start

    print(f"\n[SONUC] Egitim Suresi   : {train_time/60:.1f} dakika")
    print(f"         Train Loss      : {train_result.training_loss:.4f}")

    gpu_hours      = train_time / 3600
    estimated_cost = gpu_hours * 0.35
    print(f"         GPU Saati       : {gpu_hours:.2f}h")
    print(f"         Tahmini Maliyet : ~${estimated_cost:.2f} (Kaggle ucretsiz)")

    print(f"[INFO] Adapter kaydediliyor: {ADAPTER_DIR}")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    if hasattr(trainer.state, "log_history") and trainer.state.log_history:
        logs = trainer.state.log_history
        train_losses = [(l["step"], l["loss"]) for l in logs if "loss" in l]
        eval_losses  = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]

        fig, ax = plt.subplots(figsize=(10, 5))
        if train_losses:
            steps, losses = zip(*train_losses)
            ax.plot(steps, losses, label="Train Loss", alpha=0.7)
        if eval_losses:
            steps, losses = zip(*eval_losses)
            ax.plot(steps, losses, label="Eval Loss", marker="o", color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("QLoRA Fine-Tuning Loss — Qwen2.5-3B")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_png = os.path.join(ADAPTER_DIR, "training_loss_qwen.png")
        plt.savefig(loss_png, dpi=150)
        print(f"[INFO] {loss_png} kaydedildi")
        plt.close()

    report = {
        "model":             MODEL_ID,
        "method":            "QLoRA (4-bit, nf4)",
        "lora_r":            32,
        "lora_alpha":        64,
        "target_modules":    ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        "completion_only":   True,
        "train_examples":    len(train_data),
        "val_examples":      len(val_data),
        "total_qa_pairs":    len(qa_pairs),
        "epochs":            num_epochs,
        "learning_rate":     2e-4,
        "batch_size":        "2 (grad_accum=4)",
        "max_seq_length":    512,
        "warmup_ratio":      0.05,
        "training_time_min": round(train_time / 60, 1),
        "gpu_hours":         round(gpu_hours, 2),
        "estimated_cost_usd":round(estimated_cost, 2),
        "final_train_loss":  round(train_result.training_loss, 4),
        "hardware":          "Kaggle 2xT4 GPU",
        "framework":         "transformers + peft + trl",
        "adapter_path":      ADAPTER_DIR,
    }
    report_path = os.path.join(OUTPUT_DIR, "finetune_report_qwen.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("[INFO] finetune_report_qwen.json kaydedildi")

else:
    print("[UYARI] Model yuklenemedi — demo modunda tamamlandi.")
    report = {
        "model": MODEL_ID, "adapter_path": ADAPTER_DIR,
        "status": "demo_mode", "note": "Model yuklenemedi",
    }
    with open(os.path.join(OUTPUT_DIR, "finetune_report_qwen.json"), "w") as f:
        json.dump(report, f, indent=2)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(" QWEN QLORA FINE-TUNING TAMAMLANDI!")
print(f" Adapter: {ADAPTER_DIR}")
print(" Sonraki adim: Final_Model_Evaluator.py (MODEL_KEY='qwen_3b')")
print("=" * 70)
