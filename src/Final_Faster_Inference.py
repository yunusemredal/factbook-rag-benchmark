#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL FASTER — Tek Dosya Inference Hizlandirici Benchmark
# ============================================================================
# AMAC: Bir modeli (Llama-3.1-8B) DOSYA ICINDE hizli QLoRA ile fine-tune edip,
#       KLASIK RAG ile birlikte farkli inference backend'lerinin GERCEKTEN
#       hizlandirip hizlandirmadigini olcer. Hiz yaninda KALITE'yi (EM/F1) de
#       olcer -> hizlanma cevabi bozuyor mu kontrol edilir.
#
#   >>> TAMAMEN SELF-CONTAINED — baska dosya/notebook ciktisina BAGIMLI DEGIL. <<<
#   >>> Tek dis girdi: countries.json (+ istege bagli eval_set.json).          <<<
#   >>> ANA PROJEDEN AYRI bir Kaggle hesabinda calistirilmak uzere tasarlandi. <<<
#
# Test edilen backend'ler (asagidaki BACKENDS ile ac/kapat):
#   1. transformers_4bit        — base + LoRA adapter, 4-bit        [BASELINE]
#   2. transformers_merged_fp16 — adapter merge + fp16
#   3. torch_compile            — merged fp16 + torch.compile        [YENI]
#   4. vllm                     — PagedAttention + batched + LoRA (TP=GPU say.)
#   5. ctranslate2              — merge -> int8 convert
#   6. onnxruntime              — Optimum ORT (merge -> ONNX export) [YENI / "N"]
#
# Olculen: yukleme suresi, ort. latency, token/s, throughput (q/s), VRAM,
#          EM, F1, baseline'a gore HIZLANMA (x).
#
# ── AKIS ───────────────────────────────────────────────────────────────────
#   1) countries.json -> QA verisi -> hizli QLoRA FT (Llama-3.1-8B, 4-bit)
#   2) RAG kur (chunk + embedding + ChromaDB) -> her soru icin context cek
#   3) adapter merge -> fp16  (fp16/compile/CT2/ONNX backend'leri icin)
#   4) backend'leri benchmark et (hepsi AYNI FT model + AYNI RAG context'leri)
#   5) karsilastirma tablosu + grafik
#
# ── DONANIM / BELLEK ───────────────────────────────────────────────────────
#   8B model agirdir. merged-fp16 ~16GB -> TEK T4'e SIGMAZ.
#   >>> Accelerator: GPU T4 x2 SECIN. <<<  (device_map=auto / vLLM TP=2)
#   Backend'ler OOM olursa: (a) MODEL_ID'yi kucuk modele cevir (asagida),
#   (b) BACKENDS'ten bazilarini False yap, (c) ayri kernel'larda calistir.
#
# ── KURULUM (Kaggle ilk hucre) — agir; cakisirsa backend'leri ayirin ───────
# !pip install -q vllm
# !pip install -q ctranslate2
# !pip install -q "optimum[onnxruntime-gpu]"
# !pip install -q transformers peft trl bitsandbytes accelerate datasets \
#                 sentence-transformers chromadb pandas matplotlib seaborn
#
# Kaggle Secrets: HF_TOKEN (Llama gated -> model erisim onayi da alin).
# ============================================================================

import json, os, gc, time, random, re, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Secrets ---
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print("[INFO] HF_TOKEN alindi.")
except Exception:
    print("[UYARI] HF_TOKEN alinamadi. Gated model icin gerekli olabilir.")


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

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASYON
# ══════════════════════════════════════════════════════════════════════════════
# Tek sabit: OOM olursa kucuk modele tek satirla gecin.
#   Alternatifler (tum backend'ler destekler — Llama mimarisi):
#     "meta-llama/Llama-3.2-1B-Instruct"  (gated, kucuk)
#     "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (gated DEGIL, en garanti)
#     "meta-llama/Meta-Llama-3-8B-Instruct" (ana proje ile ayni; 3.1 erisimi yoksa)
MODEL_ID    = "meta-llama/Llama-3.1-8B-Instruct"

INPUT_JSON  = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
ADAPTER_DIR = "./final_finetuned/llama_faster"   # dosya icinde egitilen adapter
MERGED_DIR  = "./llama_merged_fp16"              # merge edilmis fp16 (fp16/compile/CT2/ONNX)
CT2_DIR     = "./llama_ct2_int8"
ONNX_DIR    = "./llama_onnx"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

# Fine-tune (hizli olsun diye sinirli)
FT_MAX_STEPS    = 150
FT_MAX_COUNTRIES = 150
FT_LR           = 2e-4

# RAG (pipeline_config'a BAGIMLI DEGIL — burada sabit, iyi varsayilan)
EMB_MODEL_NAME  = "intfloat/e5-base-v2"
TOP_K           = 7

# Benchmark
N_QUERIES = 40       # benchmark soru sayisi (hizli olsun)
MAX_NEW   = 120      # uretilecek max token

BACKENDS = {
    "transformers_4bit":        True,   # BASELINE
    "transformers_merged_fp16": True,
    "torch_compile":            True,   # YENI
    "vllm":                     True,
    "ctranslate2":              True,
    "onnxruntime":              True,   # YENI ("N" = ONNX Runtime / Optimum)
}

_BF16_OK = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
_COMPUTE_DTYPE = torch.bfloat16 if _BF16_OK else torch.float16

print("=" * 70)
print(" FINAL FASTER — TEK DOSYA HIZLANDIRICI BENCHMARK")
print(f" Model: {MODEL_ID}")
print(f" GPU: {torch.cuda.device_count()} adet | bf16: {_BF16_OK}")
print(f" Backend'ler: {[k for k, v in BACKENDS.items() if v]}")
print("=" * 70)


def reset_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

# ══════════════════════════════════════════════════════════════════════════════
# 1. DOSYA ICINDE HIZLI QLoRA FINE-TUNE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Hizli QLoRA fine-tune basliyor...")

from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, prepare_model_for_kbit_training

# Llama-3 chat template
BOT, SH, EH, EOT = "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
SYS_MSG = "You are a knowledgeable assistant about world countries and geography."

def fmt_llama3(q, a):
    return (f"{BOT}{SH}system{EH}\n\n{SYS_MSG}{EOT}"
            f"{SH}user{EH}\n\n{q}{EOT}"
            f"{SH}assistant{EH}\n\n{a}{EOT}")

TEMPLATES = {
    "Government: Capital - name": ("What is the capital city of {c}?", "The capital city of {c} is {v}."),
    "Geography: Location":        ("Where is {c} located geographically?", "{c} is located in {v}."),
    "Geography: Climate":         ("Describe the climate of {c}.", "The climate of {c} is: {v}."),
    "Geography: Terrain":         ("What does the terrain of {c} look like?", "The terrain of {c}: {v}."),
    "Geography: Natural resources":("What natural resources does {c} have?", "The natural resources of {c} include: {v}."),
    "People and Society: Population - total":("What is the total population of {c}?", "The total population of {c} is approximately {v}."),
    "People and Society: Languages":("What languages are spoken in {c}?", "The languages spoken in {c} include: {v}."),
    "Economy: Economic overview": ("Give a brief economic overview of {c}.", "Economic overview of {c}: {v}."),
    "Government: Government type": ("What form of government does {c} have?", "{c} government: {v}."),
}

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    _raw = json.load(f)

_pairs = []
for country, det in list(_raw.items())[:FT_MAX_COUNTRIES]:
    for key, (qt, at) in TEMPLATES.items():
        if key in det and len(str(det[key]).strip()) > 5:
            v = str(det[key])[:400]
            _pairs.append({"text": fmt_llama3(qt.format(c=country), at.format(c=country, v=v))})
random.shuffle(_pairs)
_split = int(len(_pairs) * 0.9)
train_ds = Dataset.from_list(_pairs[:_split])
val_ds   = Dataset.from_list(_pairs[_split:])
print(f"[INFO] FT verisi: train={len(train_ds)} val={len(val_ds)}")

_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                          bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=_COMPUTE_DTYPE)
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_tok.pad_token = _tok.eos_token
_tok.padding_side = "right"

_ft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=_bnb, device_map="auto", torch_dtype=_COMPUTE_DTYPE)
_ft_model.config.use_cache = False
_ft_model = prepare_model_for_kbit_training(_ft_model)

_peft_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"])

# Sürüm-uyumlu egitim argumanlari (eski/yeni trl + transformers)
import inspect
from trl import SFTTrainer
try:
    from trl import SFTConfig
    _HAS_SFTCONFIG = True
except Exception:
    _HAS_SFTCONFIG = False

_ArgsClass = SFTConfig if _HAS_SFTCONFIG else TrainingArguments
_args_sig  = inspect.signature(_ArgsClass.__init__).parameters
_ta = dict(output_dir=ADAPTER_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=4,
           optim="paged_adamw_32bit", max_steps=FT_MAX_STEPS, save_steps=FT_MAX_STEPS,
           logging_steps=10, learning_rate=FT_LR, fp16=False, bf16=_BF16_OK, max_grad_norm=0.3,
           warmup_ratio=0.05, lr_scheduler_type="cosine", report_to="none")
if _HAS_SFTCONFIG:
    if "dataset_text_field" in _args_sig: _ta["dataset_text_field"] = "text"
    if   "max_seq_length" in _args_sig:   _ta["max_seq_length"] = 512
    elif "max_length"     in _args_sig:   _ta["max_length"] = 512
_ta = {k: v for k, v in _ta.items() if k in _args_sig}
_targs = _ArgsClass(**_ta)

_sft_sig = inspect.signature(SFTTrainer.__init__).parameters
_sk = dict(model=_ft_model, train_dataset=train_ds, eval_dataset=val_ds,
           peft_config=_peft_cfg, args=_targs)
_sk["tokenizer" if "tokenizer" in _sft_sig else "processing_class"] = _tok
if not _HAS_SFTCONFIG:
    if "dataset_text_field" in _sft_sig: _sk["dataset_text_field"] = "text"
    if "max_seq_length"     in _sft_sig: _sk["max_seq_length"] = 512

# Completion-only: loss sadece cevap tokenlarinda
if "data_collator" in _sft_sig:
    try:
        from trl import DataCollatorForCompletionOnlyLM
        _sk["data_collator"] = DataCollatorForCompletionOnlyLM(
            response_template="<|start_header_id|>assistant<|end_header_id|>\n\n", tokenizer=_tok)
        print("[INFO] Completion-only loss aktif.")
    except Exception as _e:
        print(f"[UYARI] Completion-only kurulamadi: {_e}")

_trainer = SFTTrainer(**_sk)
# Egitilebilir LoRA parametrelerini fp32'ye sabitle (bf16 grad/scaler uyumu)
for _n, _p in _trainer.model.named_parameters():
    if _p.requires_grad and _p.dtype != torch.float32:
        _p.data = _p.data.to(torch.float32)

print(f"[INFO] Egitim ({FT_MAX_STEPS} adim)...")
_res = _trainer.train()
print(f"[SONUC] FT train loss: {_res.training_loss:.4f}")
_trainer.model.save_pretrained(ADAPTER_DIR)
_tok.save_pretrained(ADAPTER_DIR)
print(f"[OK] Adapter kaydedildi: {ADAPTER_DIR}")

del _ft_model, _trainer
reset_gpu()

# ══════════════════════════════════════════════════════════════════════════════
# 2. RAG: chunk + embedding + ChromaDB + context cekme
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] RAG kuruluyor...")

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for country, det in data.items():
        txt = f"Country Name: {country}\n\n"
        secs = {}
        for k, v in det.items():
            if k == "url": continue
            ps = k.split(":", 1)
            mc, sc = ps[0].strip(), (ps[1].strip() if len(ps) > 1 else "")
            secs.setdefault(mc, []).append(f"{sc}: {v}" if sc else str(v))
        for cat, ls in secs.items():
            txt += f"=== {cat} ===\n" + "".join(f"- {l}\n" for l in ls) + "\n"
        docs.append({"country": country, "text": txt, "metadata": {"country": country}})
    return docs, data

def chunk_sentence_based(docs, max_chars=600, min_chars=80):
    sent_re = re.compile(r"(?<=[.!?])\s+")
    out = []
    for d in docs:
        buf = ""
        for sent in sent_re.split(d["text"]):
            sent = sent.strip()
            if not sent: continue
            if len(buf) + len(sent) <= max_chars:
                buf += " " + sent
            else:
                if len(buf) >= min_chars:
                    out.append({"text": f"Country: {d['country']}\n{buf.strip()}", "metadata": d["metadata"]})
                buf = sent
        if len(buf) >= min_chars:
            out.append({"text": f"Country: {d['country']}\n{buf.strip()}", "metadata": d["metadata"]})
    return out

docs, raw_data = load_data(INPUT_JSON)
chunks = chunk_sentence_based(docs)

# Eval sorulari
eval_path = _resolve("eval_set.json", "")
if eval_path and os.path.exists(eval_path):
    with open(eval_path) as f:
        eval_questions = json.load(f)
else:
    eval_questions = []
    for c in list(raw_data.keys()):
        if "Government: Capital - name" in raw_data[c]:
            eval_questions.append({"question": f"What is the capital city of {c}?",
                                   "ground_truth": str(raw_data[c]["Government: Capital - name"]),
                                   "metadata": {"country": c}})
eval_questions = eval_questions[:N_QUERIES]
print(f"[INFO] {len(eval_questions)} soru.")

import chromadb
from sentence_transformers import SentenceTransformer
device = "cuda:0" if torch.cuda.is_available() else "cpu"
_emb = SentenceTransformer(EMB_MODEL_NAME, device=device)
chunk_embs = _emb.encode([c["text"] for c in chunks], batch_size=256, show_progress_bar=True)
q_embs = _emb.encode([q["question"] for q in eval_questions], batch_size=128)
del _emb
reset_gpu()

_cdb = chromadb.PersistentClient(path="./chroma_faster")
try: _cdb.delete_collection("faster")
except: pass
coll = _cdb.create_collection(name="faster")
for i in range(0, len(chunks), 5000):
    e = min(i + 5000, len(chunks))
    coll.add(embeddings=chunk_embs[i:e].tolist(),
             documents=[c["text"] for c in chunks[i:e]],
             metadatas=[c["metadata"] for c in chunks[i:e]],
             ids=[f"f_{i+j}" for j in range(e - i)])
del chunk_embs

print("[INFO] RAG context'leri cekiliyor (tum backend'ler paylasir)...")
contexts, golds, questions = [], [], []
for idx, item in enumerate(eval_questions):
    res = coll.query(query_embeddings=[q_embs[idx].tolist()], n_results=TOP_K, include=["documents"])
    contexts.append("\n---\n".join(res["documents"][0]))
    golds.append(item["ground_truth"])
    questions.append(item["question"])

# ── Llama-3 RAG prompt ────────────────────────────────────────────────────────
SYS_RAG = ("You are a concise assistant that answers questions about countries. "
           "Answer ONLY using the provided context. "
           "If the context lacks the answer, say exactly: 'I don't know'. "
           "Never fabricate information.")

def build_prompt(question, context):
    usr = (f"Context:\n{context[:1500]}\n\nQuestion: {question}\n"
           f"Answer in 1-3 sentences using ONLY the context.")
    return (f"{BOT}{SH}system{EH}\n\n{SYS_RAG}{EOT}"
            f"{SH}user{EH}\n\n{usr}{EOT}"
            f"{SH}assistant{EH}\n\n")

prompts = [build_prompt(q, c) for q, c in zip(questions, contexts)]

# ── Metrikler ─────────────────────────────────────────────────────────────────
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def exact_match(p, g): return int(normalize(p) == normalize(g))

def calc_f1(p, g):
    pt, gt = normalize(p).split(), normalize(g).split()
    if not pt or not gt: return float(pt == gt)
    common = Counter(pt) & Counter(gt); nc = sum(common.values())
    if nc == 0: return 0.0
    pr, rc = nc / len(pt), nc / len(gt)
    return 2 * pr * rc / (pr + rc)

def score(texts):
    em = float(np.mean([exact_match(t, g) for t, g in zip(texts, golds)]))
    f1 = float(np.mean([calc_f1(t, g) for t, g in zip(texts, golds)]))
    return round(em, 4), round(f1, 4)

results = []

# ══════════════════════════════════════════════════════════════════════════════
# 3. ADAPTER MERGE -> fp16  (fp16/compile/CT2/ONNX backend'leri icin bir kez)
# ══════════════════════════════════════════════════════════════════════════════
def ensure_merged():
    if os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        return True
    print("\n[3/5] Adapter merge ediliyor (fp16)...")
    try:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
        merged.save_pretrained(MERGED_DIR, safe_serialization=True)
        AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(MERGED_DIR)
        del base, merged
        reset_gpu()
        print(f"[OK] Merge edildi: {MERGED_DIR}")
        return True
    except Exception as e:
        print(f"[UYARI] Merge basarisiz: {e}")
        reset_gpu()
        return False

# ══════════════════════════════════════════════════════════════════════════════
# 4. BACKEND'LER
# ══════════════════════════════════════════════════════════════════════════════
def _hf_terminators(tok):
    ids = [tok.eos_token_id]
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot >= 0:
            ids.append(eot)
    except Exception:
        pass
    return ids

def _gen_hf_sequential(model, tok, plist, max_new=MAX_NEW):
    term = _hf_terminators(tok)
    lats, total_new, outs = [], 0, []
    t0 = time.time()
    for p in plist:
        inp = tok(p, return_tensors="pt", truncation=True, max_length=2048)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        s = time.time()
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               eos_token_id=term, pad_token_id=tok.pad_token_id)
        lats.append(time.time() - s)
        n_new = o.shape[1] - inp["input_ids"].shape[1]
        total_new += n_new
        outs.append(tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return outs, time.time() - t0, total_new, lats

def _record(name, load_s, total, total_new, lats, texts, batched=False):
    em, f1 = score(texts)
    vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    avg_lat = (total / len(prompts)) if batched else float(np.mean(lats))
    results.append({
        "Backend": name, "Load_s": round(load_s, 1), "Total_s": round(total, 2),
        "Avg_Latency_s": round(avg_lat, 3),
        "Tokens_per_s": round(total_new / total, 1) if total > 0 else 0,
        "Throughput_qps": round(len(prompts) / total, 2) if total > 0 else 0,
        "VRAM_MB": round(vram, 0), "EM": em, "F1": f1,
    })
    print(f"  [{name}] {total:.1f}s | {total_new/total:.1f} tok/s | "
          f"{len(prompts)/total:.2f} q/s{' (BATCHED)' if batched else ''} | F1={f1:.3f}")


def bench_transformers(quant):
    reset_gpu()
    name = "transformers_4bit" if quant == "4bit" else "transformers_merged_fp16"
    print(f"\n{'='*60}\n  BACKEND: {name}\n{'='*60}")
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        if quant == "4bit":
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                         device_map="auto", torch_dtype=torch.float16)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        else:
            if not ensure_merged():
                print("  [ATLANDI] merged model yok."); return
            model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        load_s = time.time() - t0
        outs, total, tn, lats = _gen_hf_sequential(model, tok, prompts)
        _record(name, load_s, total, tn, lats, outs)
        del model
    except Exception as e:
        print(f"  [HATA] {name}: {e}")
    reset_gpu()


def bench_torch_compile():
    reset_gpu()
    print(f"\n{'='*60}\n  BACKEND: torch_compile\n{'='*60}")
    if not ensure_merged():
        print("  [ATLANDI] merged model yok."); return
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MERGED_DIR)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float16, device_map="auto")
        model.eval()
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
        load_s = time.time() - t0
        # Warmup (derleme maliyeti olcumden DISARIDA)
        print("  [INFO] torch.compile warmup (derleme)...")
        _ = _gen_hf_sequential(model, tok, prompts[:2], max_new=32)
        outs, total, tn, lats = _gen_hf_sequential(model, tok, prompts)
        _record("torch_compile", load_s, total, tn, lats, outs)
        del model
    except Exception as e:
        print(f"  [HATA] torch_compile (8B/sharded modelde graph-break olabilir): {e}")
    reset_gpu()


def bench_vllm():
    reset_gpu()
    print(f"\n{'='*60}\n  BACKEND: vllm\n{'='*60}")
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as e:
        print(f"  [ATLANDI] vLLM import: {e}"); return
    try:
        t0 = time.time()
        tp = max(1, torch.cuda.device_count())
        llm = LLM(model=MODEL_ID, dtype="half", gpu_memory_utilization=0.88, max_model_len=2048,
                  tensor_parallel_size=tp, enable_lora=True, max_lora_rank=32)
        load_s = time.time() - t0
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
        lora = LoRARequest("ft", 1, ADAPTER_DIR)
        t1 = time.time()
        outs = llm.generate(prompts, sp, lora_request=lora)   # BATCHED
        total = time.time() - t1
        texts = [o.outputs[0].text.strip() for o in outs]
        try:
            tn = sum(len(o.outputs[0].token_ids) for o in outs)
        except AttributeError:
            _fallback_tok = AutoTokenizer.from_pretrained(MODEL_ID)
            tn = sum(len(_fallback_tok.encode(t)) for t in texts)
        _record("vllm", load_s, total, tn, [], texts, batched=True)
        del llm
    except Exception as e:
        print(f"  [HATA] vLLM: {e}")
    reset_gpu()


def bench_ctranslate2():
    reset_gpu()
    print(f"\n{'='*60}\n  BACKEND: ctranslate2\n{'='*60}")
    try:
        import ctranslate2  # noqa
    except Exception as e:
        print(f"  [ATLANDI] ctranslate2 import: {e}"); return
    if not ensure_merged():
        print("  [ATLANDI] merged model yok."); return
    if not os.path.exists(os.path.join(CT2_DIR, "model.bin")):
        r = subprocess.run(["ct2-transformers-converter", "--model", MERGED_DIR,
                            "--output_dir", CT2_DIR, "--quantization", "int8", "--force"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [ATLANDI] CT2 convert hatasi:\n   {r.stderr.strip()[-400:]}"); return
    try:
        import ctranslate2
        t_load = time.time()
        gen = ctranslate2.Generator(CT2_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MERGED_DIR)
        load_s = time.time() - t_load
        outs, lats, tn = [], [], 0
        t0 = time.time()
        for p in prompts:
            toks = tok.convert_ids_to_tokens(tok(p).input_ids)
            s = time.time()
            res = gen.generate_batch([toks], max_length=MAX_NEW, sampling_temperature=0.0,
                                     include_prompt_in_result=False)
            lats.append(time.time() - s)
            out_ids = res[0].sequences_ids[0]
            tn += len(out_ids)
            outs.append(tok.decode(out_ids, skip_special_tokens=True).strip())
        _record("ctranslate2_int8", load_s, time.time() - t0, tn, lats, outs)
    except Exception as e:
        print(f"  [HATA] CT2: {e}")
    reset_gpu()


def bench_onnxruntime():
    reset_gpu()
    print(f"\n{'='*60}\n  BACKEND: onnxruntime (Optimum)\n{'='*60}")
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except Exception as e:
        print(f"  [ATLANDI] optimum/onnxruntime import: {e}"); return
    if not ensure_merged():
        print("  [ATLANDI] merged model yok."); return
    try:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MERGED_DIR)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        model = ORTModelForCausalLM.from_pretrained(MERGED_DIR, export=True, provider=provider)
        load_s = time.time() - t0
        term = _hf_terminators(tok)
        lats, tn, outs = [], 0, []
        tA = time.time()
        for p in prompts:
            inp = tok(p, return_tensors="pt", truncation=True, max_length=2048)
            s = time.time()
            o = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                               eos_token_id=term, pad_token_id=tok.pad_token_id)
            lats.append(time.time() - s)
            n_new = o.shape[1] - inp["input_ids"].shape[1]; tn += n_new
            outs.append(tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        _record("onnxruntime", load_s, time.time() - tA, tn, lats, outs)
        del model
    except Exception as e:
        print(f"  [HATA] ONNX (8B export bellek-yogun olabilir): {e}")
    reset_gpu()

# ══════════════════════════════════════════════════════════════════════════════
# 4b. CALISTIR  (vLLM GPU'yu blokladigi icin EN SONA)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Backend'ler calistiriliyor...")
if BACKENDS.get("transformers_4bit"):        bench_transformers("4bit")
if BACKENDS.get("transformers_merged_fp16"): bench_transformers(None)
if BACKENDS.get("torch_compile"):            bench_torch_compile()
if BACKENDS.get("ctranslate2"):              bench_ctranslate2()
if BACKENDS.get("onnxruntime"):              bench_onnxruntime()
if BACKENDS.get("vllm"):                     bench_vllm()

# ══════════════════════════════════════════════════════════════════════════════
# 5. KARSILASTIRMA + GORSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Sonuclar...")
df = pd.DataFrame(results)
if df.empty:
    print("[UYARI] Hicbir backend calismadi.")
else:
    base = df.loc[df["Backend"] == "transformers_4bit", "Tokens_per_s"]
    ref = float(base.iloc[0]) if len(base) else float(max(df["Tokens_per_s"].min(), 0.01))
    df["Speedup_x"] = (df["Tokens_per_s"] / max(ref, 0.01)).round(2)
    df = df.sort_values("Tokens_per_s", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "faster_inference_comparison.csv"), index=False)

    print("\n" + "=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    print("[INFO] faster_inference_comparison.csv kaydedildi")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    pal = plt.cm.viridis(np.linspace(0.2, 0.85, len(df)))
    def _bar(ax, col, title, unit=""):
        bars = ax.barh(df["Backend"], df[col], color=pal); ax.set_title(title); ax.invert_yaxis()
        for b, v in zip(bars, df[col]):
            ax.text(b.get_width(), b.get_y() + b.get_height()/2, f" {v}{unit}", va="center", fontsize=9)
    _bar(axes[0, 0], "Tokens_per_s", "Token/s (yuksek=iyi)")
    _bar(axes[0, 1], "Throughput_qps", "Throughput soru/s (yuksek=iyi)")
    _bar(axes[1, 0], "Avg_Latency_s", "Ort. Latency (dusuk=iyi)", "s")
    _bar(axes[1, 1], "F1", "F1 (kalite — degismemeli)")
    plt.suptitle(f"Final Faster — Inference Backend Karsilastirmasi ({MODEL_ID.split('/')[-1]}, FT+RAG)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "faster_inference_comparison.png"), dpi=150)
    print("[OK] faster_inference_comparison.png")
    plt.close()

    fast = df.iloc[0]
    f1_rng = df["F1"].max() - df["F1"].min()
    print(f"\n  En hizli   : {fast['Backend']} ({fast['Tokens_per_s']} tok/s, {fast['Speedup_x']}x baseline)")
    print(f"  F1 araligi : {df['F1'].min():.3f}-{df['F1'].max():.3f} "
          f"({'kalite korundu' if f1_rng < 0.05 else 'kalite degisti, incele'})")

print("\n" + "=" * 70)
print(" FINAL FASTER TAMAMLANDI!")
print("=" * 70)
