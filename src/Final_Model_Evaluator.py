#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — Model Evaluator (5 Seviyeli Degerlendirme)
# ============================================================================
# Her lokal model icin AYRI bir Kaggle notebook'ta calistirin.
# Sadece MODEL_KEY degerini degistirin.
#
# 5 Seviye:
#   A) Retriever-Only    (retriever_results.csv'den okunur)
#   B) Base LLM (No RAG)
#   C) Fine-Tuned LLM (No RAG)  — adapter otomatik tespit edilir
#   D) RAG + Base LLM
#   E) RAG + Fine-Tuned LLM    — adapter otomatik tespit edilir
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q chromadb sentence-transformers langchain_text_splitters
# !pip install -q transformers accelerate bitsandbytes peft pandas matplotlib seaborn psutil
#
# Kaggle Secrets: HF_TOKEN tanimli olmalidir.
# ============================================================================

import json, os, gc, time, random, re, psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import chromadb
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter

# --- Secrets ---
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    print("[INFO] HF_TOKEN alindi.")
except Exception:
    print("[UYARI] HF_TOKEN alinamadi.")


# ── Mevcut modeller ───────────────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "qwen_3b":    {"id": "Qwen/Qwen2.5-3B-Instruct",           "size": "3B",   "quant": "4bit", "tmpl": "chatml"},
    "phi_35":     {"id": "microsoft/Phi-3.5-mini-instruct",     "size": "3.8B", "quant": "none", "tmpl": "phi3"},
    "llama3_8b":  {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "size": "8B",   "quant": "8bit", "tmpl": "llama3"},
    "mistral_7b": {"id": "mistralai/Mistral-7B-Instruct-v0.2",  "size": "7B",   "quant": "8bit", "tmpl": "mistral"},
    "gemma_2b":   {"id": "google/gemma-2-2b-it",                "size": "2B",   "quant": "none", "tmpl": "gemma"},
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  >> SADECE BU KISMI DEGISTIRIN (Veya komut satırından parametre verin)║
# ║  Secenekler: "qwen_3b", "phi_35", "llama3_8b",                     ║
# ║              "mistral_7b", "gemma_2b"                               ║
import sys
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in AVAILABLE_MODELS else "llama3_8b"
# ╚══════════════════════════════════════════════════════════════════════╝

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

INPUT_JSON  = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
CHROMA_DIR  = "./chroma_db_final"
FT_DIR = _resolve_dir("final_finetuned")    # Fine-tuning scriptlerinin cikti dizini
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Fine-tuned adapter yollarini OTOMATIK tespit et ───────────────────────────
# Fine-tuning scriptleri adapter'i FT_DIR/{model_key}/ altina kaydeder.
FT_ADAPTER_PATHS = {
    "llama3_8b": os.path.join(FT_DIR, "llama3"),
    "qwen_3b":   os.path.join(FT_DIR, "qwen"),
    "phi_35":    os.path.join(FT_DIR, "phi35"),   # Phi-3.5 fine-tune adapter'i
}

def detect_adapter(model_key):
    """adapter_config.json varsa adapter path doner, yoksa None."""
    path = FT_ADAPTER_PATHS.get(model_key)
    if path and os.path.exists(os.path.join(path, "adapter_config.json")):
        print(f"[OK] Fine-tuned adapter bulundu: {path}")
        return path
    print(f"[INFO] '{model_key}' icin fine-tuned adapter bulunamadi — C/E seviyeleri atlanacak.")
    return None

mcfg          = AVAILABLE_MODELS[MODEL_KEY]
MODEL_ID      = mcfg["id"]
CHAT_TMPL     = mcfg["tmpl"]
FT_ADAPTER    = detect_adapter(MODEL_KEY)   # None ise fine-tuned testler atlanir

print("=" * 70)
print(f" FINAL MODEL EVALUATOR — {MODEL_KEY} ({MODEL_ID})")
print(f" Fine-Tuned Adapter : {FT_ADAPTER or 'YOK'}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE KONFIGURASYONU + VERI + CHROMADB
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Pipeline konfig + veri + ChromaDB hazirlaniyor...")

cfg_path = os.path.join(OUTPUT_DIR, "pipeline_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        pcfg = json.load(f)
    BEST_COLL      = pcfg.get("best_collection",  "final_e5_base_sentence_based")
    BEST_K         = pcfg.get("best_top_k",        7)
    BEST_EMB_KEY   = pcfg.get("best_emb_key",      "e5_base")
    BEST_CHUNK_KEY = pcfg.get("best_chunk_key",    "sentence_based")
    BEST_THRESHOLD = pcfg.get("best_threshold",    "None")
    print(f"[INFO] pipeline_config.json okundu: {BEST_COLL} | k={BEST_K}")
else:
    BEST_COLL      = "final_e5_base_sentence_based"
    BEST_K         = 7
    BEST_EMB_KEY   = "e5_base"
    BEST_CHUNK_KEY = "sentence_based"
    BEST_THRESHOLD = "None"
    print("[UYARI] pipeline_config.json yok — varsayilan degerler kullanildi.")

EMBEDDING_MODELS_MAP = {
    "e5_base":            "intfloat/e5-base-v2",
    "bge_base":           "BAAI/bge-base-en-v1.5",
    "gte_base":           "thenlper/gte-base",
    "multilingual_mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
EMB_MODEL_NAME = EMBEDDING_MODELS_MAP.get(BEST_EMB_KEY, "intfloat/e5-base-v2")

# Veri yukle
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
            mc = ps[0].strip()
            sc = ps[1].strip() if len(ps) > 1 else ""
            secs.setdefault(mc, []).append(f"{sc}: {v}" if sc else str(v))
        for cat, ls in secs.items():
            txt += f"=== {cat} ===\n" + "".join(f"- {l}\n" for l in ls) + "\n"
        docs.append({"country": country, "text": txt, "metadata": {"country": country}})
    return docs, data

docs, raw_data = load_data(INPUT_JSON)

# Chunking fonksiyonlari (Final pipeline ile ayni)
def chunk_sliding_window(docs, chunk_size=300, step=150):
    out = []
    for d in docs:
        text = d["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if len(piece) >= 50:
                out.append({"text": f"Country: {d['country']}\n{piece}", "metadata": d["metadata"]})
            if end == len(text): break
            start += step
    return out

def chunk_sentence_based(docs, max_chars=600, min_chars=80):
    sent_re = re.compile(r"(?<=[.!?])\s+")
    out = []
    for d in docs:
        sentences = sent_re.split(d["text"])
        buffer = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            if len(buffer) + len(sent) <= max_chars:
                buffer += " " + sent
            else:
                if len(buffer) >= min_chars:
                    out.append({"text": f"Country: {d['country']}\n{buffer.strip()}", "metadata": d["metadata"]})
                buffer = sent
        if len(buffer) >= min_chars:
            out.append({"text": f"Country: {d['country']}\n{buffer.strip()}", "metadata": d["metadata"]})
    return out

def chunk_hierarchical(docs):
    out = []
    for d in docs:
        out.append({"text": f"[SUMMARY] Country: {d['country']}\n{d['text'][:400].strip()}", "metadata": {**d["metadata"], "level": "summary"}})
        secs = d["text"].split("===")
        for i in range(1, len(secs), 2):
            if i + 1 < len(secs):
                out.append({"text": f"[DETAIL] Country: {d['country']}\n=== {secs[i].strip()} ===\n{secs[i+1].strip()}", "metadata": {**d["metadata"], "level": "detail"}})
    return out

CHUNK_FUNCS = {
    "sliding_window": chunk_sliding_window,
    "sentence_based": chunk_sentence_based,
    "hierarchical":   chunk_hierarchical,
}
chunk_func = CHUNK_FUNCS.get(BEST_CHUNK_KEY, chunk_sentence_based)
chunks = chunk_func(docs)

# Embed + ChromaDB
device = "cuda:0" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
embeddings = st_model.encode([c["text"] for c in chunks], batch_size=256, show_progress_bar=True)
del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

_cdb = chromadb.PersistentClient(path=f"./chroma_eval_{MODEL_KEY}")
try: _cdb.delete_collection(BEST_COLL)
except: pass

coll = _cdb.create_collection(name=BEST_COLL)
for i in range(0, len(chunks), 5000):
    end = min(i + 5000, len(chunks))
    coll.add(
        embeddings=embeddings[i:end].tolist(),
        documents=[c["text"] for c in chunks[i:end]],
        metadatas=[c["metadata"] for c in chunks[i:end]],
        ids=[f"{BEST_COLL}_{i+j}" for j in range(end-i)],
    )
print(f"[INFO] ChromaDB: {coll.count()} chunk | Top-K: {BEST_K}")
del embeddings

# Eval set + stress test sorularini yukle
eval_path = os.path.join(OUTPUT_DIR, "eval_set_extended.json")
if not os.path.exists(eval_path):
    eval_path = _resolve("eval_set.json", "/kaggle/input/datasets/yunusemredal/cfbook/eval_set.json")
with open(eval_path, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)
print(f"[INFO] Eval set: {len(eval_questions)} soru")

stress_path = os.path.join(OUTPUT_DIR, "stress_test_questions.json")
if os.path.exists(stress_path):
    with open(stress_path) as f: stress_questions = json.load(f)
else:
    stress_questions = {
        "paraphrase": [{"question": "Tell me the capital of France.", "expected_country": "France", "expected_key": ""}],
        "ood":        [{"question": "What is the capital of Mars?",   "expected_country": "",        "expected_key": ""}],
    }

# Query embeddinglerini on-hesapla
device = "cuda:0" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
eq_embs = st_model.encode([q["question"] for q in eval_questions], batch_size=128)
all_stress_texts = [sq["question"] for qs in stress_questions.values() for sq in qs]
stress_embs = st_model.encode(all_stress_texts, batch_size=64)
del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

print("[OK] Veri hazirlama tamamlandi.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL YUKLEME (Base + Fine-Tuned)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/8] {MODEL_ID} yukleniyor...")

def get_bnb_config(quant_type):
    if quant_type == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    if quant_type == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

bnb_cfg = get_bnb_config(mcfg["quant"])
model_load_start = time.time()
base_model = None
tokenizer = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16 if bnb_cfg is None else None,
        trust_remote_code=True,
    )
    base_model.eval()
    load_time = time.time() - model_load_start
    print(f"[OK] Base model yuklendi ({load_time:.1f}s)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/(1024**3):.2f} GB VRAM")
except Exception as e:
    print(f"[HATA] Base model yuklenemedi: {e}")
"""try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # Bellek temizligini garantileyelim
    import gc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # GPU'lara esit yuk vermek icin max_memory siniri
    max_mem = {0: "10GiB", 1: "10GiB"} if torch.cuda.device_count() > 1 else None

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto",
        max_memory=max_mem,             # <--- EKLENEN SATIR
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16 if bnb_cfg is None else None,
        trust_remote_code=True,
    )

    base_model.eval()
    load_time = time.time() - model_load_start
    print(f"[OK] Base model yuklendi ({load_time:.1f}s)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/(1024**3):.2f} GB VRAM")
except Exception as e:
    print(f"[HATA] Base model yuklenemedi: {e}")
"""
# Fine-tuned model (adapter varsa)
ft_model = None
if FT_ADAPTER and base_model is not None:
    try:
        from peft import PeftModel
        ft_model = PeftModel.from_pretrained(base_model, FT_ADAPTER)
        ft_model.eval()
        print(f"[OK] Fine-tuned adapter yuklendi: {FT_ADAPTER}")
    except Exception as e:
        print(f"[UYARI] Adapter yuklenemedi: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PROMPT + GENERATION FONKSIYONLARI
# ══════════════════════════════════════════════════════════════════════════════
SYS_NO_RAG = (
    "You are a concise assistant that answers questions about countries. "
    "If you don't know, say exactly: 'I don't know'. Never fabricate information."
)
SYS_RAG = (
    "You are a concise assistant that answers questions about countries. "
    "Answer ONLY using the provided context. "
    "If the context lacks the answer, say exactly: 'I don't know'. "
    "Never fabricate or invent information."
)


def build_prompt(question, context="", use_rag=True):
    sys_msg = SYS_RAG if (use_rag and context) else SYS_NO_RAG
    if use_rag and context:
        usr_msg = f"Context:\n{context[:1500]}\n\nQuestion: {question}\nAnswer in 1-3 sentences using ONLY the context."
    else:
        usr_msg = f"Question: {question}\nAnswer concisely in 1-3 sentences."

    if CHAT_TMPL == "llama3":
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{usr_msg}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif CHAT_TMPL == "mistral":
        return f"<s>[INST] {sys_msg}\n\n{usr_msg} [/INST]"
    elif CHAT_TMPL == "chatml":
        return (f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{usr_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    elif CHAT_TMPL == "phi3":
        return (f"<|system|>\n{sys_msg}<|end|>\n"
                f"<|user|>\n{usr_msg}<|end|>\n"
                f"<|assistant|>\n")
    elif CHAT_TMPL == "gemma":
        return f"<start_of_turn>user\n{sys_msg}\n\n{usr_msg}<end_of_turn>\n<start_of_turn>model\n"
    return f"System: {sys_msg}\nUser: {usr_msg}\nAssistant:"


def generate(question, context="", use_rag=True, active_model=None, max_tokens=120):
    if active_model is None:
        return "Model yuklu degil.", 0, 0, 0

    prompt = build_prompt(question, context, use_rag)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    dev = next(active_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    mem_before = torch.cuda.memory_allocated(dev) / (1024 ** 2) if torch.cuda.is_available() else 0
    t0 = time.time()
    with torch.no_grad():
        out = active_model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - t0
    mem_after = torch.cuda.memory_allocated(dev) / (1024 ** 2) if torch.cuda.is_available() else 0

    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    tps = new_tokens / latency if latency > 0 else 0
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response, round(latency, 3), round(tps, 2), round(mem_after - mem_before, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRİK FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════
def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def calc_f1(pred, gold):
    pt = normalize(pred).split()
    gt = normalize(gold).split()
    if not pt or not gt: return int(pt == gt)
    common = Counter(pt) & Counter(gt)
    nc = sum(common.values())
    if nc == 0: return 0.0
    p, r = nc / len(pt), nc / len(gt)
    return 2 * p * r / (p + r)

def detect_hallucination(response, gold, context=""):
    resp_l = response.lower().strip()
    refuse = ["i don't know", "i do not know", "not available", "no information",
              "i cannot", "unknown", "not provided", "not specified", "no data"]
    if any(p in resp_l for p in refuse): return False
    if len(resp_l) < 5: return False
    if context and gold.lower()[:30] not in context.lower() and len(response) > 10:
        return True
    if not context:
        gw = set(normalize(gold).split())
        rw = set(normalize(response).split())
        if len(gw & rw) == 0 and len(response) > 10: return True
    return False

def human_eval_proxy(response, gold, question):
    rn, gn = normalize(response), normalize(gold)
    qkw = set(normalize(question).split())
    relevance = 1
    if len(qkw & set(rn.split())) >= 2: relevance = 3
    if gn[:20] in rn: relevance = 5
    elif any(w in rn for w in gn.split()[:3]): relevance = 4
    f1 = calc_f1(response, gold)
    faithfulness = 1
    if f1 > 0.8: faithfulness = 5
    elif f1 > 0.5: faithfulness = 4
    elif f1 > 0.2: faithfulness = 3
    elif f1 > 0.05: faithfulness = 2
    fluency = 3
    if len(response.split()) < 3: fluency = 2
    elif len(response.split()) > 5 and response[0].isupper(): fluency = 4
    return {"relevance": relevance, "faithfulness": faithfulness, "fluency": fluency}

def calc_cost(latency_s, model_size_b="8B"):
    """Tahmini GPU maliyet hesabi (T4 ~ $0.35/saat)."""
    gpu_hours = latency_s / 3600
    return round(gpu_hours * 0.35, 6)

# ══════════════════════════════════════════════════════════════════════════════
# 5. 5 SEVİYELİ DEĞERLENDİRME (A–E)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[3/8] 5-Seviyeli Degerlendirme basliyor ({len(eval_questions)} soru)...")

results_b    = []   # B) Base LLM
results_d    = []   # D) RAG + Base LLM
results_c    = []   # C) Fine-Tuned LLM (No RAG)
results_e    = []   # E) RAG + Fine-Tuned LLM

BEST_THR = float(BEST_THRESHOLD) if BEST_THRESHOLD not in ("None", None) else None

for idx, item in tqdm(enumerate(eval_questions), total=len(eval_questions), desc="Evaluation"):
    q      = item["question"]
    gold   = item["ground_truth"]
    country = item.get("metadata", {}).get("country", "")

    # Context cek (similarity threshold uygula)
    q_emb = eq_embs[idx].tolist()
    ctx_res = coll.query(query_embeddings=[q_emb], n_results=BEST_K,
                         include=["documents", "metadatas", "distances"])
    docs_ret = ctx_res["documents"][0]
    dists    = ctx_res["distances"][0]

    if BEST_THR is not None:
        docs_ret = [d for d, dist in zip(docs_ret, dists) if dist < BEST_THR]

    context = "\n---\n".join(docs_ret)

    def eval_single(response, lat, tps, mem, is_rag=False):
        em_   = exact_match(response, gold)
        f1_   = calc_f1(response, gold)
        hall_ = detect_hallucination(response, gold, context if is_rag else "")
        he_   = human_eval_proxy(response, gold, q)
        cost_ = calc_cost(lat)
        return {
            "question": q, "gold": gold[:200], "response": response[:300],
            "country": country,
            "EM": em_, "F1": round(f1_, 4), "Hallucination": hall_,
            "Latency": lat, "Tokens/s": tps, "Mem_Spike_MB": mem,
            "Cost_USD": cost_,
            **{f"HE_{k}": v for k, v in he_.items()},
        }

    # B) Base LLM (No RAG)
    r, l, t, m = generate(q, use_rag=False, active_model=base_model)
    results_b.append(eval_single(r, l, t, m, is_rag=False))

    # D) RAG + Base LLM
    r, l, t, m = generate(q, context=context, use_rag=True, active_model=base_model)
    results_d.append(eval_single(r, l, t, m, is_rag=True))

    # C) Fine-Tuned LLM (No RAG)  — sadece adapter varsa
    if ft_model is not None:
        r, l, t, m = generate(q, use_rag=False, active_model=ft_model)
        results_c.append(eval_single(r, l, t, m, is_rag=False))

        # E) RAG + Fine-Tuned LLM
        r, l, t, m = generate(q, context=context, use_rag=True, active_model=ft_model)
        results_e.append(eval_single(r, l, t, m, is_rag=True))

# DataFrame + kaydet
def save_results(data, name):
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    path = os.path.join(OUTPUT_DIR, f"results_{MODEL_KEY}_{name}.csv")
    df.to_csv(path, index=False)
    return df

df_b = save_results(results_b, "base")
df_d = save_results(results_d, "rag_base")
df_c = save_results(results_c, "ft_norag")
df_e = save_results(results_e, "rag_ft")

# ══════════════════════════════════════════════════════════════════════════════
# 6. STRESS TEST + OOD (Model Seviyesi)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/8] Stress Test ({MODEL_KEY})...")

stress_model_rows = []
emb_idx = 0
for category, questions in stress_questions.items():
    for sq in questions:
        q_emb = stress_embs[emb_idx].tolist()
        emb_idx += 1
        ctx_r = coll.query(query_embeddings=[q_emb], n_results=BEST_K,
                           include=["documents", "metadatas", "distances"])
        docs_r = ctx_r["documents"][0]
        dists_r = ctx_r["distances"][0]
        if BEST_THR is not None:
            docs_r = [d for d, dist in zip(docs_r, dists_r) if dist < BEST_THR]
        ctx = "\n---\n".join(docs_r)

        ans, lat, tps, mem = generate(sq["question"], context=ctx, use_rag=True, active_model=base_model)
        is_ood    = category == "ood"
        refused   = any(p in ans.lower() for p in ["i don't know", "not available", "no information", "i cannot"])
        c_refusal = is_ood and refused
        halluc    = is_ood and not refused and len(ans) > 10

        stress_model_rows.append({
            "Model": MODEL_KEY, "Kategori": category,
            "Soru": sq["question"][:120], "Cevap": ans[:200],
            "OOD": is_ood, "Reddetme": refused,
            "Dogru_Reddetme": c_refusal, "Halucinasyon": halluc,
            "Latency": lat,
        })

df_stress = pd.DataFrame(stress_model_rows)
df_stress.to_csv(os.path.join(OUTPUT_DIR, f"stress_{MODEL_KEY}.csv"), index=False)

print("\n--- Stress Test Sonuclari ---")
for cat in stress_questions.keys():
    sub = df_stress[df_stress["Kategori"] == cat]
    if cat == "ood":
        ref_r  = sub["Dogru_Reddetme"].mean() if len(sub) > 0 else 0
        hall_r = sub["Halucinasyon"].mean() if len(sub) > 0 else 0
        print(f"  {'OOD':>12s}: Dogru Reddetme={ref_r:.0%} | Halucinasyon={hall_r:.0%}")
    else:
        print(f"  {cat:>12s}: {len(sub)} soru test edildi")

# ══════════════════════════════════════════════════════════════════════════════
# 7. OZET TABLO + GORSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] Ozet tablo olusturuluyor...")

sys_info = {
    "ram_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
    "vram_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2) if torch.cuda.is_available() else 0,
}

def make_summary_row(level_label, df):
    if df is None or df.empty: return None
    row = {
        "Sistem":               f"{MODEL_KEY} — {level_label}",
        "EM":                   round(df["EM"].mean(), 4),
        "F1":                   round(df["F1"].mean(), 4),
        "Hallucination_Rate":   round(df["Hallucination"].mean(), 4),
        "Avg_Latency_s":        round(df["Latency"].mean(), 3),
        "Avg_Tokens/s":         round(df["Tokens/s"].mean(), 2),
        "Avg_Mem_Spike_MB":     round(df["Mem_Spike_MB"].mean(), 2),
        "Total_Cost_USD":       round(df["Cost_USD"].sum(), 4),
        "RAM_GB":               sys_info["ram_gb"],
        "VRAM_MB":              sys_info["vram_mb"],
    }
    for col in ["HE_relevance", "HE_faithfulness", "HE_fluency"]:
        row[col] = round(df[col].mean(), 2) if col in df.columns else 0
    return row

summary_rows = []
for label, df in [("B_Base_LLM", df_b), ("D_RAG+Base", df_d),
                   ("C_FT_NoRAG", df_c), ("E_RAG+FT",   df_e)]:
    r = make_summary_row(label, df)
    if r: summary_rows.append(r)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUTPUT_DIR, f"summary_{MODEL_KEY}.csv"), index=False)

print("\n--- Model Performans Ozeti ---")
if not df_summary.empty:
    print(df_summary[["Sistem", "EM", "F1", "Hallucination_Rate",
                        "Avg_Latency_s", "Avg_Tokens/s", "Total_Cost_USD"]].to_string(index=False))

# Gorsel: EM, F1, Hallucination
if not df_summary.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (metric, color_p) in enumerate([("EM", "viridis"), ("F1", "plasma"), ("Hallucination_Rate", "Reds_r")]):
        ax = axes[idx]
        bars = ax.barh(df_summary["Sistem"], df_summary[metric],
                       color=sns.color_palette(color_p, len(df_summary)))
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} — {MODEL_KEY}")
        for bar, val in zip(bars, df_summary[metric]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"metrics_{MODEL_KEY}.png"), dpi=150)
    print(f"[OK] metrics_{MODEL_KEY}.png kaydedildi")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 8. TEMIZLIK
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] GPU bellegi temizleniyor...")
del base_model
if ft_model is not None: del ft_model
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("[OK] Bellek temizlendi.")

print("\n" + "=" * 70)
print(f" {MODEL_KEY} DEGERLENDIRME TAMAMLANDI!")
print(f" Sonuclar: {OUTPUT_DIR}/results_{MODEL_KEY}_*.csv")
print(" Sonraki adim: Final_Comparison.py")
print("=" * 70)
