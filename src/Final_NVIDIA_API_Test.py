#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — NVIDIA API Test (§6 API LLM — Gemini'ye EK 2 model)
# ============================================================================
# NVIDIA build.nvidia.com (NIM) servisi OpenAI-uyumlu bir endpoint sunar:
#   base_url = https://integrate.api.nvidia.com/v1
# Bu script 2 NVIDIA modelini RAG ve No-RAG modunda degerlendirir; ciktiyi
# summary_nvidia.csv olarak yazar -> Final_Comparison.py zaten summary_*.csv
# glob'u ile otomatik okur (ekstra ayar gerekmez).
#
# Modeller: https://build.nvidia.com/models  (asagidaki NVIDIA_MODELS sozlugu)
#
# Kaydedilen dosyalar -> final_outputs/:
#   results_nvidia_<key>_base.csv  — No-RAG sonuclari (her model)
#   results_nvidia_<key>_rag.csv   — RAG sonuclari (her model)
#   summary_nvidia.csv             — Final_Comparison.py tarafindan okunur
#   results_nvidia_ood.csv         — OOD reddetme testi
#   nvidia_comparison.png
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q openai chromadb sentence-transformers tqdm pandas matplotlib seaborn
#
# Kaggle Secrets: NVIDIA_API_KEY tanimli olmalidir.
#   (build.nvidia.com -> bir modelin sayfasi -> "Get API Key" ile alinir;
#    "nvapi-..." ile baslar.)
# ============================================================================

import json, os, time, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from collections import Counter

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


# ── Konfigürasyon ─────────────────────────────────────────────────────────────
INPUT_JSON  = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TEST EDILECEK 2 NVIDIA MODELI                                       ║
# ║  Istediginizi build.nvidia.com/models'tan secip degistirin.         ║
# ║  key (kisa ad) -> NVIDIA model id                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
NVIDIA_MODELS = {
    "llama31_8b":  "meta/llama-3.1-8b-instruct",
    "mistral_7b":  "mistralai/mistral-7b-instruct-v0.3",
    # Alternatifler (degistirebilirsiniz):
    #   "nemotron_70b": "nvidia/llama-3.1-nemotron-70b-instruct",
    #   "llama33_70b":  "meta/llama-3.3-70b-instruct",
    #   "qwen25_7b":    "qwen/qwen2.5-7b-instruct",
    #   "gemma2_9b":    "google/gemma-2-9b-it",
}

RATE_LIMIT_S = 0.5      # Istekler arasi bekleme (free tier rate limit)
MAX_EVAL_Q   = 60       # API maliyet/sure kontrolu icin soru siniri
# build.nvidia.com ucretsiz kredilerle gelir; gercek fiyat ~0 kabul edilir.
# Raporda token sayisi uzerinden tahmini maliyet icin nominal oran:
COST_PER_1M_TOKENS = 0.0   # Ucretsizse 0 birakin; ucretliyse ornek 0.20 girin

print("=" * 70)
print(f" FINAL — NVIDIA API TEST ({len(NVIDIA_MODELS)} model)")
print(f" Modeller: {list(NVIDIA_MODELS.values())}")
print("=" * 70)

# ── NVIDIA (OpenAI-uyumlu) istemci ────────────────────────────────────────────
NVIDIA_API_KEY = None
client = None
try:
    from kaggle_secrets import UserSecretsClient
    NVIDIA_API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")
    print("[INFO] NVIDIA_API_KEY alindi.")
except Exception:
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
    if NVIDIA_API_KEY:
        print("[INFO] NVIDIA_API_KEY ortam degiskeninden alindi.")
    else:
        print("[UYARI] NVIDIA_API_KEY Secrets veya ortam degiskeninden alinamadi.")

if NVIDIA_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        print(f"[OK] NVIDIA istemcisi hazir ({NVIDIA_BASE_URL})")
    except Exception as e:
        print(f"[HATA] OpenAI istemcisi kurulamadi: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ + CHROMADB (pipeline_config.json'dan en iyi koleksiyon)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Veri + ChromaDB hazirlaniyor...")

cfg_path = os.path.join(OUTPUT_DIR, "pipeline_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        pcfg = json.load(f)
    BEST_EMB_KEY   = pcfg.get("best_emb_key",   "e5_base")
    BEST_CHUNK_KEY = pcfg.get("best_chunk_key",  "sentence_based")
    BEST_K         = pcfg.get("best_top_k",      7)
    BEST_THR       = pcfg.get("best_threshold",  "None")
    print(f"[INFO] En iyi embedding: {BEST_EMB_KEY}, chunking: {BEST_CHUNK_KEY}, k={BEST_K}")
else:
    BEST_EMB_KEY, BEST_CHUNK_KEY, BEST_K, BEST_THR = "e5_base", "sentence_based", 7, "None"
    print("[UYARI] pipeline_config.json yok — varsayilan degerler kullanildi.")

EMBEDDING_MODELS_MAP = {
    "e5_base":            "intfloat/e5-base-v2",
    "bge_base":           "BAAI/bge-base-en-v1.5",
    "gte_base":           "thenlper/gte-base",
    "multilingual_mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
EMB_MODEL_NAME = EMBEDDING_MODELS_MAP.get(BEST_EMB_KEY, "intfloat/e5-base-v2")


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


docs, raw_data = load_data(INPUT_JSON)


def chunk_sentence_based(docs, max_chars=600, min_chars=80):
    sent_re = re.compile(r"(?<=[.!?])\s+")
    out = []
    for d in docs:
        sentences = sent_re.split(d["text"])
        buf = ""
        for sent in sentences:
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


import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
chunks = chunk_sentence_based(docs)

st_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
embeddings = st_model.encode([c["text"] for c in chunks], batch_size=256, show_progress_bar=True)

COLL_NAME = f"final_{BEST_EMB_KEY}_{BEST_CHUNK_KEY}"
_cdb = chromadb.PersistentClient(path="./chroma_eval_nvidia")
try: _cdb.delete_collection(COLL_NAME)
except: pass
coll = _cdb.create_collection(name=COLL_NAME)
for i in range(0, len(chunks), 5000):
    end = min(i + 5000, len(chunks))
    coll.add(
        embeddings=embeddings[i:end].tolist(),
        documents=[c["text"] for c in chunks[i:end]],
        metadatas=[c["metadata"] for c in chunks[i:end]],
        ids=[f"nv_{i+j}" for j in range(end - i)],
    )
print(f"[INFO] ChromaDB: {coll.count()} chunk | k={BEST_K}")
del embeddings

# Eval set yukle
eval_path = os.path.join(OUTPUT_DIR, "eval_set_extended.json")
if not os.path.exists(eval_path):
    eval_path = _resolve("eval_set.json", "/kaggle/input/datasets/yunusemredal/cfbook/eval_set.json")
with open(eval_path, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)
eval_questions = eval_questions[:MAX_EVAL_Q]
print(f"[INFO] Eval set: {len(eval_questions)} soru (API limiti: {MAX_EVAL_Q})")

# Query + OOD embeddingleri (bir kez hesapla, tum modeller paylasir)
eq_embs = st_model.encode([q["question"] for q in eval_questions], batch_size=128)

stress_path = os.path.join(OUTPUT_DIR, "stress_test_questions.json")
ood_questions = []
if os.path.exists(stress_path):
    with open(stress_path) as f:
        ood_questions = json.load(f).get("ood", [])
ood_embs = st_model.encode([sq["question"] for sq in ood_questions], batch_size=32) if ood_questions else None

del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 2. METRİK FONKSİYONLARI (Evaluator/Gemini ile ayni)
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
    elif len(response.split()) > 5 and response and response[0].isupper(): fluency = 4
    return {"relevance": relevance, "faithfulness": faithfulness, "fluency": fluency}

# ══════════════════════════════════════════════════════════════════════════════
# 3. NVIDIA API ÇAĞRI FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════
SYS_NO_RAG = (
    "You are a concise assistant that answers questions about countries. "
    "If you don't know the answer, say exactly: 'I don't know'. "
    "Never fabricate information."
)
SYS_RAG = (
    "You are a concise assistant that answers questions about countries. "
    "Answer ONLY using the provided context. "
    "If the context doesn't contain the answer, say: 'I don't know'. "
    "Never fabricate information."
)


def call_nvidia(model_id, question, context="", use_rag=True):
    """NVIDIA NIM (OpenAI-uyumlu) chat.completions cagrisi.
       Doner: (text, latency_s, tokens_per_s, completion_tokens)"""
    if client is None:
        return "NVIDIA API kullanilabilir degil.", 0, 0, 0

    if use_rag and context:
        sys_msg = SYS_RAG
        usr_msg = (f"Context:\n{context[:2000]}\n\nQuestion: {question}\n"
                   f"Answer in 1-3 sentences using ONLY the context.")
    else:
        sys_msg = SYS_NO_RAG
        usr_msg = f"Question: {question}\nAnswer concisely in 1-3 sentences."

    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user",   "content": usr_msg}],
                temperature=0.0,
                max_tokens=160,
            )
            latency = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            comp_tokens = 0
            try:
                comp_tokens = resp.usage.completion_tokens
            except Exception:
                comp_tokens = len(text.split())
            tps = comp_tokens / latency if latency > 0 else 0
            return text, round(latency, 3), round(tps, 2), comp_tokens
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"API Hatasi: {e}", 0, 0, 0
    return "Yanit alinamadi.", 0, 0, 0

# ══════════════════════════════════════════════════════════════════════════════
# 4. HER MODEL İÇİN DEĞERLENDİRME (No-RAG + RAG)
# ══════════════════════════════════════════════════════════════════════════════
BEST_THR_FLOAT = float(BEST_THR) if BEST_THR not in ("None", None) else None
summary_rows = []
ood_all_rows = []

# Context'leri bir kez cek (tum modeller ayni context'i paylasir)
contexts = []
for idx in range(len(eval_questions)):
    ctx_res = coll.query(query_embeddings=[eq_embs[idx].tolist()], n_results=BEST_K,
                         include=["documents", "distances"])
    d_ret = ctx_res["documents"][0]
    d_dst = ctx_res["distances"][0]
    if BEST_THR_FLOAT is not None:
        d_ret = [d for d, dist in zip(d_ret, d_dst) if dist < BEST_THR_FLOAT]
    contexts.append("\n---\n".join(d_ret))

for mkey, model_id in NVIDIA_MODELS.items():
    print(f"\n[2/5] Model degerlendiriliyor: {mkey} ({model_id})")
    res_base, res_rag = [], []

    for idx, item in tqdm(enumerate(eval_questions), total=len(eval_questions), desc=mkey):
        q, gold = item["question"], item["ground_truth"]
        context = contexts[idx]

        # No-RAG
        a, lat, tps, tok = call_nvidia(model_id, q, use_rag=False)
        he = human_eval_proxy(a, gold, q)
        res_base.append({
            "question": q, "gold": gold[:200], "response": a[:300],
            "EM": exact_match(a, gold), "F1": round(calc_f1(a, gold), 4),
            "Hallucination": detect_hallucination(a, gold, ""),
            "Latency": lat, "Tokens/s": tps, "Comp_Tokens": tok,
            **{f"HE_{k}": v for k, v in he.items()},
        })
        time.sleep(RATE_LIMIT_S)

        # RAG
        a, lat, tps, tok = call_nvidia(model_id, q, context=context, use_rag=True)
        he = human_eval_proxy(a, gold, q)
        res_rag.append({
            "question": q, "gold": gold[:200], "response": a[:300],
            "EM": exact_match(a, gold), "F1": round(calc_f1(a, gold), 4),
            "Hallucination": detect_hallucination(a, gold, context),
            "Latency": lat, "Tokens/s": tps, "Comp_Tokens": tok,
            **{f"HE_{k}": v for k, v in he.items()},
        })
        time.sleep(RATE_LIMIT_S)

    df_base = pd.DataFrame(res_base)
    df_rag  = pd.DataFrame(res_rag)
    df_base.to_csv(os.path.join(OUTPUT_DIR, f"results_nvidia_{mkey}_base.csv"), index=False)
    df_rag.to_csv(os.path.join(OUTPUT_DIR, f"results_nvidia_{mkey}_rag.csv"),   index=False)

    def _cost(df):
        return round(df["Comp_Tokens"].sum() / 1e6 * COST_PER_1M_TOKENS, 6)

    for label, df in [("No RAG", df_base), ("RAG", df_rag)]:
        summary_rows.append({
            "Sistem":             f"NVIDIA {mkey} — {label}",
            "EM":                 round(df["EM"].mean(), 4),
            "F1":                 round(df["F1"].mean(), 4),
            "Hallucination_Rate": round(df["Hallucination"].mean(), 4),
            "Avg_Latency_s":      round(df["Latency"].mean(), 3),
            "Avg_Tokens/s":       round(df["Tokens/s"].mean(), 2),
            "Avg_Mem_Spike_MB":   0,
            "Total_Cost_USD":     _cost(df),
            "RAM_GB": 0, "VRAM_MB": 0,
            "HE_relevance":    round(df["HE_relevance"].mean(), 2),
            "HE_faithfulness": round(df["HE_faithfulness"].mean(), 2),
            "HE_fluency":      round(df["HE_fluency"].mean(), 2),
        })

    print(f"  [{mkey}] No-RAG F1={df_base['F1'].mean():.3f} | RAG F1={df_rag['F1'].mean():.3f} | "
          f"RAG EM={df_rag['EM'].mean():.3f}")

    # OOD testi (RAG modunda)
    if ood_questions:
        for j, sq in enumerate(ood_questions):
            ctx_r = coll.query(query_embeddings=[ood_embs[j].tolist()], n_results=BEST_K,
                               include=["documents"])
            ctx = "\n---\n".join(ctx_r["documents"][0])
            a, lat, tps, tok = call_nvidia(model_id, sq["question"], context=ctx, use_rag=True)
            refused = any(p in a.lower() for p in ["i don't know", "not available", "i cannot", "no information"])
            ood_all_rows.append({
                "Model": f"nvidia_{mkey}", "Soru": sq["question"][:100], "Cevap": a[:200],
                "Reddetme": refused, "Halucinasyon": (not refused) and len(a) > 10,
                "Latency": lat,
            })
            time.sleep(RATE_LIMIT_S)

# ══════════════════════════════════════════════════════════════════════════════
# 5. ÖZET + GÖRSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Ozet yaziliyor...")

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_nvidia.csv"), index=False)
print("\n--- NVIDIA Sonuclari ---")
print(df_summary[["Sistem", "EM", "F1", "Hallucination_Rate",
                  "Avg_Latency_s", "Avg_Tokens/s"]].to_string(index=False))

if ood_all_rows:
    df_ood = pd.DataFrame(ood_all_rows)
    df_ood.to_csv(os.path.join(OUTPUT_DIR, "results_nvidia_ood.csv"), index=False)
    print("\n--- OOD (Dogru Reddetme / Halucinasyon) ---")
    for m, sub in df_ood.groupby("Model"):
        print(f"  {m}: Reddetme={sub['Reddetme'].mean():.0%} | Halucinasyon={sub['Halucinasyon'].mean():.0%}")

# Gorsel: modeller x mod (EM, F1, Latency)
if not df_summary.empty:
    print("\n[4/5] Gorsel uretiliyor...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["EM", "F1", "Avg_Latency_s"]):
        bars = ax.barh(df_summary["Sistem"], df_summary[metric].astype(float),
                       color=plt.cm.viridis(np.linspace(0.2, 0.8, len(df_summary))))
        ax.set_title(metric)
        ax.invert_yaxis()
        for bar, val in zip(bars, df_summary[metric].astype(float)):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:.3f}", va="center", fontsize=8)
    plt.suptitle("NVIDIA API — Model x Mod Karsilastirmasi", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nvidia_comparison.png"), dpi=150)
    print("[OK] nvidia_comparison.png")
    plt.close()

print("\n" + "=" * 70)
print(" NVIDIA API TEST TAMAMLANDI!")
print(" Kaydedilen: results_nvidia_*.csv, summary_nvidia.csv")
print(" Final_Comparison.py artik NVIDIA sonuclarini da otomatik okuyacak.")
print("=" * 70)
