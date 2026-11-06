#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — Gemini API Test (§6 API LLM Zorunlu)
# ============================================================================
# PDF §6 "Faz 1 — API LLM: ≥1 API model" gereksinimini karşılar.
# Gemini 1.5 Flash modeli ile RAG ve No-RAG karşılaştırması yapar.
#
# Kaydedilen dosyalar → final_outputs/:
#   results_gemini_base.csv   — No-RAG sonuçları
#   results_gemini_rag.csv    — RAG sonuçları
#   summary_gemini.csv        — Final_Comparison.py tarafından okunur
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q google-generativeai chromadb sentence-transformers
# !pip install -q langchain_text_splitters tqdm pandas matplotlib seaborn
#
# Kaggle Secrets: GEMINI_API_KEY tanimli olmalidir.
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
CHROMA_DIR  = "./chroma_db_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gemini API modeli seçimi
GEMINI_MODEL  = "gemini-1.5-flash"    # Hızlı ve ucuz
RATE_LIMIT_S  = 1.0                   # Saniyede max 1 istek (ücretsiz tier)
MAX_EVAL_Q    = 60                     # API maliyet kontrolü için sınır

print("=" * 70)
print(f" FINAL — GEMINI API TEST ({GEMINI_MODEL})")
print("=" * 70)

# ── Gemini SDK Kurulumu ───────────────────────────────────────────────────────
GEMINI_API_KEY = None
genai = None
gemini_model_obj = None

try:
    from kaggle_secrets import UserSecretsClient
    GEMINI_API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")
    print("[INFO] GEMINI_API_KEY alindi.")
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        print("[INFO] GEMINI_API_KEY ortam degiskeninden alindi.")
    else:
        print("[UYARI] GEMINI_API_KEY Secrets veya ortam degiskeninden alinamadi.")

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai_sdk
        genai_sdk.configure(api_key=GEMINI_API_KEY)
        gemini_model_obj = genai_sdk.GenerativeModel(GEMINI_MODEL)
        genai = genai_sdk
        print(f"[OK] Gemini modeli hazir: {GEMINI_MODEL}")
    except Exception as e:
        print(f"[HATA] Gemini SDK: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ + CHROMADB (pipeline_config.json'dan en iyi koleksiyon okunur)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Veri + ChromaDB hazirlaniyor...")

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
            mc, sc = ps[0].strip(), (ps[1].strip() if len(ps) > 1 else "")
            secs.setdefault(mc, []).append(f"{sc}: {v}" if sc else str(v))
        for cat, ls in secs.items():
            txt += f"=== {cat} ===\n" + "".join(f"- {l}\n" for l in ls) + "\n"
        docs.append({"country": country, "text": txt, "metadata": {"country": country}})
    return docs, data

docs, raw_data = load_data(INPUT_JSON)

# Chunking (sentence_based — Final pipeline ile ayni)
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
del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

COLL_NAME = f"final_{BEST_EMB_KEY}_{BEST_CHUNK_KEY}"
_cdb = chromadb.PersistentClient(path="./chroma_eval_gemini")
try: _cdb.delete_collection(COLL_NAME)
except: pass
coll = _cdb.create_collection(name=COLL_NAME)
for i in range(0, len(chunks), 5000):
    end = min(i + 5000, len(chunks))
    coll.add(
        embeddings=embeddings[i:end].tolist(),
        documents=[c["text"] for c in chunks[i:end]],
        metadatas=[c["metadata"] for c in chunks[i:end]],
        ids=[f"gemini_{i+j}" for j in range(end - i)],
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

# Query embeddingleri hesapla
device = "cuda:0" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
eq_embs = st_model.encode([q["question"] for q in eval_questions], batch_size=128)
del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 2. METRİK FONKSİYONLARI
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

# ══════════════════════════════════════════════════════════════════════════════
# 3. GEMİNİ API ÇAĞRI FONKSİYONU
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

def call_gemini(question, context="", use_rag=True):
    """Gemini API'ye istek gönderir. Rate limit ve hata yönetimi içerir."""
    if gemini_model_obj is None:
        return "Gemini API kullanilabilir degil.", 0, 0

    if use_rag and context:
        prompt = (
            f"{SYS_RAG}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Question: {question}\n"
            f"Answer in 1-3 sentences using ONLY the context."
        )
    else:
        prompt = f"{SYS_NO_RAG}\n\nQuestion: {question}\nAnswer concisely in 1-3 sentences."

    for attempt in range(3):
        try:
            t0 = time.time()
            response = gemini_model_obj.generate_content(prompt)
            latency = time.time() - t0
            text = response.text.strip() if hasattr(response, "text") else "I don't know."
            # Tahmini token/s (Gemini token sayısını açıklamıyor, kelime tabanlı)
            word_count = len(text.split())
            # Maliyet: Gemini 1.5 Flash ~$0.075/1M input token, ~$0.30/1M output
            cost_approx = round(len(prompt.split()) * 0.00000075 + word_count * 0.0000003, 6)
            return text, round(latency, 3), cost_approx
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"API Hatasi: {e}", 0, 0

    return "Yanit alinamadi.", 0, 0

# ══════════════════════════════════════════════════════════════════════════════
# 4. DEĞERLENDİRME — No-RAG ve RAG
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/6] Gemini Degerlendirme ({len(eval_questions)} soru x 2 mod)...")

BEST_THR_FLOAT = float(BEST_THR) if BEST_THR not in ("None", None) else None

results_base = []
results_rag  = []

for idx, item in tqdm(enumerate(eval_questions), total=len(eval_questions), desc="Gemini Eval"):
    q    = item["question"]
    gold = item["ground_truth"]

    # Context al (threshold uygula)
    q_emb = eq_embs[idx].tolist()
    ctx_res = coll.query(query_embeddings=[q_emb], n_results=BEST_K,
                          include=["documents", "metadatas", "distances"])
    docs_ret = ctx_res["documents"][0]
    dists    = ctx_res["distances"][0]
    if BEST_THR_FLOAT is not None:
        docs_ret = [d for d, dist in zip(docs_ret, dists) if dist < BEST_THR_FLOAT]
    context = "\n---\n".join(docs_ret)

    # No-RAG
    ans_base, lat_base, cost_base = call_gemini(q, use_rag=False)
    em_b  = exact_match(ans_base, gold)
    f1_b  = calc_f1(ans_base, gold)
    he_b  = human_eval_proxy(ans_base, gold, q)
    results_base.append({
        "question": q, "gold": gold[:200], "response": ans_base[:300],
        "EM": em_b, "F1": round(f1_b, 4),
        "Latency": lat_base, "Cost_USD": cost_base,
        **{f"HE_{k}": v for k, v in he_b.items()},
    })

    # RAG
    ans_rag, lat_rag, cost_rag = call_gemini(q, context=context, use_rag=True)
    em_r  = exact_match(ans_rag, gold)
    f1_r  = calc_f1(ans_rag, gold)
    he_r  = human_eval_proxy(ans_rag, gold, q)
    results_rag.append({
        "question": q, "gold": gold[:200], "response": ans_rag[:300],
        "EM": em_r, "F1": round(f1_r, 4),
        "Latency": lat_rag, "Cost_USD": cost_rag,
        **{f"HE_{k}": v for k, v in he_r.items()},
    })

    time.sleep(RATE_LIMIT_S)   # Rate limit

df_base = pd.DataFrame(results_base)
df_rag  = pd.DataFrame(results_rag)
df_base.to_csv(os.path.join(OUTPUT_DIR, "results_gemini_base.csv"), index=False)
df_rag.to_csv(os.path.join(OUTPUT_DIR, "results_gemini_rag.csv"),   index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5. OOD TEST (Gemini ile)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] OOD Test (Gemini)...")

stress_path = os.path.join(OUTPUT_DIR, "stress_test_questions.json")
ood_questions = []
if os.path.exists(stress_path):
    with open(stress_path) as f:
        stress_data = json.load(f)
    ood_questions = stress_data.get("ood", [])

ood_results = []
for sq in ood_questions:
    q_emb_enc = SentenceTransformer(EMB_MODEL_NAME)
    qe = q_emb_enc.encode([sq["question"]])[0].tolist()
    del q_emb_enc

    ctx_r = coll.query(query_embeddings=[qe], n_results=BEST_K, include=["documents"])
    ctx   = "\n---\n".join(ctx_r["documents"][0])

    ans, lat, cost = call_gemini(sq["question"], context=ctx, use_rag=True)
    refused = any(p in ans.lower() for p in ["i don't know", "not available", "i cannot"])
    ood_results.append({
        "Soru": sq["question"][:100], "Cevap": ans[:200],
        "Reddetme": refused, "Halucinasyon": not refused and len(ans) > 10,
        "Latency": lat, "Cost_USD": cost,
    })
    time.sleep(RATE_LIMIT_S)

if ood_results:
    df_ood = pd.DataFrame(ood_results)
    df_ood.to_csv(os.path.join(OUTPUT_DIR, "results_gemini_ood.csv"), index=False)
    ref_r  = df_ood["Reddetme"].mean()
    hall_r = df_ood["Halucinasyon"].mean()
    print(f"  OOD Dogru Reddetme : {ref_r:.0%}")
    print(f"  OOD Halucinasyon   : {hall_r:.0%}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ÖZET + GÖRSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Ozet hesaplaniyor...")

total_cost_base = df_base["Cost_USD"].sum()
total_cost_rag  = df_rag["Cost_USD"].sum()

summary = pd.DataFrame([
    {
        "Sistem":           "Gemini Flash — No RAG",
        "EM":               round(df_base["EM"].mean(), 4),
        "F1":               round(df_base["F1"].mean(), 4),
        "Hallucination_Rate": 0,
        "Avg_Latency_s":    round(df_base["Latency"].mean(), 3),
        "Avg_Tokens/s":     0,
        "Avg_Mem_Spike_MB": 0,
        "Total_Cost_USD":   round(total_cost_base, 4),
        "RAM_GB": 0, "VRAM_MB": 0,
        "HE_relevance":    round(df_base["HE_relevance"].mean(), 2),
        "HE_faithfulness": round(df_base["HE_faithfulness"].mean(), 2),
        "HE_fluency":      round(df_base["HE_fluency"].mean(), 2),
    },
    {
        "Sistem":           "Gemini Flash — RAG",
        "EM":               round(df_rag["EM"].mean(), 4),
        "F1":               round(df_rag["F1"].mean(), 4),
        "Hallucination_Rate": 0,
        "Avg_Latency_s":    round(df_rag["Latency"].mean(), 3),
        "Avg_Tokens/s":     0,
        "Avg_Mem_Spike_MB": 0,
        "Total_Cost_USD":   round(total_cost_rag, 4),
        "RAM_GB": 0, "VRAM_MB": 0,
        "HE_relevance":    round(df_rag["HE_relevance"].mean(), 2),
        "HE_faithfulness": round(df_rag["HE_faithfulness"].mean(), 2),
        "HE_fluency":      round(df_rag["HE_fluency"].mean(), 2),
    },
])

summary.to_csv(os.path.join(OUTPUT_DIR, "summary_gemini.csv"), index=False)

print("\n--- Gemini Sonuclari ---")
print(summary[["Sistem", "EM", "F1", "Avg_Latency_s", "Total_Cost_USD"]].to_string(index=False))

# Gorsel: No-RAG vs RAG
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ["EM", "F1", "Avg_Latency_s"]
labels  = ["EM", "F1", "Latency (s)"]
colors  = ["#3498db", "#2ecc71"]
for idx, (m, label) in enumerate(zip(metrics, labels)):
    ax = axes[idx]
    vals = summary[m].astype(float)
    bars = ax.bar(summary["Sistem"], vals, color=colors)
    ax.set_title(label)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.suptitle(f"Gemini {GEMINI_MODEL} — No-RAG vs RAG Karsilastirma", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gemini_comparison.png"), dpi=150)
print("[OK] gemini_comparison.png kaydedildi")
plt.close()

# Maliyet analizi
print(f"\n  Toplam API Maliyeti (No-RAG) : ~${total_cost_base:.4f}")
print(f"  Toplam API Maliyeti (RAG)    : ~${total_cost_rag:.4f}")
print(f"  Toplam                       : ~${total_cost_base + total_cost_rag:.4f}")
print(f"  Sorgu basina (RAG)           : ~${total_cost_rag / max(len(eval_questions), 1):.6f}")

print("\n" + "=" * 70)
print(" GEMINI API TEST TAMAMLANDI!")
print(" Kaydedilen: results_gemini_*.csv, summary_gemini.csv")
print(" Final_Comparison.py artik Gemini sonuclarini da okuyacak.")
print("=" * 70)
