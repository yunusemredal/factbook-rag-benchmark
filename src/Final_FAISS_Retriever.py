#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — FAISS Retriever (PDF §5.3 İkinci Vektör Veritabanı)
# ============================================================================
# PDF §5.3 "FAISS, Chroma, Qdrant" örnek olarak veriliyor.
# Bu script FAISS ile ChromaDB'yi 4 farklı index tipi üzerinde karşılaştırır.
#
# FAISS Index Tipleri:
#   IndexFlatL2    — Brüt kuvvet, kesin sonuç, yavaş büyük verilerde
#   IndexFlatIP    — Inner Product (cosine için normalize et), kesin
#   IndexIVFFlat   — Yaklaşık, inverted file index, hızlı
#   IndexHNSWFlat  — Hierarchical Navigable Small World, çok hızlı + doğru
#
# Kaydedilen dosyalar → final_outputs/:
#   faiss_vs_chroma_comparison.csv
#   faiss_index_comparison.png
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q faiss-cpu sentence-transformers chromadb tqdm pandas matplotlib seaborn
# (GPU varsa: !pip install -q faiss-gpu)
# ============================================================================

import json, os, time, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import chromadb

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
INPUT_JSON = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K_VALUES = [1, 7, 15]   # Final pipeline ile aynı

print("=" * 70)
print(" FINAL — FAISS vs ChromaDB Karşılaştırması")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE KONFİGÜRASYONUNDAN EN İYİ EMBEDDİNG OKU
# ══════════════════════════════════════════════════════════════════════════════
cfg_path = os.path.join(OUTPUT_DIR, "pipeline_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        pcfg = json.load(f)
    BEST_EMB_KEY   = pcfg.get("best_emb_key",   "e5_base")
    BEST_CHUNK_KEY = pcfg.get("best_chunk_key",  "sentence_based")
    print(f"[INFO] En iyi embedding: {BEST_EMB_KEY}, chunking: {BEST_CHUNK_KEY}")
else:
    BEST_EMB_KEY, BEST_CHUNK_KEY = "e5_base", "sentence_based"
    print("[UYARI] pipeline_config.json yok — varsayilan kullanildi.")

EMBEDDING_MODELS_MAP = {
    "e5_base":            "intfloat/e5-base-v2",
    "bge_base":           "BAAI/bge-base-en-v1.5",
    "gte_base":           "thenlper/gte-base",
    "multilingual_mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
EMB_MODEL_NAME = EMBEDDING_MODELS_MAP.get(BEST_EMB_KEY, "intfloat/e5-base-v2")

# ══════════════════════════════════════════════════════════════════════════════
# 2. VERİ + CHUNKING + EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Veri yukleniyor ve embeddingler hesaplaniyor...")

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
                    out.append({"text": f"Country: {d['country']}\n{buf.strip()}",
                                "metadata": d["metadata"]})
                buf = sent
        if len(buf) >= min_chars:
            out.append({"text": f"Country: {d['country']}\n{buf.strip()}",
                        "metadata": d["metadata"]})
    return out

chunks = chunk_sentence_based(docs)
texts  = [c["text"] for c in chunks]
metas  = [c["metadata"] for c in chunks]
print(f"[INFO] {len(chunks)} chunk hazir.")

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(EMB_MODEL_NAME, device=device)

print(f"[INFO] Embeddingler hesaplaniyor: {EMB_MODEL_NAME}...")
t0 = time.time()
embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
enc_time = time.time() - t0
DIM = embeddings.shape[1]
print(f"[INFO] {len(embeddings)} embedding | dim={DIM} | sure={enc_time:.1f}s")
del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Eval set yukle ve encode et
eval_path = os.path.join(OUTPUT_DIR, "eval_set_extended.json")
if not os.path.exists(eval_path):
    eval_path = _resolve("eval_set.json", "/kaggle/input/datasets/yunusemredal/cfbook/eval_set.json")
with open(eval_path, "r", encoding="utf-8") as f:
    eval_questions = json.load(f)
print(f"[INFO] Eval set: {len(eval_questions)} soru")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
st_model = SentenceTransformer(EMB_MODEL_NAME, device=device)
eq_embs = st_model.encode([q["question"] for q in eval_questions],
                           batch_size=128, normalize_embeddings=True)
del st_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 3. METRİK FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_retriever_generic(query_fn, eval_set, query_embs, top_k=7):
    """
    query_fn(q_emb, k) -> (docs: List[str], metas: List[dict])
    """
    hit = 0; mrr = 0.0; lat = 0.0
    for idx, item in enumerate(eval_set):
        expected = item["ground_truth"].lower()
        target   = item.get("metadata", {}).get("country", "")
        q_emb    = query_embs[idx]

        t0 = time.time()
        docs_ret, metas_ret = query_fn(q_emb, top_k)
        lat += time.time() - t0

        rank = 0
        found = False
        for i, (doc, meta) in enumerate(zip(docs_ret, metas_ret)):
            if expected[:50] in doc.lower() and meta.get("country") == target:
                if not found:
                    rank = i + 1
                found = True
        if found:
            hit += 1
            mrr += 1.0 / rank

    N = max(1, len(eval_set))
    return {
        "Hit Rate@k":       round(hit / N, 4),
        "MRR@k":            round(mrr / N, 4),
        "Recall@k":         round(hit / N, 4),
        "Avg Latency (ms)": round(lat / N * 1000, 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 4. FAISS INDEX TİPLERİ KURU
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] FAISS index'leri kuruluyor...")

try:
    import faiss
    FAISS_AVAILABLE = True
    print(f"[OK] FAISS {faiss.__version__} yuklendi.")
except ImportError:
    FAISS_AVAILABLE = False
    print("[UYARI] faiss-cpu yuklu degil. 'pip install faiss-cpu' ile kurun.")

faiss_results = []
embs_float32 = embeddings.astype(np.float32)

if FAISS_AVAILABLE:
    # (1) IndexFlatIP — Kesin, Inner Product (normalize edilmiş → cosine)
    print("\n--- IndexFlatIP (Kesin, Cosine) ---")
    t_build = time.time()
    idx_flat = faiss.IndexFlatIP(DIM)
    idx_flat.add(embs_float32)
    build_time = time.time() - t_build
    print(f"  Build: {build_time:.2f}s | {idx_flat.ntotal} vektor")

    def query_flat_ip(q_emb, k):
        qv = np.array([q_emb], dtype=np.float32)
        D, I = idx_flat.search(qv, k)
        docs_r = [texts[i] for i in I[0] if i < len(texts)]
        metas_r = [metas[i] for i in I[0] if i < len(metas)]
        return docs_r, metas_r

    for k in TOP_K_VALUES:
        res = evaluate_retriever_generic(query_flat_ip, eval_questions, eq_embs, top_k=k)
        res.update({"DB": "FAISS", "Index": "IndexFlatIP", "Top-K": k,
                    "Build_Time_s": round(build_time, 2), "Exact": True})
        faiss_results.append(res)
        print(f"  k={k}: Hit Rate={res['Hit Rate@k']:.4f} | MRR={res['MRR@k']:.4f} | Lat={res['Avg Latency (ms)']:.2f}ms")

    # (2) IndexFlatL2 — Kesin, L2 mesafesi
    print("\n--- IndexFlatL2 (Kesin, L2) ---")
    t_build = time.time()
    idx_l2 = faiss.IndexFlatL2(DIM)
    idx_l2.add(embs_float32)
    build_time = time.time() - t_build
    print(f"  Build: {build_time:.2f}s")

    def query_flat_l2(q_emb, k):
        qv = np.array([q_emb], dtype=np.float32)
        D, I = idx_l2.search(qv, k)
        docs_r = [texts[i] for i in I[0] if i < len(texts)]
        metas_r = [metas[i] for i in I[0] if i < len(metas)]
        return docs_r, metas_r

    for k in TOP_K_VALUES:
        res = evaluate_retriever_generic(query_flat_l2, eval_questions, eq_embs, top_k=k)
        res.update({"DB": "FAISS", "Index": "IndexFlatL2", "Top-K": k,
                    "Build_Time_s": round(build_time, 2), "Exact": True})
        faiss_results.append(res)
        print(f"  k={k}: Hit Rate={res['Hit Rate@k']:.4f} | MRR={res['MRR@k']:.4f} | Lat={res['Avg Latency (ms)']:.2f}ms")

    # (3) IndexIVFFlat — Yaklaşık, Inverted File
    print("\n--- IndexIVFFlat (Yaklasik, IVF) ---")
    nlist = min(100, len(chunks) // 10)   # Küme sayısı
    quantizer = faiss.IndexFlatIP(DIM)
    idx_ivf = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    t_build = time.time()
    idx_ivf.train(embs_float32)
    idx_ivf.add(embs_float32)
    idx_ivf.nprobe = 10   # Arama sırasında kontrol edilecek küme sayısı
    build_time = time.time() - t_build
    print(f"  Build+Train: {build_time:.2f}s | nlist={nlist} | nprobe=10")

    def query_ivf(q_emb, k):
        qv = np.array([q_emb], dtype=np.float32)
        D, I = idx_ivf.search(qv, k)
        docs_r = [texts[i] for i in I[0] if 0 <= i < len(texts)]
        metas_r = [metas[i] for i in I[0] if 0 <= i < len(metas)]
        return docs_r, metas_r

    for k in TOP_K_VALUES:
        res = evaluate_retriever_generic(query_ivf, eval_questions, eq_embs, top_k=k)
        res.update({"DB": "FAISS", "Index": "IndexIVFFlat", "Top-K": k,
                    "Build_Time_s": round(build_time, 2), "Exact": False})
        faiss_results.append(res)
        print(f"  k={k}: Hit Rate={res['Hit Rate@k']:.4f} | MRR={res['MRR@k']:.4f} | Lat={res['Avg Latency (ms)']:.2f}ms")

    # (4) IndexHNSWFlat — HNSW (hızlı + doğru yaklaşım)
    print("\n--- IndexHNSWFlat (HNSW) ---")
    M_hnsw = 32   # Graf komşu sayısı (yüksek = daha doğru, daha yavaş build)
    idx_hnsw = faiss.IndexHNSWFlat(DIM, M_hnsw, faiss.METRIC_INNER_PRODUCT)
    idx_hnsw.hnsw.efConstruction = 200   # Build kalitesi
    idx_hnsw.hnsw.efSearch = 50          # Arama kalitesi
    t_build = time.time()
    idx_hnsw.add(embs_float32)
    build_time = time.time() - t_build
    print(f"  Build: {build_time:.2f}s | M={M_hnsw} | efC=200 | efS=50")

    def query_hnsw(q_emb, k):
        qv = np.array([q_emb], dtype=np.float32)
        D, I = idx_hnsw.search(qv, k)
        docs_r = [texts[i] for i in I[0] if 0 <= i < len(texts)]
        metas_r = [metas[i] for i in I[0] if 0 <= i < len(metas)]
        return docs_r, metas_r

    for k in TOP_K_VALUES:
        res = evaluate_retriever_generic(query_hnsw, eval_questions, eq_embs, top_k=k)
        res.update({"DB": "FAISS", "Index": "IndexHNSWFlat", "Top-K": k,
                    "Build_Time_s": round(build_time, 2), "Exact": False})
        faiss_results.append(res)
        print(f"  k={k}: Hit Rate={res['Hit Rate@k']:.4f} | MRR={res['MRR@k']:.4f} | Lat={res['Avg Latency (ms)']:.2f}ms")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CHROMADB REFERANS (Karşılaştırma için)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] ChromaDB referans olcumu...")

_cdb = chromadb.PersistentClient(path="./chroma_faiss_compare")
COLL_NAME = f"faiss_compare_{BEST_EMB_KEY}"
try: _cdb.delete_collection(COLL_NAME)
except: pass
coll = _cdb.create_collection(name=COLL_NAME)
for i in range(0, len(chunks), 5000):
    end = min(i + 5000, len(chunks))
    coll.add(
        embeddings=embs_float32[i:end].tolist(),
        documents=texts[i:end],
        metadatas=metas[i:end],
        ids=[f"cf_{i+j}" for j in range(end - i)],
    )
print(f"[INFO] ChromaDB: {coll.count()} chunk")

chroma_results = []
for k in TOP_K_VALUES:
    def query_chroma(q_emb, k=k):
        res = coll.query(query_embeddings=[q_emb.tolist()], n_results=k,
                         include=["documents", "metadatas"])
        return res["documents"][0], res["metadatas"][0]

    res = evaluate_retriever_generic(query_chroma, eval_questions, eq_embs, top_k=k)
    res.update({"DB": "ChromaDB", "Index": "ChromaDB (HNSW)", "Top-K": k,
                "Build_Time_s": enc_time, "Exact": False})
    chroma_results.append(res)
    print(f"  k={k}: Hit Rate={res['Hit Rate@k']:.4f} | MRR={res['MRR@k']:.4f} | Lat={res['Avg Latency (ms)']:.2f}ms")

# ══════════════════════════════════════════════════════════════════════════════
# 6. KARŞILAŞTIRMA + GÖRSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Karsilastirma tablosu olusturuluyor...")

all_results = faiss_results + chroma_results
df = pd.DataFrame(all_results)
df.to_csv(os.path.join(OUTPUT_DIR, "faiss_vs_chroma_comparison.csv"), index=False)
print(f"[INFO] faiss_vs_chroma_comparison.csv kaydedildi ({len(df)} satir)")

# k=7 icin ozet
df_k7 = df[df["Top-K"] == 7].copy()
print("\n--- k=7 Karsilastirmasi ---")
print(df_k7[["DB", "Index", "Hit Rate@k", "MRR@k",
              "Avg Latency (ms)", "Build_Time_s", "Exact"]].to_string(index=False))

# Gorseller
if not df_k7.empty:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) Hit Rate karsilastirma (k=7)
    ax = axes[0, 0]
    db_colors = {"FAISS": "#3498db", "ChromaDB": "#e74c3c"}
    for db_name, grp in df_k7.groupby("DB"):
        ax.barh(grp["Index"], grp["Hit Rate@k"],
                color=db_colors.get(db_name, "gray"),
                label=db_name, alpha=0.8)
    ax.set_title("Hit Rate@7 Karşılaştırması")
    ax.set_xlabel("Hit Rate@7")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Latency karsilastirma
    ax = axes[0, 1]
    bars = ax.barh(df_k7["Index"], df_k7["Avg Latency (ms)"],
                   color=[db_colors.get(r["DB"], "gray") for _, r in df_k7.iterrows()])
    ax.set_title("Ortalama Sorgu Latency (k=7)")
    ax.set_xlabel("Latency (ms)")
    for bar, val in zip(bars, df_k7["Avg Latency (ms)"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}ms", va="center", fontsize=9)

    # (3) Top-K etkisi (her index tipi)
    ax = axes[1, 0]
    for index_name, grp in df.groupby("Index"):
        ax.plot(grp["Top-K"], grp["Hit Rate@k"], marker="o", label=index_name)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Hit Rate@k")
    ax.set_title("Top-K Etkisi (Tüm Index Tipleri)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (4) Build Time vs Hit Rate trade-off
    ax = axes[1, 1]
    for _, row in df_k7.iterrows():
        color = db_colors.get(row["DB"], "gray")
        ax.scatter(row["Build_Time_s"], row["Hit Rate@k"],
                   s=200, c=color, alpha=0.8, edgecolors="black", linewidths=1)
        ax.annotate(row["Index"].split("Index")[-1] or row["Index"],
                    (row["Build_Time_s"], row["Hit Rate@k"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Index Build Süresi (s)")
    ax.set_ylabel("Hit Rate@7")
    ax.set_title("Build Time vs Hit Rate Trade-off")
    ax.grid(True, alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=db) for db, c in db_colors.items()]
    ax.legend(handles=legend_elems)

    plt.suptitle(f"FAISS vs ChromaDB Karşılaştırması ({BEST_EMB_KEY})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "faiss_index_comparison.png"), dpi=150)
    print("\n[OK] faiss_index_comparison.png kaydedildi")
    plt.close()

# Analiz ozeti
print("\n--- FAISS vs ChromaDB Analiz Ozeti ---")
if not df_k7.empty:
    best_hit   = df_k7.loc[df_k7["Hit Rate@k"].idxmax()]
    fastest    = df_k7.loc[df_k7["Avg Latency (ms)"].idxmin()]
    print(f"  En Yuksek Hit Rate : {best_hit['Index']} ({best_hit['Hit Rate@k']:.4f})")
    print(f"  En Hizli Sorgu     : {fastest['Index']} ({fastest['Avg Latency (ms)']:.2f}ms)")
    print("\n  Yorum:")
    print("  - HNSW en iyi latency/dogruluk dengesini sunar")
    print("  - IVFFlat buyuk koleksiyonlarda olceklenir")
    print("  - FlatIP/FlatL2 kucuk veri seti icin ideal (kesin sonuc)")
    print("  - ChromaDB varsayilan HNSW kullanir, benzer performans gosterir")

print("\n" + "=" * 70)
print(" FAISS vs CHROMADB KARSILASTIRMASI TAMAMLANDI!")
print(f" Sonuc: {OUTPUT_DIR}/faiss_vs_chroma_comparison.csv")
print(" Bu analizi raporun §5.3 Vektor Veritabani bolumunde kullanin.")
print("=" * 70)
