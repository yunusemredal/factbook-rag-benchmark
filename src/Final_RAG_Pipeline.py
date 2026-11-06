#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL RAG PIPELINE — Gelismis Sistem
# ============================================================================
# Vize'den FARKLI yeni bilesenleri kapsar:
#   Chunking  : sliding_window, sentence_based, hierarchical
#   Embedding : e5_base, bge_base, gte_base, multilingual_mpnet
#   Retrieval : k=1,7,15  +  Similarity Threshold Filtering
#   Re-ranking: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
#   Stress    : paraphrase, multi_hop, ambiguous, long_noisy, ood  (5 kategori)
#   Eval Set  : >= 120 soru
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q chromadb sentence-transformers langchain_text_splitters
# !pip install -q tqdm pandas matplotlib seaborn transformers
#
# Calisma sirasi:
#   1. Bu dosyayi calistirin  -> final_outputs/ ve chroma_db_final/ olusur
#   2. Fine-tuning scriptlerini calistirin (Llama3 / Qwen)
#   3. Final_Model_Evaluator.py her model icin ayri calistirin
#   4. Final_Comparison.py ile sonuclari birlestirin
# ============================================================================

import json, os, time, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chromadb
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import Counter

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Veri Yolu Cozumleyici ─────────────────────────────────────────────────────
# Kaggle datasetleri /kaggle/input/<slug>/ altina baglanir. Slug adi degisebildigi
# icin dosyayi olasi konumlarda arar; bulamazsa /kaggle/input agacinda glob yapar.
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
INPUT_JSON    = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
EVAL_SET_FILE = _resolve("eval_set.json",  "/kaggle/input/datasets/yunusemredal/cfbook/eval_set.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
CHROMA_DIR    = "./chroma_db_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vizeden FARKLI top-k degerleri (vize: 3,5,10)
TOP_K_VALUES = [1, 7, 15]

# Similarity threshold denemeleri (kucuk distance = daha benzer)
THRESHOLDS = [None, 0.5, 0.7, 0.9]

# Vizeden FARKLI embedding modelleri
EMBEDDING_MODELS = {
    "e5_base":           "intfloat/e5-base-v2",
    "bge_base":          "BAAI/bge-base-en-v1.5",
    "gte_base":          "thenlper/gte-base",
    "multilingual_mpnet":"sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

EMBED_BATCH = 256

print("=" * 70)
print(" FINAL RAG PIPELINE — BASLATILIYOR")
print("=" * 70)
pipeline_start = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 1. ASAMA: VERI HAZIRLIGI")
print("=" * 60)


def load_and_prepare_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Toplam Ulke: {len(data)}")

    docs = []
    for country, details in data.items():
        txt = f"Country Name: {country}\n\n"
        secs = {}
        for key, val in details.items():
            if key == "url":
                continue
            parts = key.split(":", 1)
            mc = parts[0].strip()
            sc = parts[1].strip() if len(parts) > 1 else ""
            secs.setdefault(mc, []).append(f"{sc}: {val}" if sc else str(val))
        for cat, lines in secs.items():
            txt += f"=== {cat} ===\n" + "".join(f"- {l}\n" for l in lines) + "\n"
        docs.append({"country": country, "text": txt, "metadata": {"country": country}})

    total_chars = sum(len(d["text"]) for d in docs)
    print(f"[INFO] Toplam Karakter : {total_chars:,}")
    print(f"[INFO] Tahmini Sayfa   : ~{total_chars // 2000}")
    return docs, data


docs, raw_data = load_and_prepare_data(INPUT_JSON)

# ══════════════════════════════════════════════════════════════════════════════
# 2. YENİ CHUNKING STRATEJİLERİ (Vizeden Farklı)
# ══════════════════════════════════════════════════════════════════════════════
# Vize : fixed(500/50), section_based, paragraph_based
# Final: sliding_window(300/150), sentence_based, hierarchical
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 2. ASAMA: YENİ CHUNKING STRATEJİLERİ")
print("=" * 60)


def chunk_sliding_window(docs, chunk_size=300, step=150):
    """Kaydirmalı pencere — kucuk chunk, buyuk overlap (vizeden farkli parametreler)."""
    all_chunks = []
    for d in docs:
        text = d["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if len(piece) >= 50:
                all_chunks.append({
                    "text": f"Country: {d['country']}\n{piece}",
                    "metadata": d["metadata"],
                })
            if end == len(text):
                break
            start += step
    return all_chunks


def chunk_sentence_based(docs, max_chars=600, min_chars=80):
    """Cumle sinirına gore chunking (noktalama isaretleri kullanilir)."""
    sent_re = re.compile(r"(?<=[.!?])\s+")
    all_chunks = []
    for d in docs:
        sentences = sent_re.split(d["text"])
        buffer = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(buffer) + len(sent) <= max_chars:
                buffer += " " + sent
            else:
                if len(buffer) >= min_chars:
                    all_chunks.append({
                        "text": f"Country: {d['country']}\n{buffer.strip()}",
                        "metadata": d["metadata"],
                    })
                buffer = sent
        if len(buffer) >= min_chars:
            all_chunks.append({
                "text": f"Country: {d['country']}\n{buffer.strip()}",
                "metadata": d["metadata"],
            })
    return all_chunks


def chunk_hierarchical(docs):
    """İki seviyeli hiyerarsik chunking:
       Seviye 1 — Ulke ozeti (ilk 400 karakter)
       Seviye 2 — Her bolum ayri chunk (section detayi)
    """
    all_chunks = []
    for d in docs:
        # Seviye 1: ozet
        summary = d["text"][:400].strip()
        all_chunks.append({
            "text": f"[SUMMARY] Country: {d['country']}\n{summary}",
            "metadata": {**d["metadata"], "level": "summary"},
        })
        # Seviye 2: bolum detaylari
        secs = d["text"].split("===")
        for i in range(1, len(secs), 2):
            if i + 1 < len(secs):
                detail = (
                    f"[DETAIL] Country: {d['country']}\n"
                    f"=== {secs[i].strip()} ===\n"
                    f"{secs[i+1].strip()}"
                )
                all_chunks.append({
                    "text": detail,
                    "metadata": {**d["metadata"], "level": "detail"},
                })
    return all_chunks


chunks_sliding    = chunk_sliding_window(docs)
chunks_sentence   = chunk_sentence_based(docs)
chunks_hierarchical = chunk_hierarchical(docs)

chunking_strats = {
    "sliding_window": chunks_sliding,
    "sentence_based": chunks_sentence,
    "hierarchical":   chunks_hierarchical,
}

print("\n--- Final Chunking Istatistikleri ---")
for name, chunks in chunking_strats.items():
    avg = np.mean([len(c["text"]) for c in chunks])
    print(f"  {name:>18s}: {len(chunks):>6d} chunk  |  Ort. {avg:.0f} karakter")

# ══════════════════════════════════════════════════════════════════════════════
# 3. YENİ EMBEDDING MODELLERİ + CHROMADB
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 3. ASAMA: YENİ EMBEDDING MODELLERİ + CHROMADB")
print("=" * 60)

import torch
client = chromadb.PersistentClient(path=CHROMA_DIR)
collections_map   = {}   # coll_name -> {collection, emb_name, chunk_name}
embedding_times   = {}   # emb_key   -> timing dict
query_embs_cache  = {}   # emb_key   -> np.ndarray (n_eval × dim)
stress_embs_cache = {}   # emb_key   -> np.ndarray (n_stress × dim)

# Eval sorularini yukle (on-encode icin lazim)
eval_questions = []
if os.path.exists(EVAL_SET_FILE):
    with open(EVAL_SET_FILE, "r", encoding="utf-8") as f:
        eval_questions = json.load(f)
    print(f"[INFO] Mevcut eval set: {len(eval_questions)} soru")

# Stress test sorulari (5 kategori — long_noisy eklendi)
stress_test_questions = {
    "paraphrase": [
        {"question": "Tell me the capital of France.", "expected_country": "France",
         "expected_key": "Government: Capital - name"},
        {"question": "Which city serves as Germany's seat of government?", "expected_country": "Germany",
         "expected_key": "Government: Capital - name"},
        {"question": "Can you tell me how many people live in Japan?", "expected_country": "Japan",
         "expected_key": "People and Society: Population - total"},
        {"question": "I'd like to know about the weather patterns in Brazil.", "expected_country": "Brazil",
         "expected_key": "Geography: Climate"},
        {"question": "Describe the geographical landscape and terrain of Switzerland.",
         "expected_country": "Switzerland", "expected_key": "Geography: Terrain"},
    ],
    "multi_hop": [
        {"question": "What is the capital of the country that borders both China and India in South Asia?",
         "expected_country": "Nepal", "expected_key": "Government: Capital - name"},
        {"question": "Which languages are spoken in the most populous country in Africa?",
         "expected_country": "Nigeria", "expected_key": "People and Society: Languages"},
        {"question": "What is the economic overview of the European country that has the largest area?",
         "expected_country": "France", "expected_key": "Economy: Economic overview"},
    ],
    "ambiguous": [
        {"question": "What is Georgia known for?", "expected_country": "Georgia", "expected_key": ""},
        {"question": "Tell me about Turkey's natural resources.", "expected_country": "Turkey",
         "expected_key": "Geography: Natural resources"},
        {"question": "What kind of place is Jordan?", "expected_country": "Jordan", "expected_key": ""},
    ],
    # ── YENİ KATEGORİ: Uzun ve Gurultulu Sorular ─────────────────────────
    "long_noisy": [
        {"question": "I am a student doing extensive research and after reading many contradictory sources online about European nations and their governance structures, I just need one clear and direct answer: what is the official capital city of France, the country in Western Europe?",
         "expected_country": "France", "expected_key": "Government: Capital - name"},
        {"question": "Despite all the noise and misinformation I have encountered in various articles and social media posts today about Asian demographics and unrelated topics like sports and cuisine, could you please tell me specifically and accurately the total population count of Japan?",
         "expected_country": "Japan", "expected_key": "People and Society: Population - total"},
        {"question": "Setting aside all the irrelevant political commentary and unrelated historical events that keep appearing in my search results, I need a factual geographic description of the terrain and landscape of Brazil, the large South American country.",
         "expected_country": "Brazil", "expected_key": "Geography: Terrain"},
        {"question": "After wading through a lot of confusing and contradictory information about African governments, legal systems, and economic policies from various unreliable sources, what I really need to know is simply: what type of government does Nigeria, the most populous African country, currently have?",
         "expected_country": "Nigeria", "expected_key": "Government: Government type"},
        {"question": "I have been reading about climate patterns, meteorology, oceanic currents, El Nino effects, and many other weather-related topics, but ignoring all that background noise, can you give me a concise description of what the climate is actually like in Australia?",
         "expected_country": "Australia", "expected_key": "Geography: Climate"},
    ],
    "ood": [
        {"question": "What is the capital of Mars?", "expected_country": "", "expected_key": ""},
        {"question": "What is the population of Atlantis?", "expected_country": "", "expected_key": ""},
        {"question": "Who won the FIFA World Cup 2022?", "expected_country": "", "expected_key": ""},
        {"question": "What is the recipe for chocolate cake?", "expected_country": "", "expected_key": ""},
        {"question": "Explain quantum computing in simple terms.", "expected_country": "", "expected_key": ""},
    ],
}

all_stress_texts = [sq["question"] for qs in stress_test_questions.values() for sq in qs]

for emb_key, model_name in EMBEDDING_MODELS.items():
    print(f"\n{'─'*55}")
    print(f"[INFO] Embedding: {emb_key}  ({model_name})")
    print(f"{'─'*55}")

    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(model_name, device=device)
    load_time = time.time() - t0

    # Encoding hizi olcumu (100 ornek)
    sample_texts = [c["text"] for c in chunks_sliding[:100]]
    t1 = time.time()
    _ = st_model.encode(sample_texts, batch_size=EMBED_BATCH)
    enc_100 = time.time() - t1

    embedding_times[emb_key] = {
        "model": model_name,
        "load_time_s": round(load_time, 2),
        "encode_100_s": round(enc_100, 2),
        "avg_per_sample_ms": round(enc_100 / 100 * 1000, 2),
    }
    print(f"  Yukleme: {load_time:.1f}s  |  100 ornek encode: {enc_100:.2f}s")

    # Her chunking x embedding -> ChromaDB koleksiyonu
    for ch_key, chunks in chunking_strats.items():
        coll_name = f"final_{emb_key}_{ch_key}"
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

        # Koleksiyonu embedding_function OLMADAN olustur (kendi encode ediyoruz)
        collection = client.create_collection(name=coll_name)

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        ids   = [f"{coll_name}_{j}" for j in range(len(texts))]

        t2 = time.time()
        embeddings = st_model.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=False)
        enc_total = time.time() - t2

        for i in range(0, len(texts), 5000):
            end = min(i + 5000, len(texts))
            collection.add(
                embeddings=embeddings[i:end].tolist(),
                documents=texts[i:end],
                metadatas=metas[i:end],
                ids=ids[i:end],
            )

        collections_map[coll_name] = {
            "collection": collection,
            "emb_name": emb_key,
            "chunk_name": ch_key,
        }
        print(f"  [OK] '{coll_name}' — {len(texts)} chunk — {enc_total:.1f}s")
        del embeddings

    # Eval + stress sorularini on-encode et
    if eval_questions:
        eq_texts = [q["question"] for q in eval_questions]
        query_embs_cache[emb_key] = st_model.encode(eq_texts, batch_size=128)

    stress_embs_cache[emb_key] = st_model.encode(all_stress_texts, batch_size=64)

    del st_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n[OK] Toplam {len(collections_map)} koleksiyon olusturuldu.")

df_emb_times = pd.DataFrame(embedding_times).T
df_emb_times.index.name = "Embedding"
df_emb_times.to_csv(os.path.join(OUTPUT_DIR, "embedding_times.csv"))
print("\n--- Embedding Encoding Sureleri ---")
print(df_emb_times.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. GENİŞLETİLMİŞ EVAL SETİ (≥120 soru)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 4. ASAMA: EVAL SET GENISLETME (>=120 SORU)")
print("=" * 60)

qa_templates = {
    "Government: Capital - name":              {"q": "What is the capital city of {country}?"},
    "People and Society: Population - total":  {"q": "What is the total population of {country}?"},
    "Geography: Location":                     {"q": "Where is {country} located?"},
    "Geography: Climate":                      {"q": "What is the climate like in {country}?"},
    "Geography: Terrain":                      {"q": "Describe the terrain of {country}."},
    "Economy: Economic overview":              {"q": "Provide a brief economic overview of {country}."},
    "People and Society: Religions":           {"q": "What are the major religions in {country}?"},
    "Economy: Agricultural products":          {"q": "What are the primary agricultural products of {country}?"},
    "Government: Independence":                {"q": "When did {country} gain its independence?"},
    "Government: Government type":             {"q": "What type of government does {country} have?"},
    "People and Society: Languages":           {"q": "What languages are spoken in {country}?"},
    "Geography: Natural resources":            {"q": "What are the natural resources of {country}?"},
}

existing_qs = {q.get("question", "") for q in eval_questions}
target_count = 120
all_countries = list(raw_data.keys())
random.shuffle(all_countries)
auto_id = len(eval_questions) + 1

for country in all_countries:
    if len(eval_questions) >= target_count:
        break
    for key, tmpl in qa_templates.items():
        if len(eval_questions) >= target_count:
            break
        if key in raw_data[country] and len(str(raw_data[country][key])) > 0:
            question = tmpl["q"].format(country=country)
            if question in existing_qs:
                continue
            eval_questions.append({
                "id": f"q_final_{auto_id:03d}",
                "question": question,
                "ground_truth": str(raw_data[country][key]),
                "metadata": {"country": country, "category": key},
            })
            existing_qs.add(question)
            auto_id += 1

eval_path = os.path.join(OUTPUT_DIR, "eval_set_extended.json")
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(eval_questions, f, indent=2, ensure_ascii=False)

print(f"[INFO] Toplam Eval Set: {len(eval_questions)} soru")
cat_dist = Counter(q.get("metadata", {}).get("category", "?") for q in eval_questions)
for cat, cnt in cat_dist.most_common():
    print(f"  {cat}: {cnt}")

# Query embeddinglerini guncelle
eq_texts = [q["question"] for q in eval_questions]
for emb_key, model_name in EMBEDDING_MODELS.items():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(model_name, device=device)
    query_embs_cache[emb_key] = st_model.encode(eq_texts, batch_size=128)
    del st_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("[OK] Query embedding'leri guncellendi.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. RETRIEVER-ONLY DEGERLENDIRME
#    A) Standart (k=1,7,15) — vizeden farkli top-k
#    B) Similarity Threshold Filtering
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 5. ASAMA: RETRIEVER-ONLY DEGERLENDIRME")
print("=" * 60)


def evaluate_retriever(collection, eval_set, query_embs, top_k=7, threshold=None):
    """
    Metrikler: Hit Rate@k, Recall@k, MRR@k, Avg Latency
    threshold: None = filtreleme yok; float = distance < threshold olan sonuclar kalir
    """
    hit_count = 0
    mrr_sum = 0.0
    latency_sum = 0.0
    filtered_total = 0

    for idx, item in enumerate(eval_set):
        expected_ans = item["ground_truth"].lower()
        target_country = item.get("metadata", {}).get("country", "")
        q_emb = query_embs[idx].tolist()

        start = time.time()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        latency_sum += time.time() - start

        docs_ret  = results["documents"][0]
        metas_ret = results["metadatas"][0]
        dists     = results["distances"][0]

        # Similarity Threshold Filtering
        if threshold is not None:
            filtered = [
                (doc, meta) for doc, meta, dist in zip(docs_ret, metas_ret, dists)
                if dist < threshold
            ]
            filtered_total += len(docs_ret) - len(filtered)
            if filtered:
                docs_ret, metas_ret = zip(*filtered)
            else:
                docs_ret, metas_ret = [], []

        hit = False
        rank = 0
        for i, (doc, meta) in enumerate(zip(docs_ret, metas_ret)):
            if expected_ans[:50] in doc.lower() and meta.get("country") == target_country:
                if not hit:
                    rank = i + 1
                hit = True

        if hit:
            hit_count += 1
            mrr_sum += 1.0 / rank

    N = max(1, len(eval_set))
    return {
        "Hit Rate@k":          round(hit_count / N, 4),
        "MRR@k":               round(mrr_sum / N, 4),
        "Recall@k":            round(hit_count / N, 4),
        "Avg Latency (ms)":    round(latency_sum / N * 1000, 2),
        "Avg Filtered Out":    round(filtered_total / N, 2),
    }


retriever_metrics = []

for coll_name, coll_info in tqdm(collections_map.items(), desc="Retriever Evaluation"):
    col     = coll_info["collection"]
    emb_key = coll_info["emb_name"]
    q_embs  = query_embs_cache[emb_key]

    for top_k in TOP_K_VALUES:
        for threshold in THRESHOLDS:
            res = evaluate_retriever(col, eval_questions, q_embs, top_k=top_k, threshold=threshold)
            res["Koleksiyon"]  = coll_name
            res["Embedding"]   = emb_key
            res["Chunking"]    = coll_info["chunk_name"]
            res["Top-K"]       = top_k
            res["Threshold"]   = threshold if threshold is not None else "None"
            retriever_metrics.append(res)

df_retriever = pd.DataFrame(retriever_metrics)
df_retriever = df_retriever.sort_values("Hit Rate@k", ascending=False)

best_row            = df_retriever.iloc[0]
best_collection     = best_row["Koleksiyon"]
best_top_k          = int(best_row["Top-K"])
best_emb_key        = best_row["Embedding"]
best_threshold      = best_row["Threshold"]

print(f"\n[SONUC] En Iyi Kombinasyon : {best_collection}")
print(f"         Top-K              : {best_top_k}")
print(f"         Threshold          : {best_threshold}")
print(f"         Hit Rate           : {best_row['Hit Rate@k']:.4f}")
print(f"         MRR                : {best_row['MRR@k']:.4f}")

df_retriever.to_csv(os.path.join(OUTPUT_DIR, "retriever_results.csv"), index=False)
print(f"[INFO] retriever_results.csv kaydedildi ({len(df_retriever)} satir)")

print("\n--- Top 10 Kombinasyon ---")
print(df_retriever.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6. RE-RANKING (Cross-Encoder)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 6. ASAMA: RE-RANKING")
print("=" * 60)

rerank_summary = None
try:
    reranker_device = "cpu"
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                            max_length=512, device=reranker_device)
    print("[INFO] Cross-encoder yuklendi.")

    best_col_obj = collections_map[best_collection]["collection"]
    best_q_embs  = query_embs_cache[best_emb_key]
    rr_hit = 0
    rr_mrr = 0.0

    for idx, item in tqdm(enumerate(eval_questions), total=len(eval_questions), desc="Re-ranking"):
        q            = item["question"]
        expected_ans = item["ground_truth"].lower()
        q_emb        = best_q_embs[idx].tolist()

        results = best_col_obj.query(
            query_embeddings=[q_emb], n_results=20,
            include=["documents", "metadatas", "distances"],
        )
        docs_raw = results["documents"][0]
        dists    = results["distances"][0]

        # Threshold uygula (varsa)
        if best_threshold != "None":
            docs_raw = [d for d, dist in zip(docs_raw, dists) if dist < float(best_threshold)]

        if not docs_raw:
            continue

        scores       = reranker.predict([[q, doc] for doc in docs_raw])
        ranked_idx   = np.argsort(scores)[::-1][:best_top_k]
        reranked_docs = [docs_raw[i] for i in ranked_idx]

        for rank_i, doc in enumerate(reranked_docs):
            if expected_ans[:50] in doc.lower():
                rr_hit += 1
                rr_mrr += 1.0 / (rank_i + 1)
                break

    N = max(1, len(eval_questions))
    rerank_summary = {
        "Hit Rate@k": round(rr_hit / N, 4),
        "MRR@k":      round(rr_mrr / N, 4),
    }
    imp = ((rerank_summary["Hit Rate@k"] - best_row["Hit Rate@k"]) / max(best_row["Hit Rate@k"], 0.001)) * 100
    print(f"\n[SONUC] Re-ranking Hit Rate : {rerank_summary['Hit Rate@k']:.4f}  ({imp:+.1f}%)")
    print(f"         Re-ranking MRR      : {rerank_summary['MRR@k']:.4f}")

    del reranker

except Exception as e:
    print(f"[UYARI] Re-ranking hatasi: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. STRESS TEST (5 Kategori — long_noisy dahil)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 7. ASAMA: STRESS TEST (5 KATEGORI)")
print("=" * 60)

best_col_obj    = collections_map[best_collection]["collection"]
best_stress_embs = stress_embs_cache[best_emb_key]
stress_results  = []
emb_idx = 0

for category, questions in stress_test_questions.items():
    for sq in questions:
        q_emb = best_stress_embs[emb_idx].tolist()
        emb_idx += 1

        results = best_col_obj.query(
            query_embeddings=[q_emb],
            n_results=best_top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieved_countries = [m.get("country", "") for m in results["metadatas"][0]]
        correct = (
            sq["expected_country"] in retrieved_countries
            if sq["expected_country"] else False
        )
        stress_results.append({
            "Kategori":        category,
            "Soru":            sq["question"][:120],
            "Beklenen Ulke":   sq["expected_country"],
            "Cekilen Ulkeler": ", ".join(retrieved_countries[:3]),
            "Dogru Bulundu":   correct,
            "OOD":             category == "ood",
        })

df_stress = pd.DataFrame(stress_results)
df_stress.to_csv(os.path.join(OUTPUT_DIR, "stress_test_retriever.csv"), index=False)

print("\n--- Stress Test: Retriever Sonuclari ---")
for cat in stress_test_questions.keys():
    sub = df_stress[df_stress["Kategori"] == cat]
    if cat == "ood":
        print(f"  {'ood':>12s}: {len(sub)} OOD soru (model testinde degerlendirilecek)")
    else:
        acc = sub["Dogru Bulundu"].mean()
        print(f"  {cat:>12s}: {acc:.0%} basari ({len(sub)} soru)")

with open(os.path.join(OUTPUT_DIR, "stress_test_questions.json"), "w", encoding="utf-8") as f:
    json.dump(stress_test_questions, f, indent=2, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════════════════════
# 8. GÖRSELLEŞTİRMELER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 8. ASAMA: GORSELLESTIRMELER")
print("=" * 60)

df_k7_no_thr = df_retriever[
    (df_retriever["Top-K"] == 7) & (df_retriever["Threshold"] == "None")
].copy()

# Grafik 1: Heatmap — Embedding x Chunking (k=7, no threshold)
if not df_k7_no_thr.empty:
    pivot = df_k7_no_thr.pivot_table(index="Embedding", columns="Chunking", values="Hit Rate@k")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Final — Embedding x Chunking Hit Rate@7 (No Threshold)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_heatmap_embedding_chunking.png"), dpi=150)
    print("[OK] final_heatmap_embedding_chunking.png")
    plt.close()

# Grafik 2: Bar chart — tum kombinasyonlar (k=7)
if not df_k7_no_thr.empty:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, metric in enumerate(["Hit Rate@k", "MRR@k", "Recall@k"]):
        df_p = df_k7_no_thr.copy()
        df_p["Label"] = df_p["Embedding"] + "\n" + df_p["Chunking"]
        ax = axes[idx]
        bars = ax.bar(df_p["Label"], df_p[metric],
                      color=sns.color_palette("viridis", len(df_p)))
        ax.set_title(f"{metric} (Top-K=7)", fontsize=11)
        ax.tick_params(axis="x", rotation=55, labelsize=7)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6)
    plt.suptitle("Final — Retriever Performans (Embedding x Chunking, K=7)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_retriever_k7.png"), dpi=150)
    print("[OK] final_retriever_k7.png")
    plt.close()

# Grafik 3: Threshold etkisi (en iyi embedding + chunking ile)
df_thr = df_retriever[
    (df_retriever["Embedding"] == best_emb_key) &
    (df_retriever["Chunking"]  == best_row["Chunking"])
].copy()

if not df_thr.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k_val, ax in zip([7, 15], axes):
        sub = df_thr[df_thr["Top-K"] == k_val]
        if sub.empty:
            continue
        thr_labels = sub["Threshold"].astype(str)
        ax.bar(thr_labels, sub["Hit Rate@k"],
               color=sns.color_palette("coolwarm", len(sub)))
        ax.set_xlabel("Threshold (distance)")
        ax.set_ylabel("Hit Rate@k")
        ax.set_title(f"Threshold Etkisi (k={k_val}, {best_emb_key})")
    plt.suptitle("Similarity Threshold Filtering Etkisi", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_threshold_effect.png"), dpi=150)
    print("[OK] final_threshold_effect.png")
    plt.close()

# Grafik 4: Top-K etkisi (en iyi embedding + chunking, threshold=None)
df_topk = df_retriever[
    (df_retriever["Embedding"]  == best_emb_key) &
    (df_retriever["Threshold"]  == "None")
].copy()

if not df_topk.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    for ch in df_topk["Chunking"].unique():
        sub = df_topk[df_topk["Chunking"] == ch]
        ax.plot(sub["Top-K"], sub["Hit Rate@k"], marker="o", label=ch)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Hit Rate@k")
    ax.set_title(f"Top-K Etkisi (Final, {best_emb_key})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_topk_effect.png"), dpi=150)
    print("[OK] final_topk_effect.png")
    plt.close()

# Grafik 5: Encoding sureleri
fig, ax = plt.subplots(figsize=(9, 4))
emb_keys = list(embedding_times.keys())
enc_vals = [embedding_times[k]["encode_100_s"] for k in emb_keys]
bars = ax.barh(emb_keys, enc_vals, color=sns.color_palette("viridis", len(emb_keys)))
ax.set_xlabel("100 Ornek Encode Suresi (saniye)")
ax.set_title("Final — Embedding Encoding Hizi")
for bar, val in zip(bars, enc_vals):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_embedding_times.png"), dpi=150)
print("[OK] final_embedding_times.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 9. YAPILANDIRMA KAYDI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" 9. ASAMA: YAPILANDIRMA KAYDI")
print("=" * 60)

pipeline_config = {
    "best_collection":    best_collection,
    "best_top_k":         best_top_k,
    "best_emb_key":       best_emb_key,
    "best_chunk_key":     best_row["Chunking"],
    "best_threshold":     best_threshold,
    "best_hit_rate":      float(best_row["Hit Rate@k"]),
    "best_mrr":           float(best_row["MRR@k"]),
    "total_eval_q":       len(eval_questions),
    "total_collections":  len(collections_map),
    "embedding_models":   EMBEDDING_MODELS,
    "chunking_strategies":list(chunking_strats.keys()),
    "top_k_values":       TOP_K_VALUES,
    "thresholds":         [str(t) for t in THRESHOLDS],
    "reranking_available":rerank_summary is not None,
    "rerank_hit_rate":    rerank_summary["Hit Rate@k"] if rerank_summary else None,
    "chroma_dir":         CHROMA_DIR,
    "data_file":          INPUT_JSON,
    "seed":               SEED,
    "total_runtime_min":  round((time.time() - pipeline_start) / 60, 1),
    "ft_dir":             "./final_finetuned",
}

cfg_path = os.path.join(OUTPUT_DIR, "pipeline_config.json")
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(pipeline_config, f, indent=2, ensure_ascii=False)

print("[INFO] pipeline_config.json kaydedildi")
print(f"\n[INFO] Cikti Dosyalari ({OUTPUT_DIR}):")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    fsize = os.path.getsize(fpath)
    print(f"  {fname:>45s}  ({fsize:>10,} bytes)")

print("\n" + "=" * 70)
print(f" FINAL RAG PIPELINE TAMAMLANDI! ({pipeline_config['total_runtime_min']} dakika)")
print(" Sonraki adim: Final_QLoRA_Finetune_Llama3.py veya Final_QLoRA_Finetune_Qwen.py")
print("=" * 70)
