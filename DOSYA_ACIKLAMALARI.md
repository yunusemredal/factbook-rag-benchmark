# Final Aşaması — Dosya Açıklamaları

Bu belge `Final/` klasöründeki her dosyanın **ne yaptığını**, **hangi PDF
gereksinimini karşıladığını**, **girdi/çıktılarını** ve **çalıştırma sırasını**
anlatır. Domain: CIA World Factbook ülke verisi (`countries.json`, 258 ülke).

> Çalıştırma sırası ve Kaggle ön koşulları için: **`CALISTIRMA_SIRASI.md`**.

---

## 0. Genel Mimari

```
countries.json (ham veri)
        │
        ▼
[1] Final_RAG_Pipeline.py ──► pipeline_config.json + chroma + eval_set_extended.json
        │                          (en iyi embedding×chunking×k×threshold seçilir)
        ▼
[2] Fine-tuning (paralel)   ──► final_finetuned/{qwen,llama3,phi35}/ (LoRA adapter)
        │
        ▼
[3] Final_Model_Evaluator.py ──► her model için A–E seviyeleri + stress + özet CSV
        │
        ▼
[4] Final_Gemini_API_Test.py ──► API LLM sonuçları (opsiyonel)
        │
        ▼
[5] Final_Comparison.py ──► final_comparison.csv + grafikler + final_analysis.md

Bağımsız bonuslar: FAISS_Retriever, LoRA_Config_Comparison, Synthetic_Data_Generator
```

Tüm scriptler veriyi `_resolve()` fonksiyonuyla `/kaggle/input/` ağacında otomatik
arar; dataset slug adınız ne olursa olsun yolu elle düzeltmeniz gerekmez.

---

## ZORUNLU DOSYALAR

### 1. `Final_RAG_Pipeline.py` — Gelişmiş Retriever Pipeline
**PDF karşılığı:** §4.1 Gelişmiş Pipeline, §4.2 Genişletilmiş Kombinasyon Analizi,
§5 (Chunking/Embedding/Vektör DB/Retriever), §7-A Retriever-Only, §8 Stress/OOD.

**Ne yapar:**
- **3 yeni chunking** (vizeden farklı): `sliding_window` (300/150), `sentence_based`,
  `hierarchical` (özet + bölüm detayı 2 seviye).
- **4 yeni embedding** (vizeden farklı): `e5-base-v2`, `bge-base-en-v1.5`,
  `gte-base`, `multilingual-mpnet`. Yükleme + encode süreleri ölçülür.
- **ChromaDB** koleksiyonları: 4 embedding × 3 chunking = **12 koleksiyon**.
- **Retriever-only metrikler:** Hit Rate@k, MRR@k, Recall@k, latency —
  k ∈ {1,7,15} × threshold ∈ {None, 0.5, 0.7, 0.9} (Similarity Threshold Filtering).
- **Re-ranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) ile en iyi koleksiyonda
  top-20 → yeniden sıralama; iyileşme yüzdesi raporlanır.
- **Stress test (5 kategori):** paraphrase, multi_hop, ambiguous, long_noisy, ood.
- **Eval set'i ≥120 soruya** genişletir (12 kategori şablonu).
- En iyi konfigürasyonu seçip **5 grafik** üretir.

**Girdi:** `countries.json`, (varsa) `eval_set.json`
**Çıktı:** `pipeline_config.json` *(sonraki tüm scriptler bunu okur)*,
`retriever_results.csv`, `eval_set_extended.json`, `stress_test_questions.json`,
`embedding_times.csv`, 5 PNG, `chroma_db_final/`.

> **Bu dosya çalışmadan diğer hiçbiri çalışmaz** (pipeline_config.json bağımlılığı).

---

### 2A/2B/2C. Fine-Tuning Scriptleri (QLoRA)
**PDF karşılığı:** §9 Model Fine-Tuning (zorunlu), Parameter-Efficient (QLoRA),
Instruction Tuning (QA çiftleri), §9 veri miktarı (≥500 QA).

Üç dosya da **aynı mantık**, farklı model + chat şablonu:

| Dosya | Model | Şablon | LoRA target | Adapter çıktısı |
|---|---|---|---|---|
| `Final_QLoRA_Finetune_Llama3.py` | Llama-3-8B-Instruct | llama3 | q,k,v,o_proj | `final_finetuned/llama3/` |
| `Final_QLoRA_Finetune_Qwen.py` | Qwen2.5-3B-Instruct | ChatML | q,k,v,o_proj | `final_finetuned/qwen/` |
| `Final_QLoRA_Finetune_Phi35.py` | Phi-3.5-mini | phi3 | qkv_proj,o_proj | `final_finetuned/phi35/` |

**Ne yapar:**
- 14 alan şablonundan **≥500 QA çifti** üretir (258 ülke × ~14 alan ≈ 3000+ örnek),
  %90/%10 train/val böler.
- **4-bit QLoRA** (nf4, double-quant, fp16 compute — T4 uyumlu).
- LoRA: r=16, α=32, dropout=0.05. 3 epoch, lr=2e-4, cosine scheduler.
- Eğitim sonrası: adapter + tokenizer kaydeder, **loss grafiği** çizer,
  `finetune_report_*.json` (süre, GPU saati, maliyet, final loss) yazar.

**Girdi:** `countries.json` + `HF_TOKEN`
**Çıktı:** `final_finetuned/<model>/adapter_config.json` + rapor JSON.

> Phi-3.5 farklı attention katman adı (`qkv_proj`) kullanır — kod bunu zaten
> doğru hedefliyor. 3 model = **3 farklı model ailesi** fine-tune edilmiş olur
> (PDF "en az birini" ister; burada fazlasıyla karşılanıyor).

---

### 3. `Final_Model_Evaluator.py` — 5 Seviyeli Değerlendirme (A–E)
**PDF karşılığı:** §7 (5 seviyeli analiz), §10 Karşılaştırma Tablosu metrikleri,
§8 OOD, Human Evaluation (proxy), Latency/Token-s/Memory/Cost.

**Ne yapar:** Tek bir `MODEL_KEY` için (`qwen_3b`, `phi_35`, `llama3_8b`,
`mistral_7b`, `gemma_2b`) şu seviyeleri ölçer:
- **B)** Base LLM (No RAG)
- **C)** Fine-Tuned LLM (No RAG) — adapter varsa otomatik
- **D)** RAG + Base LLM
- **E)** RAG + Fine-Tuned LLM — adapter varsa otomatik

*(Seviye A = Retriever-Only, pipeline'dan gelir.)*

**Metrikler:** EM, F1, Hallucination rate, Latency, Token/s, VRAM spike,
Cost, Human-eval proxy (Relevance/Faithfulness/Fluency 1–5).
Ayrıca **stress + OOD** testini model seviyesinde (doğru reddetme / halüsinasyon)
çalıştırır.

**Kullanım:** Her model için **ayrı Kaggle notebook**, sadece üstteki `MODEL_KEY`
satırını değiştir.
**Girdi:** `pipeline_config.json`, `eval_set_extended.json`, ilgili adapter, `HF_TOKEN`
**Çıktı:** `results_{MODEL_KEY}_{base,rag_base,ft_norag,rag_ft}.csv`,
`stress_{MODEL_KEY}.csv`, `summary_{MODEL_KEY}.csv`, `metrics_{MODEL_KEY}.png`.

> **Not (insan değerlendirmesi):** `human_eval_proxy` otomatik bir yaklaşımdır.
> PDF gerçek, çok-değerlendiricili Likert puanı ister — raporda en az birkaç soru
> için elle Relevance/Faithfulness/Fluency puanı ekleyip ortalamayı sununuz.

---

### 4. `Final_Gemini_API_Test.py` — API LLM (Faz 1)
**PDF karşılığı:** §6 "≥1 API model" zorunluluğu.

**Ne yapar:** Gemini 1.5 Flash ile **No-RAG vs RAG** karşılaştırması (60 soru
sınırı, maliyet kontrolü) + OOD reddetme testi. Rate-limit ve 3-deneme hata
yönetimi içerir. Tahmini API maliyetini hesaplar.

**Girdi:** `pipeline_config.json`, `eval_set_extended.json`, `GEMINI_API_KEY`
**Çıktı:** `results_gemini_base.csv`, `results_gemini_rag.csv`,
`summary_gemini.csv` *(Comparison bunu okur)*, `results_gemini_ood.csv`,
`gemini_comparison.png`.

---

### 4B. `Final_NVIDIA_API_Test.py` — NVIDIA API (Gemini'ye EK 2 model)
**PDF karşılığı:** §6 API LLM (API model çeşitliliğini artırır).

**Ne yapar:** NVIDIA'nın OpenAI-uyumlu NIM endpoint'i
(`https://integrate.api.nvidia.com/v1`) üzerinden **2 NVIDIA modelini**
(varsayılan: `meta/llama-3.1-8b-instruct`, `mistralai/mistral-7b-instruct-v0.3`)
No-RAG vs RAG modunda değerlendirir + OOD testi. Token/s'i API'nin `usage`
bilgisinden gerçek olarak hesaplar; halüsinasyon oranını da ölçer.

Modelleri değiştirmek için script üstündeki `NVIDIA_MODELS` sözlüğünü düzenleyin
(alternatifler yorum satırında: nemotron-70b, llama-3.3-70b, qwen2.5-7b, gemma-2-9b).

**Girdi:** `pipeline_config.json`, `eval_set_extended.json`, `NVIDIA_API_KEY` (Secrets), `openai` paketi
**Çıktı:** `results_nvidia_<model>_base.csv` / `_rag.csv`,
`summary_nvidia.csv` *(Comparison `summary_*.csv` glob'uyla otomatik okur)*,
`results_nvidia_ood.csv`, `nvidia_comparison.png`.

> API anahtarı: build.nvidia.com → herhangi bir model sayfası → "Get API Key"
> (`nvapi-...`). Gemini testiyle birlikte toplam **3 API modeli** karşılaştırılmış olur.

---

### 5. `Final_Comparison.py` — Nihai Karşılaştırma (EN SON)
**PDF karşılığı:** §10 Zorunlu Karşılaştırma Tablosu, §11 Analiz ve Tartışma.

**Ne yapar:** Tüm `summary_*.csv`, `retriever_results.csv`,
`finetune_report_*.json`, `stress_*.csv` dosyalarını birleştirir:
- **Seviye A–E + Gemini** tek tabloda (`final_comparison.csv`).
- 6 grafik: EM/F1, halüsinasyon, latency, human-eval radar, maliyet/bellek, OOD.
- `finetune_summary.csv` (tüm FT raporları).
- `final_analysis.md` — en iyi kombinasyon, FT etkisi, RAG vs FT vs Hybrid,
  stress/OOD bulguları için **otomatik tartışma metni**.

**Girdi:** `final_outputs/` altındaki tüm önceki çıktılar
**Çıktı:** `final_comparison.csv`, `final_analysis.md`, 6 PNG.

---

## BONUS DOSYALAR (ek puan)

### `Final_FAISS_Retriever.py` — İkinci Vektör DB
**PDF §5.3.** FAISS'in 4 index tipini (FlatIP, FlatL2, IVFFlat, HNSW) ChromaDB
ile karşılaştırır: Hit Rate, latency, build süresi, trade-off grafikleri.
Raporun "Vektör Veritabanı" bölümünü güçlendirir.
**Çıktı:** `faiss_vs_chroma_comparison.csv`, `faiss_index_comparison.png`.

### `Final_LoRA_Config_Comparison.py` — Hiperparametre Analizi
**PDF §9.** 3 LoRA konfigürasyonunu (r=8/16/32, farklı target katmanlar)
Qwen2.5-3B üzerinde 80 adımda karşılaştırır: eğitilebilir parametre sayısı,
train/eval loss, süre, maliyet trade-off'u. "Neden r=16 seçtik" gerekçesi.
**Çıktı:** `lora_config_comparison.csv`, `lora_config_comparison.png`.

### `Final_Synthetic_Data_Generator.py` — Sentetik Veri
**PDF §9 Synthetic Veri.** Gemini ile çeşitli QA çiftleri üretir (factual,
descriptive, inferential), kalite filtreler, ChatML formatına çevirir.
`GEMINI_API_KEY` yoksa **demo modda** şablon verisi üretir. Üretim yöntemi +
kalite analizi raporlanır (PDF gereği).
**Çıktı:** `synthetic_qa_pairs.jsonl`, `synthetic_qa_analysis.csv`,
`synthetic_training_data.jsonl`, `synthetic_data_analysis.png`.

### `CALISTIRMA_SIRASI.md`
Kaggle çalıştırma sırası, sürüm pinleri, Secrets, bağımlılık şeması, sık hatalar.

---

## Bu İncelemede Düzeltilen Hatalar

| # | Dosya | Sorun | Düzeltme |
|---|---|---|---|
| 1 | `Final_LoRA_Config_Comparison.py` | Bozuk f-string `{eval_loss:.4f if eval_loss else 'N/A'}` → `ValueError` ile çökerdi | Önce string'e çevirip yazdırma |
| 2 | `Final_Model_Evaluator.py` | `FT_ADAPTER_PATHS` içinde `phi_35` yoktu → Phi-3.5 fine-tune adapter'ı hiç yüklenmez, C/E seviyeleri boşa giderdi | `phi_35` yolu eklendi |
| 3 | `Final_Comparison.py` | Gemini hem `summary_gemini.csv` glob'undan hem manuel bloktan eklenip **mükerrer satır** üretiyordu | Manuel blok kaldırıldı |
| 4 | Tüm scriptler (8 dosya) | Veri yolu `/kaggle/input/datasets/yunusemredal/cfbook/...` Kaggle'ın gerçek mount şeması değil → `FileNotFoundError` | `_resolve()` ile `/kaggle/input` ağacında otomatik arama |
| 5 | `CALISTIRMA_SIRASI.md` | Pinsiz kurulum → güncel `trl`/`transformers` eski `SFTTrainer`/`evaluation_strategy` API'sini kırar | Uyumlu sürümler pinlendi, `faiss-cpu`'ya geçildi |

## Çalıştırmadan Önce Kontrol Listesi
- [ ] `countries.json` + `eval_set.json` dataset olarak notebook'a eklendi.
- [ ] `HF_TOKEN` Secrets'ta + Llama-3 / Gemma model sayfalarından erişim onayı alındı.
- [ ] `GEMINI_API_KEY` Secrets'ta (Gemini ve sentetik veri için).
- [ ] Fine-tuning notebook'ları **pinli** kurulum hücresiyle başlatıldı.
- [ ] Sıra: Pipeline → Fine-tune → Evaluator (her model) → Gemini → Comparison.
- [ ] Rapora **gerçek human-eval** Likert puanları elle eklendi (proxy yeterli değil).
