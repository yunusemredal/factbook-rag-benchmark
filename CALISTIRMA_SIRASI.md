# Final Proje — Kaggle Çalıştırma Sırası

## Ön Koşullar (Her notebook'un ilk hücresine koy)

> **ÖNEMLİ — Sürüm pinleri:** Fine-tuning scriptleri `SFTTrainer(dataset_text_field=…,
> max_seq_length=…, tokenizer=…)` ve `TrainingArguments(evaluation_strategy=…)` eski
> API'sini kullanır. Güncel `trl`/`transformers` bu imzayı kaldırdı; bu yüzden
> aşağıdaki uyumlu sürümleri pinliyoruz (Qwen2.5, Phi-3.5, Llama-3, Gemma-2 destekli).

```python
# Fine-tuning notebook'lari (Llama3 / Qwen / Phi35 / LoRA-config) icin:
!pip install -q "transformers==4.44.2" "trl==0.11.4" "peft==0.13.2" \
             "accelerate==0.34.2" "datasets==2.21.0" "bitsandbytes==0.43.3" \
             matplotlib pandas

# RAG / Retriever / Evaluator / FAISS / Gemini / NVIDIA notebook'lari icin:
!pip install -q sentence-transformers chromadb langchain_text_splitters \
             faiss-cpu seaborn matplotlib pandas tqdm psutil \
             google-generativeai openai
```

> FAISS: Kaggle'da `faiss-cpu` her zaman kurulur; veri seti küçük (~2.7K chunk)
> olduğu için CPU yeterince hızlıdır. GPU isterseniz `faiss-gpu` deneyin.

**Kaggle Secrets** (Add-ons → Secrets):
- `HF_TOKEN` — Hugging Face token (Llama3 için gerekli; modele erişim onayı da alın)
- `GEMINI_API_KEY` — Google AI Studio token
- `NVIDIA_API_KEY` — build.nvidia.com API anahtarı (`nvapi-...`); NVIDIA API testi için

**Veri seti:** `countries.json` ve `eval_set.json` dosyalarını içeren Kaggle
dataset'ini notebook'a ekleyin (Add Input). Scriptler dosyayı `/kaggle/input/`
ağacında otomatik arar (`_resolve` fonksiyonu), bu yüzden dataset slug adınız
farklı olsa bile yolu elle düzeltmeniz gerekmez.

---

## Zorunlu Dosyalar (PDF Puanı)

### ADIM 1 — Retriever Pipeline
**`Final_RAG_Pipeline.py`**
- Bağımlılık: Yok (ilk çalışan)
- Süre: ~25–35 dk (4 embedding modeli × 3 chunking × 3 top-k × 4 threshold)
- Üretir:
  - `final_outputs/pipeline_config.json` ← Sonraki TÜM scriptler bu dosyayı okur
  - `final_outputs/retriever_results.csv`
  - `final_outputs/eval_set_extended.json`
  - `final_outputs/stress_test_questions.json`
  - `chroma_db_final/` (ChromaDB index)

> **UYARI:** Bu adım tamamlanmadan diğer hiçbir script çalışmaz.

---

### ADIM 2 — Fine-Tuning (Paralel çalıştırılabilir — Ayrı notebook'lar)

Her biri ayrı Kaggle notebook'unda eş zamanlı başlatılabilir.

#### 2A — Llama3 Fine-Tune
**`Final_QLoRA_Finetune_Llama3.py`**
- Bağımlılık: `countries.json`, `HF_TOKEN`
- Süre: ~60–90 dk
- Üretir: `final_finetuned/llama3/adapter_config.json`

#### 2B — Qwen Fine-Tune
**`Final_QLoRA_Finetune_Qwen.py`**
- Bağımlılık: `countries.json`
- Süre: ~40–60 dk
- Üretir: `final_finetuned/qwen/adapter_config.json`

#### 2C — Phi-3.5 Fine-Tune (Bonus: 3. model)
**`Final_QLoRA_Finetune_Phi35.py`**
- Bağımlılık: `countries.json`
- Süre: ~45–65 dk
- Üretir: `final_finetuned/phi35/adapter_config.json`

---

### ADIM 3 — Model Değerlendirme (Her model için ayrı çalıştır)

**`Final_Model_Evaluator.py`**
- Bağımlılık: Adım 1 (pipeline_config.json) + ilgili modelin adapter'ı
- **Script içinde MODEL_KEY değiştir:**

```python
# Dosyanın en üstündeki satırı değiştir:
MODEL_KEY = "qwen_3b"       # Seçenek 1
MODEL_KEY = "phi_35"        # Seçenek 2
MODEL_KEY = "llama3_8b"     # Seçenek 3 (HF_TOKEN gerekli)
MODEL_KEY = "mistral_7b"    # Seçenek 4 (HF_TOKEN gerekli)
MODEL_KEY = "gemma_2b"      # Seçenek 5
```

- Süre: Model başına ~45–90 dk
- Üretir (her MODEL_KEY için):
  - `final_outputs/results_{MODEL_KEY}_base.csv`
  - `final_outputs/results_{MODEL_KEY}_rag_base.csv`
  - `final_outputs/results_{MODEL_KEY}_ft_norag.csv`
  - `final_outputs/results_{MODEL_KEY}_rag_ft.csv`
  - `final_outputs/stress_{MODEL_KEY}.csv`
  - `final_outputs/summary_{MODEL_KEY}.csv`

---

### ADIM 4 — Gemini API Testi (Opsiyonel ama önerilen)

**`Final_Gemini_API_Test.py`**
- Bağımlılık: Adım 1, `GEMINI_API_KEY` Secrets'ta tanımlı olmalı
- Süre: ~15–20 dk (rate limit nedeniyle yavaş)
- Üretir:
  - `final_outputs/results_gemini_base.csv`
  - `final_outputs/results_gemini_rag.csv`
  - `final_outputs/summary_gemini.csv`
  - `final_outputs/results_gemini_ood.csv`

---

### ADIM 4B — NVIDIA API Testi (Gemini'ye EK 2 model — opsiyonel)

**`Final_NVIDIA_API_Test.py`**
- Bağımlılık: Adım 1, `NVIDIA_API_KEY` Secrets'ta tanımlı olmalı, `openai` paketi
- Süre: ~10–20 dk (2 model × 60 soru × 2 mod)
- Test edilen modeller (script üstündeki `NVIDIA_MODELS` ile değiştirilebilir):
  - `meta/llama-3.1-8b-instruct`
  - `mistralai/mistral-7b-instruct-v0.3`
- Üretir:
  - `final_outputs/results_nvidia_<model>_base.csv` / `_rag.csv`
  - `final_outputs/summary_nvidia.csv` ← Final_Comparison.py otomatik okur
  - `final_outputs/results_nvidia_ood.csv`
  - `final_outputs/nvidia_comparison.png`

---

### ADIM 5 — Final Karşılaştırma (EN SON çalıştır)

**`Final_Comparison.py`**
- Bağımlılık: Adım 1, 2, 3, 4'ten üretilen TÜM CSV dosyaları
- Süre: ~5 dk
- Üretir:
  - `final_outputs/final_comparison.csv` ← Ana tablo
  - `final_outputs/final_analysis.md` ← Otomatik tartışma metni
  - `final_outputs/finetune_summary.csv`
  - 6 adet PNG grafik

---

## Bonus Dosyalar (Ek puan / gösteriş)

Zorunlu adımlardan bağımsız, ayrı notebook'larda çalıştırılabilir.

### BONUS A — FAISS vs ChromaDB Karşılaştırması
**`Final_FAISS_Retriever.py`**
- Bağımlılık: Adım 1 (pipeline_config.json)
- Süre: ~15–20 dk
- Üretir: `final_outputs/faiss_vs_chroma_comparison.csv`

### BONUS B — LoRA Konfigürasyon Karşılaştırması
**`Final_LoRA_Config_Comparison.py`**
- Bağımlılık: `countries.json`
- Süre: ~30–45 dk (3 config × 80 step)
- Üretir: `final_outputs/lora_config_comparison.csv`

### BONUS C — Sentetik Veri Üretimi
**`Final_Synthetic_Data_Generator.py`**
- Bağımlılık: `countries.json`, `GEMINI_API_KEY` (yoksa demo modda çalışır)
- Süre: ~20–30 dk
- Üretir: `final_outputs/synthetic_qa_pairs.jsonl`

---

## Özet Akış Şeması

```
Final_RAG_Pipeline.py          (ADIM 1 — zorunlu ilk)
         │
         ▼
┌─────────────────────────────────────────────┐
│  Fine-Tuning (Paralel — Ayrı notebook'lar) │
│  2A: Llama3   2B: Qwen   2C: Phi35         │
└─────────────────────────────────────────────┘
         │
         ▼
Final_Model_Evaluator.py       (ADIM 3 — her model için ayrı çalıştır)
         │
         ▼
Final_Gemini_API_Test.py       (ADIM 4 — opsiyonel)
         │
         ▼
Final_Comparison.py            (ADIM 5 — en son)

─── Bağımsız (istediğin zaman) ───────────────
Final_FAISS_Retriever.py
Final_LoRA_Config_Comparison.py
Final_Synthetic_Data_Generator.py

─── AYRI HESAP (ana akıştan tamamen bağımsız) ─
Final_Faster_Inference.py  (N5 — self-contained benchmark)
```

---

### BONUS D — Inference Hızlandırıcı Benchmark (AYRI Kaggle Hesabı)

**`Final_Faster_Inference.py`** — **N5 Notebook**

> ⚠️ **TAMAMEN SELF-CONTAINED:** Bu script ana projenin hiçbir çıktısına
> (pipeline_config.json, adapter, chroma_db) bağımlı DEĞİLDİR. Ayrı bir
> Kaggle hesabında, sıfırdan çalıştırılmak üzere tasarlanmıştır.

- Bağımlılık: **SADECE** `countries.json` + `HF_TOKEN` secret
- Diğer notebook çıktılarını **ALMAZ** — kendi FT + RAG pipeline'ını içerir
- Accelerator: **GPU T4 x2** (8B model — tek T4'e sığmaz)
- Süre: ~2–3 saat (FT ~20dk + RAG ~5dk + 6 backend ~90dk)
- Kurulum (ilk hücre):
```bash
!pip install -q vllm
!pip install -q ctranslate2
!pip install -q "optimum[onnxruntime-gpu]"
!pip install -q transformers peft trl bitsandbytes accelerate datasets \
                sentence-transformers chromadb pandas matplotlib seaborn
```
- Üretir:
  - `final_outputs/faster_inference_comparison.csv`
  - `final_outputs/faster_inference_comparison.png` (4'lü grafik)
- İçerdiği backend'ler (BACKENDS dict ile aç/kapat):
  1. `transformers_4bit` — baseline (4-bit + LoRA adapter)
  2. `transformers_merged_fp16` — merge + fp16
  3. `torch_compile` — merged fp16 + torch.compile
  4. `vllm` — batched + TP + LoRA
  5. `ctranslate2` — int8 convert
  6. `onnxruntime` — Optimum ONNX export
- OOM olursa: `MODEL_ID` satırını `TinyLlama/TinyLlama-1.1B-Chat-v1.0` yap

---

## Sık Yapılan Hatalar

| Hata | Sebep | Çözüm |
|---|---|---|
| `pipeline_config.json not found` | Adım 1 tamamlanmamış | Final_RAG_Pipeline.py'yi önce çalıştır |
| `adapter_config.json not found` | Fine-tune bitmemiş | İlgili fine-tune scriptini çalıştır |
| `GEMINI_API_KEY not found` | Secret tanımlı değil | Kaggle → Add-ons → Secrets |
| `HF_TOKEN not found` | Secret tanımlı değil | Kaggle → Add-ons → Secrets |
| CUDA OOM | İki model aynı anda yüklü | Notebook kernel'ı yeniden başlat |
| `No module named faiss` | GPU kurulumu | `!pip install faiss-gpu` |
