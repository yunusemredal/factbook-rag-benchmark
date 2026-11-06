# Final Aşaması — Tam Çalıştırma Planı (Kaggle)

Bu belge `Final/` klasöründeki **tüm scriptlerin** (zorunlu + bonus, 11 Python
dosyası) Kaggle'da hangi sırayla, hangi notebook'ta ve birbirine nasıl bağlanarak
çalıştırılacağını anlatır.

> Ön koşullar (kurulum hücresi, sürüm pinleri, Secrets) → `CALISTIRMA_SIRASI.md`
> Her dosyanın ne yaptığı → `DOSYA_ACIKLAMALARI.md`

---

## ⚠️ EN ÖNEMLİ KURAL: Notebook çıktısı diğerine otomatik geçmez

Kaggle'da her notebook izole çalışır. `/kaggle/working/` altına yazdığın
`final_outputs/`, `chroma_db_final/`, `final_finetuned/` dosyaları **sadece o
notebook'ta yaşar**. Bir sonraki notebook'a taşımak için "Notebook Output → Input"
zincirini kurman gerekir. (Adım adım anlatımı en altta — *"Çıktı Zincirleme"* bölümü.)

Bu yüzden plan, **kim kimin çıktısını input alacak** şeklinde kurgulanmıştır.

---

## FAZ 0 — Hazırlık (1 kez)

- [ ] Kaggle dataset oluştur/ekle: `countries.json` + `eval_set.json`
- [ ] **Add-ons → Secrets:** `HF_TOKEN`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`
- [ ] HF erişim onayı al: Llama-3 ve Gemma model sayfalarında "Agree and access"
- [ ] Her notebook'un ilk hücresine ilgili **pinli kurulum** komutunu koy

---

## FAZ 1 — Retriever Pipeline (ilk, tek notebook)

| Notebook | Script | Accelerator | Input | Süre |
|---|---|---|---|---|
| **N1** | `Final_RAG_Pipeline.py` | GPU T4 x2 | countries+eval dataset | ~25–35 dk |

**Üretir (Save Version sonrası N1-OUTPUT):** `pipeline_config.json`,
`eval_set_extended.json`, `stress_test_questions.json`, `retriever_results.csv`,
`embedding_times.csv`, grafikler, `chroma_db_final/`.

> 🔑 **N1-OUTPUT, sonraki neredeyse tüm notebook'ların input'udur.** Bitince
> mutlaka **Save Version (Commit)** yap.

---

## FAZ 2 — Fine-Tuning (3 ayrı notebook, paralel başlatılabilir)

Accelerator: **GPU T4 x2**. Input: sadece countries dataset. Secret: `HF_TOKEN`.
Kurulum hücresi **pinli sürümlerle** olmalı (yoksa eğitim çöker).

| Notebook | Script | Çıktı (Save Version) | Süre |
|---|---|---|---|
| **N2** | `Final_QLoRA_Finetune_Qwen.py` | `final_finetuned/qwen/` | ~40–60 dk |
| **N3** | `Final_QLoRA_Finetune_Llama3.py` | `final_finetuned/llama3/` | ~60–90 dk |
| **N4** | `Final_QLoRA_Finetune_Phi35.py` | `final_finetuned/phi35/` | ~45–65 dk |

> Üçü birbirinden bağımsız → aynı anda başlatılabilir. Her biri bitince
> **Save Version** yap (adapter çıktısı kalıcı olsun).

---

## FAZ 3 — Model Değerlendirme (model başına 1 notebook)

Script: `Final_Model_Evaluator.py`. Sadece üstteki **`MODEL_KEY`** satırını değiştir.
Input: **N1-OUTPUT + ilgili fine-tune notebook çıktısı**. Secret: `HF_TOKEN`.

| Notebook | MODEL_KEY | Gereken Input | Seviyeler | Süre |
|---|---|---|---|---|
| **N5** | `qwen_3b` | N1 + N2 | A–E (tam) | ~45–70 dk |
| **N6** | `llama3_8b` | N1 + N3 | A–E (tam) | ~60–90 dk |
| **N7** | `phi_35` | N1 + N4 | A–E (tam) | ~45–70 dk |
| **N8** | `mistral_7b` | N1 | B/D (base) | ~50 dk |
| **N9** | `gemma_2b` | N1 | B/D (base) | ~40 dk |

**Her biri üretir:** `summary_<MODEL_KEY>.csv`, `results_<MODEL_KEY>_*.csv`,
`stress_<MODEL_KEY>.csv`, `metrics_<MODEL_KEY>.png`.

> N5–N7 fine-tune adapter'ı yükleyip C/E seviyelerini de ölçer.
> N8–N9 adapter olmadığı için sadece B/D ölçer ama "≥4 lokal model"
> şartını doldurur. Her notebook bitince **Save Version** yap.

---

## FAZ 4 — API LLM Testleri (2 notebook)

Input: **N1-OUTPUT**. GPU gerekmez (embedding için T4 yine de iyi olur).

| Notebook | Script | Secret | Üretir | Süre |
|---|---|---|---|---|
| **N10** | `Final_Gemini_API_Test.py` | `GEMINI_API_KEY` | `summary_gemini.csv` | ~15–20 dk |
| **N11** | `Final_NVIDIA_API_Test.py` | `NVIDIA_API_KEY` | `summary_nvidia.csv` | ~10–20 dk |

> N11, NVIDIA'nın 2 modelini (`meta/llama-3.1-8b-instruct`,
> `mistralai/mistral-7b-instruct-v0.3`) test eder. Modeller script üstündeki
> `NVIDIA_MODELS` ile değiştirilebilir. `openai` paketi kurulu olmalı.

---

## FAZ 5 — Bonuslar (bağımsız notebook'lar, istediğin zaman)

| Notebook | Script | Input | Amaç (PDF) |
|---|---|---|---|
| **N12** | `Final_FAISS_Retriever.py` | N1 | FAISS vs ChromaDB (§5.3) |
| **N13** | `Final_LoRA_Config_Comparison.py` | countries | LoRA r=8/16/32 (§9) |
| **N14** | `Final_Synthetic_Data_Generator.py` | countries + `GEMINI_API_KEY` | Sentetik veri (§9) |

> Bunlar nihai karşılaştırma tablosuna girmez; rapor bölümlerini güçlendirir.

---

## FAZ 5B — Inference Hızlandırıcı Benchmark (AYRI hesap, tek notebook)

| Notebook | Script | Accelerator | Input | Süre |
|---|---|---|---|---|
| **N5 (ayrı hesap)** | `Final_Faster_Inference.py` | **GPU T4 x2** | countries dataset + `HF_TOKEN` | ~2–3 saat |

> ⚠️ **TAMAMEN BAĞIMSIZ — Ana proje çıktısına (pipeline_config, adapter, chroma)
> İHTİYACI YOK.** Tek dosya içinde kendi QLoRA FT + RAG + 6 backend benchmark'ını
> yapar. Ayrı bir Kaggle hesabında, sıfırdan çalıştırılmak üzere tasarlandı.

**Script:** `Final_Faster_Inference.py` (665 satır, self-contained)

**Ne yapar (sırayla):**
1. `countries.json`'dan QA verisi oluşturur → hızlı QLoRA fine-tune (150 adım)
2. RAG kurar: chunk → e5-base-v2 embedding → ChromaDB → context çeker
3. Adapter'ı merge → fp16 (diğer backend'ler için)
4. **6 backend'i** aynı FT model + aynı RAG context'leriyle karşılaştırır:
   - `transformers_4bit` (baseline)
   - `transformers_merged_fp16`
   - `torch_compile` (YENİ)
   - `vllm` (TP=GPU sayısı, batched)
   - `ctranslate2` (int8)
   - `onnxruntime` (Optimum ONNX export) (YENİ)
5. Karşılaştırma tablosu + 4'lü grafik çıkarır

**Üretir:**
- `final_outputs/faster_inference_comparison.csv`
- `final_outputs/faster_inference_comparison.png`

**Kurulum (ilk hücre):**
```bash
!pip install -q vllm
!pip install -q ctranslate2
!pip install -q "optimum[onnxruntime-gpu]"
!pip install -q transformers peft trl bitsandbytes accelerate datasets \
                sentence-transformers chromadb pandas matplotlib seaborn
```

**Secret:** `HF_TOKEN` (Llama-3.1-8B gated model erişimi için)

**OOM olursa:** Script üstündeki `MODEL_ID` satırını değiştir:
```python
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # gated DEĞİL, en garanti
```

> Her backend kendi `try/except` koruması içinde — biri patlarsa diğerleri yine
> sonuç verir. Tek .py, parça parça değil.

---

## FAZ 6 — Nihai Karşılaştırma (EN SON, tek notebook)

| Notebook | Script | Input | Süre |
|---|---|---|---|
| **N15** | `Final_Comparison.py` | **N1 + N5..N11'in tüm çıktıları** | ~5 dk (CPU yeter) |

**Üretir:** `final_comparison.csv` (10. bölüm tablosu), `final_analysis.md`
(otomatik tartışma), `finetune_summary.csv`, 6 PNG.

> N15'e şu çıktıları input ekle: N1 (retriever_results) + N5–N9 (lokal model
> summary'leri) + N10 (Gemini) + N11 (NVIDIA). `summary_*.csv` glob'u hepsini
> otomatik toplar — lokal modeller, Gemini ve NVIDIA tek tabloda birleşir.

---

## Özet Akış Şeması

```
FAZ 0: Hazırlık (dataset + secrets + HF onay)
   │
FAZ 1: N1  Final_RAG_Pipeline.py ──────────────► N1-OUTPUT (config + chroma + eval)
   │                                                    │
FAZ 2: N2 Qwen   N3 Llama3   N4 Phi35  (paralel, fine-tune)
   │      │          │           │
FAZ 3: N5(qwen) N6(llama3) N7(phi) N8(mistral) N9(gemma)  ◄── N1 + ilgili adapter
   │      └──────────┴───────────┴──────────────┴────────┐
FAZ 4: N10 Gemini   N11 NVIDIA  ◄── N1                    │
   │      └─────────────┴───────────────────────────────┐│
FAZ 5: N12 FAISS  N13 LoRA-cfg  N14 Synthetic (bağımsız) ││
   │                                                     ▼▼
FAZ 6: N15  Final_Comparison.py  ◄── N1 + tüm summary_*.csv
            ► final_comparison.csv + final_analysis.md

─── AYRI HESAP (ana akıştan BAĞIMSIZ) ────────────────────────
FAZ 5B: N5  Final_Faster_Inference.py  ◄── SADECE countries.json + HF_TOKEN
            ► faster_inference_comparison.csv/png
            (Kendi FT + RAG + 6 backend benchmark — self-contained)
```

**Minimum (zorunlu) yol:** N1 → (N2,N3,N4) → (N5,N6,N7) → N10 → N15
**Tam puan yolu:** yukarıdaki + N8,N9,N11,N12,N13,N14
**Ayrı hesap:** N5 (Faster Inference) — ana projeden tamamen bağımsız

---

# Sıkıştırılmış Plan (4 Notebook) — Önerilen

15 çalıştırma yerine scriptleri tek kernel'da hücre hücre birleştir. Aynı notebook
içindeki scriptler çıktıyı `/kaggle/working/`'de paylaştığı için aralarında
Add Input zincirine gerek kalmaz. Notebook'lar arası geçişte **kopya-ileri**
hücresi (en alttaki bölüm) tüm önceki çıktıları taşır → en sonda her şey birikir.

| Notebook | Accelerator | Add Input | Süre |
|---|---|---|---|
| **A — Qwen + Retriever + API + Bonus** | GPU T4 x2 | countries+eval dataset | ~3–4 saat |
| **B — Llama3** | GPU T4 x2 | **A çıktısı** + countries | ~2.5 saat |
| **C — Phi35** | GPU T4 x2 | **B çıktısı** + countries | ~2 saat |
| **D — Base modeller + Comparison** | GPU T4 x2 | **C çıktısı** | ~1.5 saat |

> Her notebook'un **1. hücresi**: pinli kurulum. **2. hücresi**: kopya-ileri
> (A hariç — A ilk üretici). Sonra aşağıdaki `.py` dosyaları sırayla ayrı hücreler.

### 1. Notebook (A) — içindeki `.py` dosyaları (hücre sırası)

| Sıra | Hücre / `.py` dosyası | Ayar | Üretir |
|---|---|---|---|
| 1 | `!pip install ...` (kurulum) | — | — |
| 2 | `Final_RAG_Pipeline.py` | — | pipeline_config, eval_set_extended, chroma |
| 3 | `Final_QLoRA_Finetune_Qwen.py` | HF_TOKEN | final_finetuned/qwen |
| 4 | `Final_Model_Evaluator.py` | `MODEL_KEY="qwen_3b"` | summary_qwen_3b.csv |
| 5 | `Final_Gemini_API_Test.py` | GEMINI_API_KEY | summary_gemini.csv |
| 6 | `Final_NVIDIA_API_Test.py` | NVIDIA_API_KEY | summary_nvidia.csv |
| 7 | `Final_FAISS_Retriever.py` | — | faiss_vs_chroma_comparison.csv |
| 8 | `Final_LoRA_Config_Comparison.py` | — | lora_config_comparison.csv |
| 9 | `Final_Synthetic_Data_Generator.py` | GEMINI_API_KEY | synthetic_qa_pairs.jsonl |

### 2. Notebook (B) — Llama3

| Sıra | Hücre / `.py` dosyası | Ayar |
|---|---|---|
| 1 | `!pip install ...` (kurulum) | — |
| 2 | kopya-ileri hücresi | A çıktısını working'e taşır |
| 3 | `Final_QLoRA_Finetune_Llama3.py` | HF_TOKEN |
| 4 | `Final_Model_Evaluator.py` | `MODEL_KEY="llama3_8b"` |

### 3. Notebook (C) — Phi35

| Sıra | Hücre / `.py` dosyası | Ayar |
|---|---|---|
| 1 | `!pip install ...` (kurulum) | — |
| 2 | kopya-ileri hücresi | B çıktısını working'e taşır |
| 3 | `Final_QLoRA_Finetune_Phi35.py` | HF_TOKEN |
| 4 | `Final_Model_Evaluator.py` | `MODEL_KEY="phi_35"` |

### 4. Notebook (D) — Base modeller + Nihai Karşılaştırma

| Sıra | Hücre / `.py` dosyası | Ayar |
|---|---|---|
| 1 | `!pip install ...` (kurulum) | — |
| 2 | kopya-ileri hücresi | C çıktısını working'e taşır |
| 3 | `Final_Model_Evaluator.py` | `MODEL_KEY="mistral_7b"` |
| 4 | `Final_Model_Evaluator.py` | `MODEL_KEY="gemma_2b"` |
| 5 | `Final_Comparison.py` | — → final_comparison.csv |

> **Birikim mantığı:** B, A'nın tüm çıktısını taşır + Llama3 ekler; C bunu taşır
> + Phi35 ekler; D bunu taşır + mistral/gemma ekler → Comparison'da **lokal
> modeller + Gemini + NVIDIA + bonuslar tek pakette** hazır olur.

---

# Çıktı Zincirleme — "Save Version → Add Input" (adım adım)

Bir notebook'un ürettiği dosyaları başka bir notebook'ta kullanmanın yolu.

### A) Üretici notebook'ta çıktıyı kalıcılaştır (Save Version)

1. Notebook'un son hücresi `final_outputs/` (vb.) dosyalarını yazmış olmalı.
   Bunlar `/kaggle/working/` altında olur — orası otomatik "Output" sayılır.
2. Sağ üst → **"Save Version"** butonuna bas.
3. Açılan pencerede **"Save & Run All (Commit)"** seç → **Save**.
   - Bu, notebook'u baştan sona temiz bir ortamda çalıştırır ve
     `/kaggle/working/` içeriğini **kalıcı bir "Notebook Output"** olarak saklar.
   - (Hızlı denemede "Quick Save" da çıktıyı saklar ama Commit en güvenlisidir.)
4. Çalışma bitince notebook'un **"Output"** sekmesinde `final_outputs/...`,
   `final_finetuned/...` dosyalarını görmelisin.

> ⏱️ Commit, notebook'u yeniden çalıştırdığı için süre alır (fine-tune'da 1 saat+).
> Bu yüzden N1'i ve fine-tune'ları bir kez commit edip çıktısını tekrar tekrar
> kullanmak en verimlisidir.

### B) Tüketici notebook'ta o çıktıyı input olarak ekle (Add Input)

1. Yeni (veya mevcut) notebook'u aç → sağ panel **"Input"** bölümü.
2. **"+ Add Input"** butonuna bas.
3. Açılan pencerede üstten **"Your Work"** / **"Notebook Output"** sekmesine geç
   (dataset değil, **notebook çıktısı** arıyoruz).
4. Üretici notebook'u (örn. *N1 - RAG Pipeline*) bul → **"Add"** / **+** ile ekle.
5. Artık o çıktı şuraya bağlanır:
   `/kaggle/input/<notebook-output-slug>/...`
6. **Yol uyumu:** Scriptler `final_outputs/` ve `final_finetuned/`'ı
   `/kaggle/working/` altında **kendileri yeniden oluşturur** (consumer çoğu
   şeyi sıfırdan kurar). Ama bazı dosyalar (pipeline_config.json, adapter)
   input'tan okunur. Bunun sorunsuz çalışması için **en pratik yöntem**:
   tüketici notebook'un **ilk hücresinde** input'tan working'e kopyala:

```python
import shutil, glob, os
# N1 ve fine-tune ciktilarini calisma dizinine kopyala
for src in glob.glob("/kaggle/input/*/final_outputs"):
    shutil.copytree(src, "./final_outputs", dirs_exist_ok=True)
for src in glob.glob("/kaggle/input/*/final_finetuned"):
    shutil.copytree(src, "./final_finetuned", dirs_exist_ok=True)
print("Kopyalandi:", os.listdir("./final_outputs") if os.path.exists("./final_outputs") else "yok")
```

> Bu hücreyi N5–N11 ve N15'in **en başına** (kurulum + script'ten önce) ekle.
> Böylece scriptler `./final_outputs/pipeline_config.json` ve
> `./final_finetuned/<model>/` dosyalarını olduğu yerde bulur — kod değişmeden çalışır.

### C) Hızlı kontrol
Tüketici notebook'ta script'ten önce şunu çalıştırıp dosyaların geldiğini doğrula:
```python
import os
print("config var mi :", os.path.exists("./final_outputs/pipeline_config.json"))
print("eval var mi   :", os.path.exists("./final_outputs/eval_set_extended.json"))
print("adapter var mi:", os.path.exists("./final_finetuned/qwen/adapter_config.json"))
```

---

## Pratik İpuçları

- **GPU kotası:** Kaggle haftalık ~30 saat GPU verir. Fine-tune'lar (N2–N4) ve
  evaluator'lar (N5–N9) en çok harcar. Önce zorunlu yolu bitir, bonusları kotaya
  göre ekle.
- **CUDA OOM:** Aynı notebook'ta iki model yüklüyse kernel'ı **Restart** et.
- **Commit zorunlu değil ama önerilir:** Çıktıyı sadece indirip yeni dataset
  olarak da yükleyebilirsin; ama "Notebook Output → Add Input" zinciri en hızlısı.
- **Sıra bozulursa:** `pipeline_config.json not found` → N1 input'u eklenmemiş.
  `adapter_config.json not found` → ilgili fine-tune çıktısı eklenmemiş.
```
