# Final Çıktıları — Dosya Dosya Açıklama Raporu

Bu rapor, `final_outputs/` klasöründeki tüm çıktı dosyalarını tek tek açıklar; her dosyanın **ne olduğunu**, **hangi sütunları içerdiğini** ve **sonuçların ne anlama geldiğini** özetler.

Pipeline; bir **RAG (retrieval) sistemi** + **çoklu LLM** (yerel QLoRA fine-tune'lu modeller, Gemini, NVIDIA NIM) üzerinde **EM/F1**, **halüsinasyon**, **latency**, **maliyet/bellek**, **insan değerlendirmesi (HE)**, **stres testi** ve **OOD (alan dışı) reddetme** ölçümlerini karşılaştırır.

> **Genel veri:** 120 değerlendirme sorusu, 12 retriever koleksiyonu, ülke ansiklopedisi (`countries.json`) verisi, seed=42, toplam çalışma süresi ~33.5 dk (Kaggle 2×T4 GPU).

---

## 1. Yapılandırma ve Genel Özet Dosyaları

### `pipeline_config.json`
Tüm pipeline'ın **çalıştırma yapılandırması ve en iyi seçimler**.
- **En iyi retriever:** `e5_base` embedding + `hierarchical` chunking + Top-K=15 → **Hit Rate=1.0, MRR=0.8222**
- Test edilen embedding modelleri: `e5_base`, `bge_base`, `gte_base`, `multilingual_mpnet`
- Chunking stratejileri: `sliding_window`, `sentence_based`, `hierarchical`
- Top-K değerleri: 1, 7, 15 | Threshold'lar: None, 0.5, 0.7, 0.9
- Reranking kullanılmadı (`reranking_available: false`)
- 120 değerlendirme sorusu, 12 koleksiyon, seed=42

### `final_analysis.md`
Projenin **11. Bölüm — Analiz ve Tartışma** metni; bulguların yazılı yorumu:
- En iyi embedding–chunking: **e5_base + hierarchical**
- En yüksek EM/F1: **qwen_3b — RAG + Base** (EM=0.0583, F1=0.6106)
- En düşük halüsinasyon: Gemini Flash (ölçüm 0.0 — aşağıdaki nota bakın)
- Fine-tuning tek başına kayda değer iyileşme sağlamadı; **RAG ile birlikte en iyi sonuç**
- RAG vs Fine-Tuning vs Hybrid tartışması + stres/OOD bulguları

### `final_comparison.csv`
**Tüm sistemlerin yan yana karşılaştırması** — ana sonuç tablosu (her satır bir sistem).
Sütunlar: `Sistem, EM, F1, Hallucination_Rate, Avg_Latency_s, Avg_Tokens/s, Avg_Mem_Spike_MB, Total_Cost_USD, RAM_GB, VRAM_MB, HE_relevance, HE_faithfulness, HE_fluency, Hit_Rate@k, MRR@k, Embedding, Chunking, Top-K`

**Öne çıkanlar:**
| Sistem | F1 | Halüsinasyon | Latency (s) | İnsan-İlgililik |
|---|---|---|---|---|
| qwen_3b — RAG+Base / RAG+FT | **0.6106** | **0.0** | 4.77 | 4.67 |
| llama3_8b — Base/FT | 0.4707 | 0.125 | 4.39 | 4.28 |
| llama3_8b — RAG | 0.4396 | 0.10 | 4.21 | 4.25 |
| qwen_3b — Base/FT | 0.3612 | 0.133 | 4.64 | 4.08 |
| gemma_2b — RAG+Base | 0.3132 | 0.10 | 1.71 | 3.08 |
| NVIDIA llama31 — RAG | 0.2859 | 0.05 | 0.84 | 3.53 |
| mistral_7b — RAG | 0.2388 | 0.125 | 16.99 | 3.88 |
| gemma_2b — Base | 0.1859 | 0.10 | 2.00 | 3.55 |
| NVIDIA mistral_7b | 0.0013 | 0.07–0.97 | 0.0 | 1.1 |
| Gemini Flash | 0.0279 | 0.0* | 0.0* | 1.77 |

> **Genel sonuç:** En iyi sistem **qwen_3b + RAG**. RAG, F1'i belirgin artırıp halüsinasyonu sıfıra indiriyor. Fine-tune'un base'e göre EM/F1 farkı yok (D≈E, B≈C) — kazanç tamamen RAG'den geliyor.

---

## 2. Sistem Bazlı Özet Dosyaları (`summary_*.csv`)

Her dosya, tek bir model ailesi için A–E senaryolarının özetidir. Senaryo kodları:
**B** = sadece Base LLM, **C** = Fine-Tuned LLM (RAG yok), **D** = RAG + Base LLM, **E** = RAG + Fine-Tuned LLM.

### `summary_qwen_3b.csv` — **en başarılı model**
- B/C (RAG yok): F1=0.3612, halüsinasyon=0.133
- D/E (RAG'li): **F1=0.6106, halüsinasyon=0.0, EM=0.0583** → RAG net kazanç
- C ≈ B ve E ≈ D: fine-tune tek başına fark yaratmıyor, etki RAG'den geliyor

### `summary_llama3_8b.csv`
- B/C: F1=0.4707 (RAG'sızken en yüksek F1)
- D/E: F1=0.4396 (RAG burada F1'i biraz düşürmüş ama halüsinasyonu 0.125→0.10'a indirmiş)
- Yüksek VRAM (5326 MB), 8B model

### `summary_gemma_2b.csv`
- En küçük yerel model; RAG ile F1 0.186→0.313'e yükseliyor
- En düşük VRAM (3309 MB) ve hızlı (~2 s); sadece B ve D senaryosu var

### `summary_mistral_7b.csv`
- RAG ile F1 0.150→0.239; halüsinasyon sabit 0.125
- **En yavaş yerel model (~17 s latency)** — verimlilik açısından dezavantajlı

### `summary_gemini.csv` (Gemini Flash, API)
- F1=0.0279, latency/maliyet 0.0 olarak kaydedilmiş
- > **Not:** Latency, halüsinasyon ve maliyetin 0.0 olması muhtemelen **ölçüm/loglama eksikliğidir** (API çağrısı süre/maliyet kaydı tutulmamış), gerçek performans değil. F1'in çok düşük olması da cevap formatı uyuşmazlığına işaret eder.

### `summary_nvidia.csv` (NVIDIA NIM API)
- `llama31_8b`: RAG ile F1 0.182→0.286, halüsinasyon 0.15→0.05, en yüksek throughput (~48 tok/s)
- `mistral_7b`: **F1≈0.0013, No-RAG halüsinasyon 0.9667** → API hatası (aşağıdaki OOD dosyasına bakın), pratikte çalışmamış

---

## 3. LLM Ham Cevap Dosyaları (`results_*.csv`)

Her dosya, ilgili model+senaryo için **soru bazında ham sonuçları** içerir (her satır bir soru).
Sütunlar: `question, gold, response, country, EM, F1, Hallucination, Latency, Tokens/s, Mem_Spike_MB, Cost_USD, HE_relevance, HE_faithfulness, HE_fluency`

Senaryolara göre dosyalar:
- **Base (RAG yok):** `results_qwen_3b_base.csv`, `results_llama3_8b_base.csv`, `results_mistral_7b_base.csv`, `results_gemma_2b_base.csv`, `results_gemini_base.csv`, `results_nvidia_*_base.csv`
- **RAG + Base:** `results_*_rag_base.csv` (örn. `results_qwen_3b_rag_base.csv`, `results_gemini_rag.csv`, `results_nvidia_*_rag.csv`)
- **Fine-Tuned, RAG yok:** `results_qwen_3b_ft_norag.csv`, `results_llama3_8b_ft_norag.csv`
- **RAG + Fine-Tuned:** `results_qwen_3b_rag_ft.csv`, `results_llama3_8b_rag_ft.csv`

Bu dosyalar `summary_*.csv` ve `final_comparison.csv` tablolarının **kaynak verisidir**; tek tek hangi soruda modelin doğru/yanlış ya da halüsinasyon yaptığını görmek için kullanılır.

---

## 4. Retriever (RAG Arama) Dosyaları

### `retriever_results.csv` — **embedding/chunking deney matrisi**
144 satır: tüm embedding × chunking × Top-K × threshold kombinasyonlarının retrieval skorları.
Sütunlar: `Hit Rate@k, MRR@k, Recall@k, Avg Latency (ms), Avg Filtered Out, Koleksiyon, Embedding, Chunking, Top-K, Threshold`

**Bulgular:**
- En iyi sıralamalar **hierarchical chunking + Top-K=15**'te toplanıyor (Hit Rate=1.0)
- `gte_base + hierarchical` en yüksek **MRR=0.8576** (e5_base'den biraz daha iyi sıralama), `e5_base` ise seçilen koleksiyon
- **Top-K=1** ve **sentence_based** chunking belirgin şekilde daha zayıf (Hit Rate 0.27–0.69)
- `multilingual_mpnet` + yüksek threshold (0.5–0.9) kombinasyonları çoğu chunk'ı filtreleyip Hit Rate'i **0.0**'a düşürüyor → bu embedding için threshold zararlı
- `Avg Filtered Out`: threshold ile elenen chunk sayısı; yükseldikçe Hit Rate düşüyor

### `stress_test_retriever.csv` — **retriever stres/dayanıklılık testi**
Sütunlar: `Kategori, Soru, Beklenen Ulke, Cekilen Ulkeler, Dogru Bulundu, OOD`
Kategoriler:
- **paraphrase / long_noisy:** Uzun, gürültülü sorularda bile doğru ülkeyi genellikle ilk sonuçlarda buluyor ✅
- **multi_hop:** Çok adımlı sorularda zorlanıyor (örn. Nijerya, Fransa sorularında yanlış) ❌ → query decomposition önerisi
- **ambiguous:** Belirsiz isimlerde (Georgia, Jordan) genelde başarılı
- **ood (Mars, Atlantis...):** Beklenen ülke yok; retriever alakasız ülkeler döndürüyor → reddetme LLM katmanında yapılmalı

---

## 5. LLM Stres Testi Dosyaları (`stress_*.csv`)

Her dosya, bir model için **dayanıklılık + OOD reddetme** davranışını ölçer.
Sütunlar: `Model, Kategori, Soru, Cevap, OOD, Reddetme, Dogru_Reddetme, Halucinasyon, Latency`

- `stress_qwen_3b.csv`, `stress_llama3_8b.csv`, `stress_mistral_7b.csv`, `stress_gemma_2b.csv`

**Örnek (qwen_3b):** paraphrase/ambiguous/long_noisy sorularına doğru ve akıcı cevap veriyor; **OOD sorularında (Mars, Atlantis, çikolatalı kek) "I don't know." diyerek doğru reddediyor** (5 OOD'den 4'ünü doğru reddetti, sadece "World Cup 2022" sorusunda halüsinasyon). Bu, RAG + iyi prompt'un OOD reddetmeyi güçlendirdiğini gösteriyor.

---

## 6. OOD (Alan Dışı) Karşılaştırma

### `results_nvidia_ood.csv`
NVIDIA modellerinin OOD davranışı:
- **`nvidia_llama31_8b`:** 5 OOD sorusunun 4'ünü "I don't know." ile doğru reddetti; sadece "World Cup 2022"de halüsinasyon
- **`nvidia_mistral_7b`:** Tüm sorularda **`API Hatası: Error code: 404` (Function not found)** → endpoint/fonksiyon ID'si geçersiz, model hiç çalışmadı. `summary_nvidia.csv`'deki F1≈0 ve %96.67 halüsinasyon bu API hatasından kaynaklanıyor; model performansı değil altyapı sorunu.

---

## 7. Fine-Tuning Raporları

### `finetune_summary.csv` — **3 modelin eğitim özeti**
Sütunlar: `Model, Method, Train_Examples, Epochs, Learning_Rate, Train_Time_Min, GPU_Hours, Cost_USD, Final_Loss`

| Model | Yöntem | Süre (dk) | GPU-saat | Maliyet | Final Loss |
|---|---|---|---|---|---|
| Llama-3-8B-Instruct | QLoRA 4-bit | 108.1 | 1.8 | $0.63 | **0.5142** |
| Qwen2.5-3B-Instruct | QLoRA 4-bit | 58.8 | 0.98 | $0.34 | 0.6721 |
| Phi-3.5-mini-instruct | QLoRA 4-bit | 210.9 | 3.52 | $1.23 | **2.4809** |

- Tümü 2971 eğitim örneği, 3 epoch, lr=2e-4
- **Llama-3** en düşük loss; **Phi-3.5** çok yüksek loss (2.48) + en pahalı/yavaş → bu eğitim verimsiz olmuş

### `finetune_report_qwen.json`, `finetune_report_llama3.json`, `finetune_report_phi35.json`
Her modelin **detaylı eğitim hiperparametreleri**: LoRA r/alpha, target_modules, batch/grad_accum, max_seq_len=512, warmup, adapter yolu, donanım (Kaggle 2×T4).
- qwen & llama3: `lora_r=16, alpha=32`, target=`q/k/v/o_proj`
- phi35: `lora_r=32, alpha=64`, daha geniş target modüller (`qkv_proj, gate_up_proj, down_proj`), `completion_only=true`

> **Not:** Fine-tune edilen adapter'lar değerlendirmede Qwen ve Llama3 için kullanıldı (senaryo C/E). Phi-3.5 eğitildi ama yüksek loss nedeniyle final karşılaştırmada öne çıkmıyor.

---

## 8. Grafik Dosyaları (`.png`)

| Dosya | Ne gösterir |
|---|---|
| `final_em_f1.png` | Tüm sistemlerin **EM ve F1** karşılaştırma bar grafiği (qwen+RAG zirvede) |
| `final_hallucination.png` | Sistem bazında **halüsinasyon oranları** (RAG'in düşürücü etkisi) |
| `final_latency.png` | **Latency** karşılaştırması (mistral_7b en yavaş, NVIDIA/Gemini en hızlı) |
| `final_cost_memory.png` | **Maliyet ve bellek (RAM/VRAM/Mem-Spike)** karşılaştırması |
| `final_human_eval_radar.png` | İnsan değerlendirmesi **radar grafiği** (relevance/faithfulness/fluency) |
| `final_ood_comparison.png` | Modellerin **OOD reddetme** başarısı karşılaştırması |

---

## 9. Genel Sonuçlar (Özet)

1. **En iyi retriever:** `e5_base + hierarchical + Top-K=15` (Hit Rate=1.0, MRR=0.82). `gte_base` MRR'de biraz daha iyi.
2. **En iyi uçtan uca sistem:** **qwen_3b + RAG** → F1=0.61, halüsinasyon=0.0, güçlü insan değerlendirmesi (relevance 4.67).
3. **RAG kazandırıyor:** Çoğu modelde F1'i artırıp halüsinasyonu düşürdü; OOD reddetmeyi güçlendirdi.
4. **Fine-tuning tek başına yetersiz:** Base ile EM/F1 farkı ~0; asıl kazanç RAG'den. En iyi kombinasyon **RAG + Fine-Tuned (hybrid)**.
5. **Zayıf noktalar:** multi-hop retrieval, mistral_7b yavaşlığı, NVIDIA mistral API 404 hatası, Gemini için eksik metrik loglama.
6. **Maliyet:** Fine-tune toplam ~$2.2 (3 model); en verimli eğitim Qwen ($0.34, 59 dk).

---

*Bu rapor `final_outputs/` klasöründeki çıktıların yorumudur. Sayısal değerler ilgili CSV/JSON dosyalarından alınmıştır.*
