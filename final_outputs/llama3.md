# LLaMA 3 8B — Yerel Model Raporu

**İlgili dosyalar:** `summary_llama3_8b.csv`, `results_llama3_8b_base.csv`, `results_llama3_8b_rag_base.csv`, `results_llama3_8b_ft_norag.csv`, `results_llama3_8b_rag_ft.csv`, `stress_llama3_8b.csv`, `finetune_report_llama3.json`

---

## Genel Profil

| Özellik | Değer |
|---|---|
| Model | meta-llama/Meta-Llama-3-8B-Instruct |
| Parametre boyutu | ~8 milyar |
| VRAM kullanımı | 5,327 MB |
| RAM kullanımı | 3.31 GB |
| Fine-tuning yöntemi | QLoRA (4-bit, nf4) |

---

## Performans Özeti (4 Senaryo)

| Senaryo | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | HE Relevance | HE Faithfulness | HE Fluency |
|---|---|---|---|---|---|---|---|---|---|
| B — Base LLM (No RAG) | 0.0 | **0.4707** | 0.125 | 4.389 | 6.71 | 0.0171 | **4.28** | 3.32 | 3.98 |
| C — Fine-Tuned (No RAG) | 0.0 | **0.4707** | 0.125 | 4.386 | 6.72 | 0.0171 | **4.28** | 3.32 | 3.98 |
| D — RAG + Base LLM | 0.0 | 0.4396 | 0.10 | 4.213 | 5.82 | 0.0164 | 4.25 | 3.18 | 3.98 |
| E — RAG + Fine-Tuned | 0.0 | 0.4396 | 0.10 | 4.219 | 5.82 | 0.0164 | 4.25 | 3.18 | 3.98 |

### Öne Çıkan Bulgular

- **RAG'sız en yüksek F1 bu modele ait** (0.4707) — büyük parametre sayısının avantajı.
- Ancak RAG eklenmesi F1'i biraz **düşürmüştür** (0.4707 → 0.4396). Bu ilginç bir bulgudur; retrieval sonuçlarının bazen modelin kendi bilgisini gölgelediğine işaret eder.
- RAG, halüsinasyonu 0.125 → 0.10 indirmiş ancak F1'e olumlu katkı yapmamıştır.
- **Fine-tuning (B vs C, D vs E) sıfır fark:** Değerler milisaniye latency farkı dışında birebir aynıdır.
- Qwen 3B'ye kıyasla VRAM kullanımı %50 daha fazla (5327 vs 3551 MB).

---

## Fine-Tuning Detayları (finetune_report_llama3.json)

| Parametre | Değer |
|---|---|
| Yöntem | QLoRA (4-bit, nf4) |
| LoRA r / alpha | 16 / 32 |
| Target modüller | q_proj, k_proj, v_proj, o_proj |
| Eğitim verisi | 2,971 örnek |
| Doğrulama verisi | 331 örnek |
| Epoch | 3 |
| Öğrenme oranı | 2e-4 |
| Batch / Grad accum | 2 / 4 (efektif batch 8) |
| Max sequence length | 512 |
| Eğitim süresi | **108.1 dakika** |
| GPU saati | 1.8 saat |
| Tahmini maliyet | **$0.63** |
| Final train loss | **0.5142** (3 model içinde en düşük) |
| Donanım | Kaggle 2×T4 GPU |
| Framework | transformers + peft + trl |

En düşük train loss değerine (0.5142) sahip model Llama3'tür. Buna rağmen eval metriklerine yansımamıştır — bu durum loss'un tek başına başarıyı ölçmediğini göstermektedir.

---

## Stres Testi Sonuçları (stress_llama3_8b.csv)

> **Not:** Stres test dosyası yalnızca 2 satır içermektedir. Tam stres testi diğer modeller için çalıştırılmış, Llama3 için sınırlı kalmıştır.

| Kategori | Soru | Yanıt | Doğru mu? |
|---|---|---|---|
| paraphrase | Tell me the capital of France. | "The capital of France is Paris." | ✅ |
| OOD | What is the capital of Mars? | "I don't know." | ✅ |

Mevcut 2 test gözleminden her ikisinde de doğru yanıt vermiştir.

---

## Karşılaştırmalı Konum

| Model | F1 (Base) | F1 (RAG) | VRAM | Hız |
|---|---|---|---|---|
| qwen_3b | 0.3612 | **0.6106** | 3,551 MB | 6.25 tok/s |
| **llama3_8b** | **0.4707** | 0.4396 | 5,327 MB | 6.71 tok/s |
| mistral_7b | 0.1505 | 0.2388 | 4,006 MB | 4.43 tok/s |
| gemma_2b | 0.1859 | 0.3132 | 3,309 MB | 16.61 tok/s |

---

## Genel Değerlendirme

LLaMA 3 8B, RAG'sız senaryolarda en güçlü yerel modeldir. Ancak RAG eklenmesiyle performansı düşmekte, bu da retrieval içeriğinin modelin yerleşik bilgisiyle çakıştığını göstermektedir. Qwen'in RAG ile sağladığı F1=0.61'e ulaşamamıştır. VRAM maliyeti de yüksektir. Eğitim loss değerinin düşük olmasına rağmen fine-tuning etkisinin eval'e yansımaması, 3-epoch QLoRA ile 3B QA verisinin bu domain için sınırda kaldığına işaret etmektedir.
