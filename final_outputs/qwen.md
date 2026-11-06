# Qwen 2.5 3B — En Başarılı Yerel Model Raporu

**İlgili dosyalar:** `summary_qwen_3b.csv`, `results_qwen_3b_base.csv`, `results_qwen_3b_rag_base.csv`, `results_qwen_3b_ft_norag.csv`, `results_qwen_3b_rag_ft.csv`, `stress_qwen_3b.csv`, `finetune_report_qwen.json`

---

## Genel Profil

| Özellik | Değer |
|---|---|
| Model | Qwen/Qwen2.5-3B-Instruct |
| Parametre boyutu | ~3 milyar |
| VRAM kullanımı | 3,551 MB |
| RAM kullanımı | 4.97 GB |
| Fine-tuning yöntemi | QLoRA (4-bit, nf4) |

---

## Performans Özeti (4 Senaryo)

| Senaryo | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | HE Relevance | HE Faithfulness | HE Fluency |
|---|---|---|---|---|---|---|---|---|---|
| B — Base LLM (No RAG) | 0.0 | 0.3612 | 0.133 | 4.641 | 6.25 | 0.0542 | 4.08 | 2.92 | 3.92 |
| C — Fine-Tuned (No RAG) | 0.0 | 0.3612 | 0.133 | 4.630 | 6.28 | 0.0540 | 4.08 | 2.92 | 3.92 |
| D — RAG + Base LLM | **0.0583** | **0.6106** | **0.0** | 4.774 | 5.84 | 0.0557 | **4.67** | **3.86** | 3.81 |
| E — RAG + Fine-Tuned | **0.0583** | **0.6106** | **0.0** | 4.767 | 5.86 | 0.0556 | **4.67** | **3.86** | 3.81 |

### Öne Çıkan Bulgular

- **RAG etkisi son derece güçlü:** F1 0.36 → 0.61 (+69%), halüsinasyon 0.133 → 0.0.
- **Fine-tuning tek başına hiçbir fark yaratmamış:** B ≈ C ve D ≈ E (EM/F1 değerleri aynı, latency farkı < 0.01s).
- **Tüm senaryolar içinde en yüksek F1 bu modele ait** (0.6106).
- Projenin **en verimli modeli** — küçük boyutuna rağmen büyük modelleri geçiyor.

---

## Fine-Tuning Detayları (finetune_report_qwen.json)

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
| Eğitim süresi | **58.8 dakika** |
| GPU saati | 0.98 saat |
| Tahmini maliyet | **$0.34** (en ucuz eğitim) |
| Final train loss | 0.6721 |
| Donanım | Kaggle 2×T4 GPU |
| Framework | transformers + peft + trl |

Fine-tuning, test değerlendirmesinde kayda değer iyileşme sağlamamıştır. Loss 0.6721 makul düzeyde ancak fine-tune edilen adapter (senaryo C/E) base model ile aynı skoru vermiştir. Bu, ya veri dağılımının eval setiyle tam örtüşmediğini ya da QA formatına uyum sağlanırken genel retrieval kalitesinin korunduğunu gösterir.

---

## Stres Testi Sonuçları (stress_qwen_3b.csv)

**Toplam test:** 23 soru (5 paraphrase, 3 multi_hop, 3 ambiguous, 5 long_noisy, 5 OOD)

### Kategori Bazlı Başarı

| Kategori | Doğru | Yanlış | Not |
|---|---|---|---|
| Paraphrase | 5/5 | 0/5 | Tüm yeniden ifade sorularını doğru yanıtladı |
| Multi-hop | 1/3 | 2/3 | Nepal sorusu doğru; Nijerya ve Fransa (büyük alan) yanlış |
| Ambiguous | 2/3 | 1/3 | Georgia ve Jordan doğru; Türkiye kaynakları yanlış bağlamdan alındı |
| Long & Noisy | 5/5 | 0/5 | Uzun gürültülü soru metinlerinde hiç yanılmadı |
| **OOD** | - | - | 5 OOD sorusundan **4'ünü "I don't know." ile doğru reddetti** |

**OOD Detayı:**
- Mars başkenti → "I don't know." ✅
- Atlantis nüfusu → "I don't know." ✅
- FIFA 2022 → "The answer cannot be determined from the given context." ❌ (halüsinasyon sayıldı)
- Çikolatalı kek → "I don't know." ✅
- Kuantum bilgisayar → "I don't know." ✅

**OOD başarı oranı: %80** (4/5 doğru reddetme)

---

## Genel Değerlendirme

Qwen 2.5 3B, projenin **en dengeli ve başarılı** modelidir. Küçük parametre sayısı (3B), düşük VRAM tüketimi (~3.5 GB) ve RAG ile birlikte ulaştığı F1=0.61 değeri olağanüstü bir maliyet-performans oranı sunmaktadır. Fine-tuning bağımsız katkı sağlamamış olsa da RAG+FT kombinasyonu en az kaynak kullanan en iyi sistem olmaya devam etmektedir.
