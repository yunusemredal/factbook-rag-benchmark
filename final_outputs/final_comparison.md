# Final Karşılaştırma — Tüm Sistemlerin Genel Raporu

**İlgili dosyalar:** `final_comparison.csv`, `final_analysis.md`, `pipeline_config.json`

---

## Sistem Listesi

Bu deneyde toplam **5 senaryo tipi** test edilmiştir (PDF'deki A–E şemasına göre):

| Kod | Senaryo | Açıklama |
|---|---|---|
| A | Retriever-Only | Yalnızca doküman arama, LLM yok |
| B | LLM-Only (Base) | RAG olmadan, base model |
| C | Fine-Tuned LLM (No RAG) | Fine-tune edilmiş, RAG olmadan |
| D | RAG + Base LLM | Retrieval + base model |
| E | RAG + Fine-Tuned LLM | Retrieval + fine-tuned model |

---

## Tam Karşılaştırma Tablosu

| Sistem | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | RAM (GB) | VRAM (MB) | HE Rel. | HE Faith. | HE Flu. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A — Retriever-Only** | — | — | — | 0.003 | — | 0.0 | 0.0 | 0.0 | — | — | — |
| Gemini Flash — No RAG | 0.0 | 0.0279 | 0.0* | 0.0* | 0.0* | 0.0* | 0.0 | 0.0 | 1.77 | 1.30 | 4.0 |
| Gemini Flash — RAG | 0.0 | 0.0279 | 0.0* | 0.0* | 0.0* | 0.0* | 0.0 | 0.0 | 1.77 | 1.30 | 4.0 |
| gemma_2b — B | 0.0 | 0.1859 | 0.10 | 1.996 | 16.61 | 0.0078 | 4.29 | 3,309 | 3.55 | 2.28 | 3.98 |
| gemma_2b — D | 0.0 | 0.3132 | 0.10 | 1.710 | 14.86 | 0.0067 | 4.29 | 3,309 | 3.08 | 2.55 | 3.75 |
| mistral_7b — B | 0.0 | 0.1505 | 0.125 | 16.711 | 4.43 | 0.065 | 3.64 | 4,006 | 3.92 | 2.12 | 4.0 |
| mistral_7b — D | 0.0 | 0.2388 | 0.125 | 16.986 | 3.98 | 0.0661 | 3.64 | 4,006 | 3.88 | 2.33 | 4.0 |
| NVIDIA llama31_8b — No RAG | 0.0 | 0.1817 | 0.15 | 0.960 | 48.47 | 0.0* | 0.0 | 0.0 | 3.75 | 2.27 | 4.0 |
| NVIDIA llama31_8b — RAG | 0.0 | 0.2859 | 0.05 | 0.835 | 41.57 | 0.0* | 0.0 | 0.0 | 3.53 | 2.38 | 3.88 |
| NVIDIA mistral_7b — No RAG | 0.0 | 0.0013 | 0.967 | 0.0* | 0.0* | 0.0* | 0.0 | 0.0 | 1.10 | 1.00 | 4.0 |
| NVIDIA mistral_7b — RAG | 0.0 | 0.0013 | 0.067 | 0.0* | 0.0* | 0.0* | 0.0 | 0.0 | 1.10 | 1.00 | 4.0 |
| llama3_8b — B | 0.0 | 0.4707 | 0.125 | 4.389 | 6.71 | 0.0171 | 3.31 | 5,327 | 4.28 | 3.32 | 3.98 |
| llama3_8b — C | 0.0 | 0.4707 | 0.125 | 4.386 | 6.72 | 0.0171 | 3.31 | 5,327 | 4.28 | 3.32 | 3.98 |
| llama3_8b — D | 0.0 | 0.4396 | 0.10 | 4.213 | 5.82 | 0.0164 | 3.31 | 5,327 | 4.25 | 3.18 | 3.98 |
| llama3_8b — E | 0.0 | 0.4396 | 0.10 | 4.219 | 5.82 | 0.0164 | 3.31 | 5,327 | 4.25 | 3.18 | 3.98 |
| **qwen_3b — B** | 0.0 | 0.3612 | 0.133 | 4.641 | 6.25 | 0.0542 | 4.97 | 3,551 | 4.08 | 2.92 | 3.92 |
| qwen_3b — C | 0.0 | 0.3612 | 0.133 | 4.630 | 6.28 | 0.0540 | 4.97 | 3,551 | 4.08 | 2.92 | 3.92 |
| **qwen_3b — D** | **0.0583** | **0.6106** | **0.0** | 4.774 | 5.84 | 0.0557 | 4.97 | 3,551 | **4.67** | **3.86** | 3.81 |
| **qwen_3b — E** | **0.0583** | **0.6106** | **0.0** | 4.767 | 5.86 | 0.0556 | 4.97 | 3,551 | **4.67** | **3.86** | 3.81 |

> `*` ile işaretlenen değerler ölçüm/loglama eksikliği veya API hatasından kaynaklanan 0.0 değerleridir; gerçek performansı yansıtmaz.

---

## Retriever-Only Skoru (Senaryo A)

| Hit Rate@15 | MRR@15 | Recall@15 | Latency (ms) |
|---|---|---|---|
| 1.0 | 0.8222 | 1.0 | 2.54 |

Retriever (e5_base + hierarchical + Top-K=15), soruların **%100'ünde** doğru dokümanları bulmuştur. Latency yalnızca ~2.5 ms.

---

## Analiz Soruları (PDF Bölüm 11'e Karşılık)

### En iyi embedding–chunking kombinasyonu?
**e5_base + hierarchical + Top-K=15** → Hit Rate=1.0, MRR=0.8222. `gte_base` alternatifi MRR=0.8576 ile biraz daha iyi sıralama sunar.

### Retrieval performansını en çok ne etkiledi?
Chunking stratejisi: `hierarchical` diğer iki stratejiden belirgin şekilde üstün çıkmıştır. Top-K=1 kullanımı Hit Rate'i dramatik düşürmüştür. Threshold, özellikle `multilingual_mpnet` için zararlı olmuştur.

### RAG ne kadar iyileştirme sağladı?
| Model | F1 (Base) | F1 (RAG) | Artış |
|---|---|---|---|
| qwen_3b | 0.3612 | **0.6106** | **+69%** |
| gemma_2b | 0.1859 | 0.3132 | +68% |
| mistral_7b | 0.1505 | 0.2388 | +59% |
| NVIDIA llama31 | 0.1817 | 0.2859 | +57% |
| llama3_8b | 0.4707 | 0.4396 | -7% (hafif düşüş) |

### Lokal ve API modelleri nasıl karşılaştırıldı?
- **Hız:** NVIDIA NIM (48 tok/s) >> Gemma 2B (16 tok/s) > Qwen/LLaMA (~6 tok/s) >> Mistral (4 tok/s)
- **Doğruluk:** Qwen 3B local > LLaMA 3 8B local > NVIDIA LLaMA > tüm API modeller (bozulmuş)
- **Maliyet:** Yerel modeller $0.007–0.066/soru; API maliyetleri kayıtlı değil

### Fine-tuned model, base modele göre ne kadar performans artışı sağladı?
**Sıfır artış.** B=C ve D=E tüm modellerde. Fine-tuning, bu deney koşullarında EM/F1'i değiştirmemiştir.

### Fine-tuning tek başına mı yoksa RAG ile birlikte mi daha etkili?
**Her ikisinde de fine-tuning etkisiz; asıl fark RAG'den gelmiştir.** En iyi performans D ve E'de (RAG+Base ≈ RAG+FT).

### RAG + Fine-Tuned model en iyi sonucu verdi mi?
Evet, **ancak yalnızca RAG katkısı nedeniyle.** E senaryosu (RAG+FT), fine-tuning sayesinde değil RAG sayesinde başarılı olmuştur.

### OOD sorularda sistem nasıl davrandı?
Qwen 3B: %80 doğru reddetme (5'ten 4'ü). NVIDIA LLaMA: %80. Gemma/Mistral: LLM katmanında test sınırlı.

---

## Trade-off Analizi

| Boyut | En İyi | En Kötü |
|---|---|---|
| F1 (doğruluk) | qwen_3b RAG (0.61) | NVIDIA mistral (0.001) |
| Latency | NVIDIA llama31 (0.84s) | mistral_7b (16.99s) |
| Throughput | NVIDIA llama31 (48 tok/s) | mistral_7b (3.98 tok/s) |
| VRAM | gemma_2b (3,309 MB) | llama3_8b (5,327 MB) |
| Halüsinasyon | qwen_3b RAG (0.0) | NVIDIA mistral No-RAG (0.967*) |
| HE Faithfulness | qwen_3b RAG (3.86) | NVIDIA mistral (1.00*) |
| Maliyet/soru | gemma_2b RAG ($0.007) | mistral_7b ($0.066) |

---

## Genel Sonuç

1. **Kazanan sistem:** `qwen_3b + RAG (D/E)` — F1=0.61, halüsinasyon=0.0, güçlü insan değerlendirmesi.
2. **RAG zorunlu:** Çoğu modelde belirgin iyileşme sağlamış; tek istisna LLaMA 3 8B (F1 hafif düşmüş).
3. **Fine-tuning etkisiz:** Bu domain/eval konfigürasyonunda 3-epoch QLoRA sıfır getiri sağlamıştır.
4. **Hız-doğruluk dengesi:** NVIDIA LLaMA 3.1 hız şampiyonu ama doğrulukta çok geride; Gemma 2B yerel modellerde hız şampiyonu ama doğrulukta orta.
5. **Başarısız denemeler:** Gemini (API 404), NVIDIA Mistral (API 404), Phi-3.5 fine-tune (loss=2.48) — 3 deney teknik sorunla sonuçlanmıştır.
