# Gemma 2B — Küçük & Hızlı Model Raporu

**İlgili dosyalar:** `summary_gemma_2b.csv`, `results_gemma_2b_base.csv`, `results_gemma_2b_rag_base.csv`, `stress_gemma_2b.csv`

---

## Genel Profil

| Özellik | Değer |
|---|---|
| Model | Gemma 2B (yerel çalıştırma) |
| Parametre boyutu | ~2 milyar |
| VRAM kullanımı | 3,309 MB (en düşük) |
| RAM kullanımı | 4.29 GB |
| Fine-tuning | Uygulanmamış |

---

## Performans Özeti (2 Senaryo)

| Senaryo | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | HE Relevance | HE Faithfulness | HE Fluency |
|---|---|---|---|---|---|---|---|---|---|
| B — Base LLM (No RAG) | 0.0 | 0.1859 | 0.10 | 1.996 | **16.61** | 0.0078 | 3.55 | 2.28 | 3.98 |
| D — RAG + Base LLM | 0.0 | 0.3132 | 0.10 | **1.710** | 14.86 | 0.0067 | 3.08 | 2.55 | 3.75 |

### Öne Çıkan Bulgular

- **En hızlı yerel model:** 16.61 tok/s (Base) ve 14.86 tok/s (RAG). Mistral'in ~4× katı hız.
- **En düşük VRAM:** 3,309 MB. Kısıtlı GPU ortamları için ideal.
- **En düşük maliyet:** $0.0078/soru (Base). Mistral'den 8× ucuz.
- **RAG etkisi:** F1 0.186 → 0.313 (+68%). Halüsinasyon sabit kalmış (0.10).
- RAG eklenmesiyle **Human Evaluation relevance düşmüş** (3.55 → 3.08) — retrieval context'i yanıtı bazen dağıtmış olabilir.
- Fine-tuning yapılmamış; C ve E senaryoları mevcut değil.

---

## Stres Testi Sonuçları (stress_gemma_2b.csv)

> **Not:** Stres testi dosyası yalnızca 2 satır içermektedir.

| Kategori | Soru | Yanıt | Değerlendirme |
|---|---|---|---|
| paraphrase | Tell me the capital of France. | "The context does not provide the capital of France." | ❌ Yanıltıcı (context'e aşırı bağımlı) |
| OOD | What is the capital of Mars? | "I don't know." | ✅ Doğru reddetme |

**Paraphrase testinde dikkat çekici bir sorun:** Model, "Fransa'nın başkenti Paris'tir" bilgisine sahip olmasına rağmen context'te açık ifade görmediği için yanıt vermemiştir. Bu, 2B modelin güçlü bir parametrik bilgiye sahip olmadığını ve tamamen retrieval context'ine yaslandığını gösterir.

**OOD testinde başarılı:** Kısa ve net bir reddetme cevabı vermiştir.

---

## Karşılaştırmalı Konum

| Metrik | Gemma 2B | Qwen 3B | LLaMA 3 8B | Mistral 7B |
|---|---|---|---|---|
| F1 (RAG) | 0.3132 | **0.6106** | 0.4396 | 0.2388 |
| Latency (s) | **1.71** | 4.77 | 4.21 | 16.99 |
| VRAM (MB) | **3,309** | 3,551 | 5,327 | 4,006 |
| Tokens/s | **16.61** | 6.25 | 6.71 | 4.43 |
| Cost/soru ($) | **0.0067** | 0.0557 | 0.0164 | 0.0661 |

---

## Genel Değerlendirme

Gemma 2B, **hız ve kaynak verimliliği** açısından grubun zirvesindedir. Ancak küçük parametre sayısı nedeniyle F1 skoru düşük kalmaktadır. Gerçek zamanlı veya düşük kaynaklı uygulamalar için güçlü bir adaydır. Fine-tuning uygulanmadığından tam potansiyeli test edilmemiştir — özellikle RAG+FT kombinasyonu Qwen'e benzer bir iyileşme sağlayabilirdi. Context'e aşırı bağımlılık eğilimi, prompt engineering ile giderilebilecek bir sorundur.
