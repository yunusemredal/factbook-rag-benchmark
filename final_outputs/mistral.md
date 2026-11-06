# Mistral 7B — Yerel Model Raporu

**İlgili dosyalar:** `summary_mistral_7b.csv`, `results_mistral_7b_base.csv`, `results_mistral_7b_rag_base.csv`, `stress_mistral_7b.csv`

---

## Genel Profil

| Özellik | Değer |
|---|---|
| Model | Mistral 7B (yerel çalıştırma) |
| Parametre boyutu | ~7 milyar |
| VRAM kullanımı | 4,006 MB |
| RAM kullanımı | 3.64 GB |
| Fine-tuning | Uygulanmamış (yalnızca B ve D senaryosu) |

---

## Performans Özeti (2 Senaryo)

| Senaryo | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | HE Relevance | HE Faithfulness | HE Fluency |
|---|---|---|---|---|---|---|---|---|---|
| B — Base LLM (No RAG) | 0.0 | 0.1505 | 0.125 | **16.711** | 4.43 | 0.065 | 3.92 | 2.12 | 4.0 |
| D — RAG + Base LLM | 0.0 | 0.2388 | 0.125 | **16.986** | 3.98 | 0.0661 | 3.88 | 2.33 | 4.0 |

### Öne Çıkan Bulgular

- **En yavaş yerel model:** Ortalama latency ~17 saniye. Aynı donanımda Gemma 2B'nin (~2s) ve Qwen 3B'nin (~4.6s) çok üzerinde.
- **RAG etkisi sınırlı:** F1 0.15 → 0.24 (+59%), ancak halüsinasyon değişmemiş (sabit 0.125).
- **Human evaluation:** Fluency=4.0 ile akıcı yanıtlar üretiyor; ancak Faithfulness=2.12/2.33 ile bilgiye bağlılık zayıf.
- **Fine-tuning yapılmamış** — dolayısıyla C ve E senaryoları mevcut değil.
- Maliyet ($0.065/soru) diğer yerel modellere kıyasla yüksek (Qwen: $0.054, Gemma: $0.0078).

---

## Stres Testi Sonuçları (stress_mistral_7b.csv)

> **Not:** Mistral için de stres testi sınırlı tutulmuş; yalnızca 2 satır mevcuttur.

| Kategori | Soru | Yanıt | Değerlendirme |
|---|---|---|---|
| paraphrase | Tell me the capital of France. | "France does not have a capital city mentioned in the provided context. The context only mentions the location of the US embassy in Paris..." | ❌ Yanlış (yanıltıcı) |
| OOD | What is the capital of Mars? | "I don't know. The context provided is about the capitals of Libya, Bulgaria, San Marino, Turkey, not Mars." | ✅ Doğru reddetme |

**Paraphrase testinde ilginç bir hata:** Yanıtta Fransa'nın başkenti söylenmemiş, context'teki farklı ülkelere atıfta bulunulmuştur. Bu, Mistral'in RAG context'ini doğru şekilde odaklamakta zorlandığına işaret eder.

**OOD testinde başarılı:** "I don't know" diyerek reddetmiş ve neden bilmediğini açıklayan bir bağlam sunmuştur.

---

## Karşılaştırmalı Konum

| Metrik | Mistral 7B | Qwen 3B | LLaMA 3 8B |
|---|---|---|---|
| F1 (Base) | 0.1505 | 0.3612 | 0.4707 |
| F1 (RAG) | 0.2388 | **0.6106** | 0.4396 |
| Latency (s) | **16.7** | 4.6 | 4.4 |
| VRAM (MB) | 4,006 | 3,551 | 5,327 |
| Tokens/s | 4.43 | 6.25 | 6.71 |

---

## Genel Değerlendirme

Mistral 7B, bu deneyde **en dezavantajlı yerel model** konumundadır. 7 milyar parametresine rağmen hem F1 hem de hız açısından çok daha küçük modellerden (Qwen 3B, Gemma 2B) geri kalmaktadır. ~17 saniyelik latency, gerçek zamanlı kullanım için kabul edilemez bir değerdir. Fine-tuning uygulanmadığından C/E senaryoları eksiktir. RAG, halüsinasyonu azaltamamış olmakla birlikte F1'e kısmi katkı sağlamıştır. Eğer fine-tuning yapılsaydı ve latency sorunu giderilseydi farklı bir tablo oluşabilirdi.
