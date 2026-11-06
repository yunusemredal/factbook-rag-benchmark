# NVIDIA NIM API — İki Model Raporu

**İlgili dosyalar:** `summary_nvidia.csv`, `results_nvidia_llama31_8b_base.csv`, `results_nvidia_llama31_8b_rag.csv`, `results_nvidia_mistral_7b_base.csv`, `results_nvidia_mistral_7b_rag.csv`, `results_nvidia_ood.csv`

---

## Genel Profil

NVIDIA NIM (NVIDIA Inference Microservice) platformu üzerinden iki model test edilmiştir:
- **NVIDIA LLaMA 3.1 8B** — çalışmış, sonuçlar geçerli
- **NVIDIA Mistral 7B** — API 404 hatası, sonuçlar geçersiz

Her iki model de bulut API olarak çalışmış; VRAM ve RAM yükü sıfır olarak raporlanmıştır.

---

## Performans Özeti

| Sistem | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | HE Relevance | HE Faithfulness | HE Fluency |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA 3.1 8B — No RAG | 0.0 | 0.1817 | 0.15 | 0.960 | **48.47** | 0.0 | 3.75 | 2.27 | 4.0 |
| LLaMA 3.1 8B — RAG | 0.0 | 0.2859 | **0.05** | **0.835** | 41.57 | 0.0 | 3.53 | 2.38 | 3.88 |
| Mistral 7B — No RAG | 0.0 | 0.0013 | **0.9667** | 0.0 | 0.0 | 0.0 | 1.10 | 1.00 | 4.0 |
| Mistral 7B — RAG | 0.0 | 0.0013 | 0.0667 | 0.0 | 0.0 | 0.0 | 1.10 | 1.00 | 4.0 |

---

## NVIDIA LLaMA 3.1 8B — Detay

### Başarılar
- **En yüksek throughput:** 48.47 tok/s (Base) — tüm test edilen sistemler içinde zirvede.
- **RAG ile anlamlı iyileşme:** F1 0.18 → 0.29 (+57%), halüsinasyon 0.15 → 0.05 (üçte bir azalma).
- **Latency mükemmel:** RAG ile 0.835 saniye — yerel modellerin 4-17 saniyesine karşı 5-20× daha hızlı.
- **Maliyet sıfır olarak kaydedilmiş** — bu büyük ihtimalle API ücret takibinin yapılmadığını gösterir, gerçek maliyet farklıdır.

### Sınırlılıklar
- F1=0.2859 değeriyle Qwen 3B+RAG (0.6106) ve LLaMA 3 8B base (0.4707) gerisinde.
- Human evaluation relevance puanı RAG ile hafif düşmüş (3.75 → 3.53).
- Yerel çalıştırmaya kıyasla veri gizliliği sorunu yaratır.

---

## NVIDIA Mistral 7B — Detay (API Hatası)

### Sorun
Tüm istekler `Error code: 404 — Function not found` hatası vermiştir. Model endpoint'i ya yanlış yapılandırılmış ya da bu NVIDIA NIM versiyonunda desteklenmemektedir.

### Sayısal Sonuçlar (Geçersiz)
- **No RAG halüsinasyon: 0.9667** — bu oran bir API hata mesajının halüsinasyon olarak işaretlenmesinden kaynaklanmaktadır.
- **RAG halüsinasyon: 0.0667** — RAG versiyonunda da aynı API hatası alınmış; bazı yanıtlar tesadüfen halüsinasyon sayılmamış.
- **F1 ≈ 0.0013** — sıfıra eşdeğer, gerçek model performansı değil.
- Human Evaluation Faithfulness=1.0 — değerlendiriciler hata metnine en düşük puanı vermiş.

> Bu sonuçlar hiçbir şekilde Mistral 7B modelinin gerçek performansını yansıtmamaktadır.

---

## OOD Testi (results_nvidia_ood.csv)

NVIDIA LLaMA 3.1 8B için OOD davranışı:

| Soru | Reddetme | Halüsinasyon |
|---|---|---|
| What is the capital of Mars? | ✅ "I don't know." | — |
| What is the population of Atlantis? | ✅ "I don't know." | — |
| Who won the FIFA World Cup 2022? | ❌ | ✅ (bilgi vermeye çalışmış) |
| What is the recipe for chocolate cake? | ✅ "I don't know." | — |
| Explain quantum computing? | ✅ "I don't know." | — |

**OOD başarı oranı (LLaMA 3.1): %80** — Qwen ile eşit.

NVIDIA Mistral 7B için OOD: tüm yanıtlar API 404 hatası, değerlendirilemez.

---

## Genel Değerlendirme

NVIDIA NIM üzerinde çalışan **LLaMA 3.1 8B**, bu deneydeki en hızlı sistemdir (~48 tok/s). Throughput açısından yerel modelleri 6-10× geçmektedir. Ancak mutlak F1 performansı orta düzeydedir. Yerel LLaMA 3 8B ile aynı parametre sayısına sahip olmasına rağmen daha düşük F1 vermesi, NIM endpoint'inin farklı bir model versiyonu veya yapılandırması kullandığına işaret edebilir.

**NVIDIA Mistral 7B** deneyde tamamen başarısız olmuş ve karşılaştırmaya dahil edilememiştir. API endpoint yapılandırması düzeltilip yeniden test edilmesi gerekmektedir.
