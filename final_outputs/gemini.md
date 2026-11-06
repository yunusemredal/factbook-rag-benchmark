# Gemini Flash — API Model Raporu

**İlgili dosyalar:** `summary_gemini.csv`, `results_gemini_base.csv`, `results_gemini_rag.csv`, `results_gemini_ood.csv`

---

## Genel Durum

Gemini Flash, Google'ın API üzerinden çalıştırılan modelidir. Bu deneyde **tüm API çağrıları 404 hatası** döndürmüştür:

```
API Hatası: 404 models/gemini-1.5-flash is not found for API version v1beta,
or is not supported for generateContent.
```

Bu nedenle, aşağıdaki tüm metrik değerleri gerçek model performansını değil **çağrı başarısızlığını** yansıtmaktadır.

---

## Performans Özeti

| Senaryo | EM | F1 | Hallucination | Latency (s) | Tokens/s | Cost ($) | HE Relevance | HE Faithfulness | HE Fluency |
|---|---|---|---|---|---|---|---|---|---|
| No RAG | 0.0 | 0.0279 | 0.0 | 0.0 | 0.0 | 0.0 | 1.77 | 1.30 | 4.0 |
| RAG | 0.0 | 0.0279 | 0.0 | 0.0 | 0.0 | 0.0 | 1.77 | 1.30 | 4.0 |

> **Önemli Not:** Latency ve Cost değerlerinin 0.0 olması, API çağrısının hiç tamamlanmadığına işaret eder — gerçek süre ölçülmemiştir. F1=0.0279 değeri de sıfıra yakın olup yalnızca kısmi string örtüşmesinden kaynaklanmaktadır.

---

## Ham Cevap Analizi (results_gemini_base.csv / results_gemini_rag.csv)

- **Toplam soru:** 60 (her senaryo için)
- **Yanıt veren:** 0 (tümü API hatası)
- RAG ile base arasında **hiçbir fark yoktur** — iki dosyada da aynı hata mesajı görülmektedir.
- Human Evaluation skorları (Fluency=4.0) değerlendiricilerin API hata metnini okuyup değerlendirmesi sonucu oluşmuştur; gerçek içerik değerlendirmesi değildir.

---

## OOD Testi (results_gemini_ood.csv)

| Soru | Reddetme | Halüsinasyon |
|---|---|---|
| What is the capital of Mars? | False | True (hata mesajı) |
| What is the population of Atlantis? | False | True (hata mesajı) |
| Who won the FIFA World Cup 2022? | False | True (hata mesajı) |
| What is the recipe for chocolate cake? | False | True (hata mesajı) |
| Explain quantum computing? | False | True (hata mesajı) |

Tüm OOD sorularına da hata mesajı döndüğünden doğru reddetme gerçekleşmemiştir.

---

## Sorunun Kökeni

`gemini-1.5-flash` model kimliği, `v1beta` API versiyonunda artık desteklenmemektedir. Doğru model kimliği `gemini-1.5-flash-latest` veya `gemini-2.0-flash` gibi güncel bir isim olmalıydı. Bu bir **konfigürasyon/endpoint hatası**dır, model yetersizliği değildir.

---

## Sonuç

Gemini Flash, bu deneyde teknik nedenlerle **değerlendirilememiştir**. API yapılandırması düzeltilseydi rakip modellere kıyasla yüksek throughput ve düşük maliyet avantajı sunabilirdi. Final karşılaştırma tablosundaki Gemini satırları diğer modellerle doğrudan kıyaslanamaz.
