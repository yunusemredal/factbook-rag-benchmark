# Fine-Tuning — Üç Model Karşılaştırma Raporu

**İlgili dosyalar:** `finetune_summary.csv`, `finetune_report_llama3.json`, `finetune_report_qwen.json`, `finetune_report_phi35.json`

---

## Genel Bakış

Üç yerel model, aynı domain verisi (ülke ansiklopedisi QA çiftleri) üzerinde **QLoRA (4-bit, nf4)** yöntemiyle fine-tune edilmiştir. Hedef, modellerin parametrelerine domain bilgisi kazandırmak ve RAG ile kıyaslamaktır.

**Ortak Ayarlar:**
| Parametre | Değer |
|---|---|
| Yöntem | QLoRA (4-bit, nf4) |
| Veri seti büyüklüğü | 3,302 QA çifti toplam |
| Eğitim / Doğrulama | 2,971 / 331 örnek (%90/%10) |
| Epoch | 3 |
| Öğrenme oranı | 2e-4 |
| Batch size | 2 (gradient accumulation=4, efektif batch=8) |
| Max seq. length | 512 token |
| Warmup oranı | 0.05 |
| Donanım | Kaggle 2×T4 GPU |
| Framework | transformers + peft + trl |

---

## Model Karşılaştırma Tablosu

| Model | LoRA r/α | Target Modüller | Süre (dk) | GPU-saat | Maliyet ($) | Final Loss |
|---|---|---|---|---|---|---|
| LLaMA 3 8B | 16/32 | q,k,v,o_proj | 108.1 | 1.8 | 0.63 | **0.5142** |
| Qwen 2.5 3B | 16/32 | q,k,v,o_proj | **58.8** | **0.98** | **0.34** | 0.6721 |
| Phi-3.5 Mini | 32/64 | qkv,o,gate_up,down_proj | 210.9 | 3.52 | 1.23 | 2.4809 |

**Toplam fine-tuning maliyeti: $2.20** (3 model, Kaggle GPU kredisi ile)

---

## Model Detayları

### LLaMA 3 8B (finetune_report_llama3.json)
- **En düşük final loss: 0.5142** — eğitim en iyi yakınsayan model.
- 108 dakika, 1.8 GPU-saat, $0.63.
- LoRA r=16, alpha=32, 4 target modül.
- Adapter kaydedildi: `./final_finetuned/llama3`
- **Eval sonucu:** Fine-tune öncesi ve sonrası EM/F1 değişmemiş (B ≈ C, D ≈ E). Düşük loss, bu domain'deki QA formatına uyum sağlandığını gösterse de retrieval kalitesini artırmamıştır.

### Qwen 2.5 3B (finetune_report_qwen.json)
- **En verimli eğitim:** 58.8 dakika, $0.34 — grubun yarısı fiyat ve süresiyle.
- Final loss: 0.6721 (LLaMA'dan yüksek ama kabul edilebilir).
- Aynı LoRA konfigürasyonu (r=16, alpha=32).
- Adapter kaydedildi: `./final_finetuned/qwen`
- **Eval sonucu:** B ≈ C ve D ≈ E; RAG'sız fine-tuning sıfır katkı sağlamış. Zaten yüksek olan RAG performansı değişmeden kalmıştır.

### Phi-3.5 Mini (finetune_report_phi35.json)
- **Problematik eğitim:** Final loss = 2.4809 — grubun 5× üzerinde, yakınsama sağlanamamış.
- 210.9 dakika, 3.52 GPU-saat, $1.23 — en pahalı ve en yavaş.
- Daha geniş LoRA (r=32, alpha=64) ve daha fazla target modül (qkv_proj, gate_up_proj, down_proj).
- `completion_only=True` kullanılmış — sadece yanıt tokenları üzerinden loss hesabı.
- Adapter kaydedildi: `./final_finetuned/phi35`
- **Eval:** Phi-3.5 final karşılaştırmaya dahil edilmemiştir (yüksek loss nedeniyle güvenilir adapter üretilememiş).

---

## Fine-Tuning'in Eval'e Etkisi

| Model | F1 (B — Base) | F1 (C — FT) | Fark | F1 (D — RAG+Base) | F1 (E — RAG+FT) | Fark |
|---|---|---|---|---|---|---|
| LLaMA 3 8B | 0.4707 | 0.4707 | **0.0** | 0.4396 | 0.4396 | **0.0** |
| Qwen 2.5 3B | 0.3612 | 0.3612 | **0.0** | 0.6106 | 0.6106 | **0.0** |

**Kritik bulgu:** Fine-tuning, EM ve F1 metriklerinde sıfır iyileşme sağlamıştır. Bunun olası nedenleri:

1. **Veri dağılımı örtüşmesi:** Fine-tuning verisi (3,302 QA) ile eval seti (120 soru) aynı kaynak (countries.json) olduğundan model zaten benzer örüntüleri öğrenmiş olabilir.
2. **3 epoch yetersizliği:** Daha fazla epoch veya daha büyük veri seti farklı sonuç verebilirdi.
3. **QLoRA sınırlaması:** 4-bit kuantizasyon, bazı ince bilgi kazanımlarını sınırlıyor olabilir.
4. **RAG dominansı:** RAG zaten doğru bilgiyi doğrudan context'e sağladığından, fine-tuning'in ek katkısı görünür olmuyor.

---

## Maliyet-Fayda Analizi

| Model | Maliyet | Performans Artışı | Verimlilik |
|---|---|---|---|
| Qwen 2.5 3B | $0.34 | %0 EM/F1 artış | Düşük — ama en ucuz |
| LLaMA 3 8B | $0.63 | %0 EM/F1 artış | Düşük |
| Phi-3.5 Mini | $1.23 | N/A (başarısız) | Çok düşük |

Mevcut deney sonuçlarına göre fine-tuning **bu domain ve eval konfigürasyonunda yatırım getirisi sağlamamıştır.** Ancak bu, fine-tuning'in değersiz olduğu anlamına gelmez — daha uzun eğitim, daha büyük veri seti veya OOD sorularında farklı sonuçlar elde edilebilirdi.

---

## Genel Değerlendirme

Fine-tuning çalışmasının en önemli çıktısı, **RAG + Fine-Tuned kombinasyonunun RAG + Base'den fark yaratmamasıdır.** Bu, mevcut domain için bilgi retrieval'ının parametre güncellemesinden daha etkin bir bilgi aktarım mekanizması olduğunu kanıtlamaktadır. Phi-3.5 eğitimi tamamen başarısız olmuş ve raporda yalnızca negativ örnek olarak yer almaktadır.
