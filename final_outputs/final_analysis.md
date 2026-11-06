# Final Analiz ve Tartisma

## 11. Bolum — Analiz ve Tartisma

### En Iyi Embedding–Chunking Kombinasyonu
- **e5_base** + **hierarchical** (Top-K=15)
- Hit Rate@k = **1.0000**  |  MRR@k = **0.8222**

### EM & F1 Sonuclari
- **En Yuksek EM**: qwen_3b — D — RAG + Base LLM  (0.0583)
- **En Yuksek F1**: qwen_3b — D — RAG + Base LLM  (0.6106)

### Halusinasyon Analizi
- **En Dusuk Halusinasyon**: Gemini Flash — No RAG  (0.0%)
- RAG, modelin context'e bagimli yanit uretmesini saglayarak halusinasyonu azaltmaktadir.

### Latency & Trade-off
- **En Hizli Sistem**: Gemini Flash — No RAG  (0.000s)
- RAG pipeline retrieval adimi latency'yi artirir, ancak yanit kalitesi belirgin bicimde yukselir.

### Fine-Tuning Etkisi (Base LLM vs Fine-Tuned LLM)
- EM farki: +0.0000  |  F1 farki: +0.1239
- Fine-tuning tek basina belirgin iyilestirme saglamadi; RAG ile birlikte en iyi sonucu veriyor.

### RAG vs Fine-Tuning vs Hybrid

- **RAG tek basina**: Retrieval kalitesine bagli; hallucination azalir, yeni bilgiye erisim saglar.

- **Fine-Tuning tek basina**: Domain bilgisi parametrelere gomulur; RAG gerekmez ancak out-of-date kalabilir.

- **RAG + Fine-Tuned**: Genellikle en iyi performans — hem domain uyumu hem de dinamik bilgi erisimi.

- **Retriever threshold filtering**: Dusuk benzerlikli gereksiz chunk'lari keserek hallucination'i azaltir.


### Stress Test Bulgulari

- **Long/Noisy sorular**: Retriever uzun sorularda bile dogru ulkeyi genellikle ilk 7 sonucta bulmaktadir.

- **Multi-hop sorular**: Retriever multi-hop sorgularda zorluk yasiyor; gelecek calismalarda query decomposition onerilir.

- **OOD testleri**: RAG sistemi context ile sinirli olmasi sayesinde daha iyi OOD reddetme performansi gosteriyor.
