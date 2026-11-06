# Retriever — Embedding × Chunking × Top-K Analizi

**İlgili dosyalar:** `retriever_results.csv`, `stress_test_retriever.csv`, `pipeline_config.json`

---

## Pipeline Konfigürasyonu (pipeline_config.json)

| Parametre | Değer |
|---|---|
| En iyi koleksiyon | `final_e5_base_hierarchical` |
| En iyi embedding | `intfloat/e5-base-v2` (e5_base) |
| En iyi chunking | `hierarchical` |
| En iyi Top-K | 15 |
| En iyi threshold | None (filtreleme yok) |
| Hit Rate@k | **1.0** |
| MRR@k | **0.8222** |
| Test sorusu | 120 |
| Koleksiyon sayısı | 12 |
| Toplam çalışma süresi | 33.5 dakika |
| Vektör veritabanı | ChromaDB |

---

## Test Edilen Bileşenler

**Embedding modelleri (4 adet):**
| Kısa Ad | Tam Model Adı |
|---|---|
| e5_base | intfloat/e5-base-v2 |
| bge_base | BAAI/bge-base-en-v1.5 |
| gte_base | thenlper/gte-base |
| multilingual_mpnet | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |

**Chunking stratejileri (3 adet):** sliding_window, sentence_based, hierarchical

**Top-K değerleri:** 1, 7, 15

**Similarity threshold'lar:** None, 0.5, 0.7, 0.9

**Toplam kombinasyon:** 4 × 3 × 3 × 4 = 144 deney

---

## En İyi 10 Konfigürasyon (retriever_results.csv)

| Hit Rate | MRR | Recall | Latency (ms) | Embedding | Chunking | Top-K | Threshold |
|---|---|---|---|---|---|---|---|
| **1.0** | 0.8222 | 1.0 | 2.54 | e5_base | hierarchical | 15 | None |
| **1.0** | 0.8576 | 1.0 | 2.46 | gte_base | hierarchical | 15 | 0.9 |
| **1.0** | 0.8576 | 1.0 | 2.50 | gte_base | hierarchical | 15 | None |
| **1.0** | 0.8277 | 1.0 | 2.54 | bge_base | hierarchical | 15 | None |
| **1.0** | 0.8222 | 1.0 | 2.23 | e5_base | hierarchical | 7 | 0.9 |
| 0.9917 | 0.8566 | 0.9917 | 2.14 | gte_base | hierarchical | 7 | None |
| 0.9917 | 0.8267 | 0.9917 | 2.22 | bge_base | hierarchical | 7 | None |
| 0.9833 | 0.8236 | 0.9833 | 2.68 | e5_base | sliding_window | 15 | None |
| 0.9750 | 0.7720 | 0.9750 | 2.46 | multilingual_mpnet | hierarchical | 15 | None |

### Temel Bulgular

**1. Chunking stratejisi belirleyici:** `hierarchical` chunking tüm embed modellerinde en yüksek Hit Rate'i vermiştir. `sentence_based` en zayıf; `sliding_window` ortada.

**2. Top-K'nın önemi:** Top-K=1 ile Hit Rate dramatik düşüyor (0.27–0.77). Top-K=7 ve 15 arasında küçük fark var; 15, 7'den hafif üstün.

**3. gte_base en yüksek MRR:** `gte_base + hierarchical + Top-K=15` → MRR=0.8576 (e5_base'in 0.8222'sinden daha iyi sıralama). Ancak proje konfigürasyonu olarak e5_base seçilmiş.

**4. Threshold zararlı olabilir:** `multilingual_mpnet` ile threshold (0.5/0.7/0.9) uygulandığında Hit Rate **0.0**'a düşüyor — bu embedding, bu eşik değerleriyle çok fazla chunk filtreliyor. `bge_base + hierarchical + threshold=0.5` da benzer şekilde Hit Rate=0.275.

**5. En tehlikeli kombinasyonlar:**
- `multilingual_mpnet + sliding_window/sentence_based + herhangi threshold` → Hit Rate = 0.0
- `bge_base + sentence_based + threshold=0.5` → Hit Rate = 0.14

---

## Embedding Modeli Karşılaştırması (hierarchical, Top-K=15)

| Embedding | Hit Rate | MRR | Latency (ms) |
|---|---|---|---|
| gte_base | **1.0** | **0.8576** | 2.50 |
| e5_base | 1.0 | 0.8222 | 2.54 |
| bge_base | 1.0 | 0.8277 | 2.54 |
| multilingual_mpnet | 0.9750 | 0.7720 | 2.46 |

`multilingual_mpnet` İngilizce domain'de en zayıf performansı vermiştir. Bu model çok dilli metin için tasarlanmış olup tek dilli İngilizce doküman koleksiyonunda dezavantaj yaşamıştır.

---

## Stres / Dayanıklılık Testi (stress_test_retriever.csv)

**Toplam test:** 23 soru (5 kategori)

### Kategori Bazlı Retrieval Başarısı

| Kategori | Soru Sayısı | Doğru Bulunan | Başarı % |
|---|---|---|---|
| paraphrase | 5 | 5 | **%100** |
| multi_hop | 3 | 1 | **%33** |
| ambiguous | 3 | 2 | **%67** |
| long_noisy | 5 | 5 | **%100** |
| OOD | 5 | 0 | — (bilgi KB'de yok) |

### Kategori Detayları

**Paraphrase (%100):** Yeniden ifade edilmiş sorularda retriever hiç yanılmamıştır. "Tell me the capital of France" → France dokümanını doğru bulmuştur.

**Long & Noisy (%100):** Gürültülü, uzun sorular retrieval'ı bozmamıştır. Örnek: "Despite all the noise and misinformation... about Asian countries... Japan's population" → Japonya dokümanını doğru çekmiştir.

**Ambiguous (%67):** "Georgia" ve "Jordan" sorgularında doğru ülkeyi çekmiş; "Turkey's natural resources" sorgusunda Türkiye dokümanını bulmuş ama Faithfulness zayıf kalmış.

**Multi-hop (%33 — kritik zayıflık):**
- "Country bordering both China and India in South Asia" → Nepal ✅
- "Most populous country in Africa's languages" → Nijerya yerine Orta Afrika Cumhuriyeti, Kongo, Namibya ❌
- "European country with largest area's economy" → Fransa yerine San Marino, Litvanya, Monako ❌

Multi-hop sorgular, tek bir embedding ile doğrudan çözülememektedir. Bu tür sorular için **query decomposition** (sorguyu alt sorulara bölme) veya **chain-of-thought retrieval** gereklidir.

**OOD:** Mars, Atlantis gibi hayali yerler için retriever KB'deki alakasız ülkeleri döndürmektedir — bu beklenen bir davranış. OOD reddetme LLM katmanında yapılmalıdır.

---

## Genel Değerlendirme

Retriever, standart sorularda çok güçlü performans sergilemiştir (Hit Rate=1.0). Temel zayıflık multi-hop sorgulardır. `e5_base + hierarchical + Top-K=15` seçimi iyi bir karardır; `gte_base` alternatif olarak daha yüksek MRR sunar. Threshold kullanımından (özellikle `multilingual_mpnet` ile) kaçınmak gereklidir. Re-ranking mevcut deneyde uygulanmamıştır; eklenmesi MRR'yi artırabilir.
