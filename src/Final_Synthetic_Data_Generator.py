#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — Sentetik Veri Üretimi (Gemini API ile)
# ============================================================================
# PDF §9 "Synthetic Veri Kullanımı" — isteğe bağlı ama bonus puan.
# Gemini 1.5 Flash kullanarak countries.json verilerinden çeşitli QA çiftleri
# üretir. Manuel şablonların ötesinde daha doğal, çeşitli sorular oluşturur.
#
# Neden sentetik veri?
#   - Manuel şablonlar sınırlı çeşitlilik sunar (aynı cümle yapısı)
#   - LLM ile üretilen sorular farklı ifade biçimleri içerir
#   - Fine-tuning için daha zengin eğitim verisi
#
# Kaydedilen dosyalar → final_outputs/:
#   synthetic_qa_pairs.jsonl   — Ham sentetik QA çiftleri
#   synthetic_qa_analysis.csv  — Kalite analizi
#   synthetic_training_data.jsonl — Fine-tuning için hazır format
#
# SAGLAYICI SECIMI: Asagidaki PROVIDER ile "gemini" veya "nvidia" sec.
#   - gemini : Gemini 1.5 Flash (Secret: GEMINI_API_KEY)
#   - nvidia : build.nvidia.com NIM, OpenAI-uyumlu (Secret: NVIDIA_API_KEY)
# Ikisi de yoksa otomatik demo moda duser (manuel sablon verisi).
#
# Bagimliliklar (Kaggle ilk hucre):
# !pip install -q google-generativeai openai pandas matplotlib tqdm
#
# Kaggle Secrets: PROVIDER'a gore GEMINI_API_KEY veya NVIDIA_API_KEY tanimli olmalidir.
# ============================================================================

import json, os, time, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def _resolve(fname, hardcoded=""):
    import glob as _glob
    candidates = [
        os.path.join(".", "data", fname),
        os.path.join("..", "data", fname),
        os.path.join(".", fname),
        os.path.join("..", fname),
        f"/kaggle/input/cfbook/{fname}",
        f"/kaggle/input/datasets/yunusemredal/cfbook/{fname}",
        f"/kaggle/working/{fname}",
        hardcoded,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    hits = _glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if hits:
        return hits[0]
    hits = _glob.glob(f"**/{fname}", recursive=True)
    if hits:
        return os.path.abspath(hits[0])
    return hardcoded or fname

def _resolve_dir(dname):
    for c in [f"./{dname}", f"../{dname}"]:
        if os.path.exists(c):
            return c
    return f"./{dname}"

# ── Konfigürasyon ─────────────────────────────────────────────────────────────
INPUT_JSON   = _resolve("countries.json", "/kaggle/input/datasets/yunusemredal/cfbook/countries.json")
OUTPUT_DIR = _resolve_dir("final_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  LLM SAGLAYICI SECIMI — "gemini" veya "nvidia"                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
PROVIDER        = "nvidia"        # "gemini" | "nvidia"
GEMINI_MODEL    = "gemini-1.5-flash"
NVIDIA_MODEL    = "meta/llama-3.1-8b-instruct"   # build.nvidia.com/models'tan degistirilebilir
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

TARGET_PAIRS = 500       # Hedef sentetik QA çifti sayısı
COUNTRIES_PER_BATCH = 5  # Her API çağrısında kaç ülke
RATE_LIMIT_S = 1.5       # API rate limit bekleme süresi
MIN_Q_LEN    = 15        # Minimum soru uzunluğu (karakter)
MIN_A_LEN    = 10        # Minimum cevap uzunluğu (karakter)

print("=" * 70)
print(" FINAL — SENTETİK VERİ ÜRETİMİ (Gemini API)")
print(f" Hedef: {TARGET_PAIRS} QA çifti")
print("=" * 70)

# ── LLM Saglayici Kurulumu (Gemini veya NVIDIA) ───────────────────────────────
gemini_model = None     # gemini model objesi
llm_client   = None     # nvidia (OpenAI-uyumlu) istemci
LLM_READY    = False


def _get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient().get_secret(name)
        if sec:
            return sec
    except Exception:
        pass
    return os.environ.get(name)


if PROVIDER == "gemini":
    GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(
                GEMINI_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=0.7, max_output_tokens=1024,
                ),
            )
            LLM_READY = True
            print(f"[OK] Gemini hazir: {GEMINI_MODEL}")
        except Exception as e:
            print(f"[HATA] Gemini SDK: {e}")
    else:
        print("[UYARI] GEMINI_API_KEY alinamadi.")

elif PROVIDER == "nvidia":
    NVIDIA_API_KEY = _get_secret("NVIDIA_API_KEY")
    if NVIDIA_API_KEY:
        try:
            from openai import OpenAI
            llm_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
            LLM_READY = True
            print(f"[OK] NVIDIA hazir: {NVIDIA_MODEL}")
        except Exception as e:
            print(f"[HATA] OpenAI/NVIDIA istemcisi: {e}")
    else:
        print("[UYARI] NVIDIA_API_KEY alinamadi.")
else:
    print(f"[UYARI] Bilinmeyen PROVIDER='{PROVIDER}' — demo moda dusulecek.")


def llm_generate(prompt):
    """Aktif saglayici ile metin uretir (hata olursa istisna firlatir)."""
    if PROVIDER == "gemini":
        resp = gemini_model.generate_content(prompt)
        return (resp.text or "").strip()
    # nvidia
    resp = llm_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()

# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Veri yukleniyor...")

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Her ülke için kısa metin özeti hazırla (prompt'a göndermek için)
def make_country_summary(country, details, max_chars=800):
    """Ülke bilgilerinden kısa prompt dostu özet üretir."""
    key_fields = [
        "Government: Capital - name",
        "Geography: Location",
        "Geography: Climate",
        "Geography: Terrain",
        "Geography: Natural resources",
        "People and Society: Population - total",
        "People and Society: Languages",
        "People and Society: Religions",
        "Economy: Economic overview",
        "Economy: Agricultural products",
        "Government: Government type",
        "Government: Independence",
    ]
    parts = [f"Country: {country}"]
    for k in key_fields:
        if k in details and str(details[k]).strip():
            field_name = k.split(":")[-1].strip()
            parts.append(f"{field_name}: {str(details[k])[:150]}")
    return "\n".join(parts)[:max_chars]

countries_list = list(raw_data.keys())
random.shuffle(countries_list)
print(f"[INFO] Toplam ulke: {len(countries_list)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SENTETİK QA ÜRETİM FONKSİYONU
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert at creating diverse, high-quality question-answer pairs
for training AI models about world countries.

Given information about one or more countries, generate varied and natural QA pairs.
Create questions with different styles:
- Factual: "What is the capital of X?"
- Descriptive: "Describe the terrain of X."
- Comparative: "How does the economy of X differ from typical..."
- Contextual: "Given that X has Y climate, what crops might grow well?"
- Inference: Questions that require combining multiple facts

Output ONLY a JSON array of objects with "question", "answer", "country", "type" fields.
Example format:
[
  {"question": "What currency does France use?", "answer": "France uses the Euro.", "country": "France", "type": "factual"},
  {"question": "Describe the landscape of Brazil.", "answer": "Brazil features...", "country": "Brazil", "type": "descriptive"}
]"""


def generate_synthetic_qa(country_summaries, n_per_country=4):
    """Aktif saglayici (Gemini/NVIDIA) ile bir batch ülke için sentetik QA üretir."""
    if not LLM_READY:
        return []

    batch_text = "\n\n---\n\n".join(country_summaries)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Generate {n_per_country} diverse QA pairs for EACH of the following countries. "
        f"Mix factual, descriptive, and inferential questions.\n\n"
        f"Country Information:\n{batch_text}\n\n"
        f"Generate {n_per_country * len(country_summaries)} total QA pairs as JSON array:"
    )

    for attempt in range(3):
        try:
            text = llm_generate(prompt)

            # JSON'u çıkart (dizinin tamamini al — greedy)
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                pairs = json.loads(json_match.group())
                # Temel doğrulama
                valid = []
                for p in pairs:
                    if (isinstance(p, dict) and
                        "question" in p and "answer" in p and
                        len(str(p.get("question", ""))) >= MIN_Q_LEN and
                        len(str(p.get("answer", ""))) >= MIN_A_LEN):
                        valid.append(p)
                return valid
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    [UYARI] Batch atlanadi: {e}")
    return []

# ══════════════════════════════════════════════════════════════════════════════
# 3. TOPLU ÜRETİM
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/5] Sentetik QA uretimi baslıyor (hedef: {TARGET_PAIRS})...")

all_pairs   = []
cost_total  = 0.0
n_per_c     = 4   # Her ülke için hedef QA sayısı
batch_size  = COUNTRIES_PER_BATCH

if not LLM_READY:
    print(f"[UYARI] '{PROVIDER}' API kullanilabilir degil — demo veri uretiliyor.")
    # Demo mode: Manuel şablonlardan oluşturulan örnek veri
    DEMO_TEMPLATES = [
        ("What is the capital of {c}?", "The capital of {c} is {v}.", "Government: Capital - name", "factual"),
        ("Where is {c} located?", "{c} is located in {v}.", "Geography: Location", "factual"),
        ("Describe the climate of {c}.", "The climate of {c} is: {v}.", "Geography: Climate", "descriptive"),
        ("What economy does {c} have?", "Economy of {c}: {v}.", "Economy: Economic overview", "descriptive"),
        ("What languages are spoken in {c}?", "Languages in {c}: {v}.", "People and Society: Languages", "factual"),
        ("What type of government does {c} have?", "{c} has a {v}.", "Government: Government type", "factual"),
    ]
    for country, details in list(raw_data.items())[:100]:
        for q_t, a_t, key, q_type in DEMO_TEMPLATES:
            if key in details and len(str(details[key]).strip()) > 5:
                all_pairs.append({
                    "question": q_t.format(c=country),
                    "answer":   a_t.format(c=country, v=str(details[key])[:200]),
                    "country":  country,
                    "type":     q_type,
                    "source":   "demo_template",
                })
    print(f"[INFO] Demo mod: {len(all_pairs)} QA çifti uretildi.")
else:
    # Gerçek Gemini üretimi
    batches = [countries_list[i:i+batch_size]
               for i in range(0, len(countries_list), batch_size)]

    for batch_countries in tqdm(batches, desc="Batch QA Uretimi"):
        if len(all_pairs) >= TARGET_PAIRS:
            break

        summaries = [
            make_country_summary(c, raw_data[c])
            for c in batch_countries if c in raw_data
        ]

        batch_pairs = generate_synthetic_qa(summaries, n_per_country=n_per_c)

        # Ülke etiketini doğrula
        for p in batch_pairs:
            if "country" not in p or not p["country"]:
                # Metinden ülkeyi bul
                for c in batch_countries:
                    if c.lower() in p.get("question", "").lower() or c.lower() in p.get("answer", "").lower():
                        p["country"] = c
                        break
            p["source"] = "gemini_synthetic"
            all_pairs.append(p)

        # Tahmini maliyet — yalnizca Gemini icin ucretli (~$0.075/1M in, $0.30/1M out)
        # NVIDIA build.nvidia.com ucretsiz kredi ile gelir -> maliyet 0 kabul edilir.
        if PROVIDER == "gemini":
            est_input_tokens  = sum(len(s.split()) for s in summaries) * 1.3
            est_output_tokens = len(batch_pairs) * 50
            cost_total += est_input_tokens * 0.000000075 + est_output_tokens * 0.0000003

        time.sleep(RATE_LIMIT_S)

    print(f"\n[INFO] Toplam uretilen: {len(all_pairs)} QA cifti")
    print(f"[INFO] Tahmini API maliyeti: ~${cost_total:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. KALİTE FİLTRELEME + ANALİZ
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Kalite filtreleme...")

def is_quality_pair(p):
    """Temel kalite kontrolleri."""
    q = str(p.get("question", "")).strip()
    a = str(p.get("answer", "")).strip()
    if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
        return False
    if q == a:
        return False
    if not q.endswith("?") and not any(q.startswith(w) for w in
        ["What", "Where", "When", "How", "Describe", "Explain", "Give", "List"]):
        return False
    # Çok kısa cevaplar
    if len(a.split()) < 3:
        return False
    return True

filtered_pairs = [p for p in all_pairs if is_quality_pair(p)]
removed = len(all_pairs) - len(filtered_pairs)
print(f"[INFO] Filtreleme: {len(all_pairs)} → {len(filtered_pairs)} ({removed} elendi)")

# İstatistikler
type_dist     = Counter(p.get("type", "unknown") for p in filtered_pairs)
country_dist  = Counter(p.get("country", "unknown") for p in filtered_pairs)
q_lens        = [len(p["question"].split()) for p in filtered_pairs]
a_lens        = [len(p["answer"].split()) for p in filtered_pairs]

print("\n--- Sentetik Veri İstatistikleri ---")
print(f"  Toplam QA Cifti      : {len(filtered_pairs)}")
print(f"  Ort. Soru Uzunlugu   : {np.mean(q_lens):.1f} kelime")
print(f"  Ort. Cevap Uzunlugu  : {np.mean(a_lens):.1f} kelime")
print(f"  Benzersiz Ulke       : {len(country_dist)}")
print("  Soru Tipi Dagilimi   :")
for qtype, cnt in type_dist.most_common():
    print(f"    {qtype:15s}: {cnt}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. KAYDET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Dosyalar kaydediliyor...")

# Ham JSONL
raw_path = os.path.join(OUTPUT_DIR, "synthetic_qa_pairs.jsonl")
with open(raw_path, "w", encoding="utf-8") as f:
    for p in filtered_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"[OK] synthetic_qa_pairs.jsonl — {len(filtered_pairs)} cift")

# Kalite analiz CSV
df_analysis = pd.DataFrame([{
    "country":       p.get("country", ""),
    "type":          p.get("type", "unknown"),
    "source":        p.get("source", ""),
    "q_len_words":   len(p["question"].split()),
    "a_len_words":   len(p["answer"].split()),
    "question":      p["question"][:100],
    "answer":        p["answer"][:150],
} for p in filtered_pairs])
df_analysis.to_csv(os.path.join(OUTPUT_DIR, "synthetic_qa_analysis.csv"), index=False)
print("[OK] synthetic_qa_analysis.csv")

# Fine-tuning icin hazir format (Qwen ChatML)
IMS, IME = "<|im_start|>", "<|im_end|>"
SYS = "You are a knowledgeable assistant about world countries."

def to_chatml(p):
    return {
        "text": (
            f"{IMS}system\n{SYS}{IME}\n"
            f"{IMS}user\n{p['question']}{IME}\n"
            f"{IMS}assistant\n{p['answer']}{IME}"
        )
    }

training_data = [to_chatml(p) for p in filtered_pairs]
train_path = os.path.join(OUTPUT_DIR, "synthetic_training_data.jsonl")
with open(train_path, "w", encoding="utf-8") as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"[OK] synthetic_training_data.jsonl — {len(training_data)} cift")

# ══════════════════════════════════════════════════════════════════════════════
# 6. GÖRSEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Gorseller uretiliyor...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Soru tipi dagilimi
ax = axes[0, 0]
if type_dist:
    labels, counts = zip(*type_dist.most_common())
    ax.barh(labels, counts, color=plt.cm.Set2(range(len(labels))))
    ax.set_title("Soru Tipi Dağılımı")
    ax.set_xlabel("Adet")
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(count + 0.3, i, str(count), va="center", fontsize=9)

# (2) Cevap uzunluğu histogramı
ax = axes[0, 1]
ax.hist(a_lens, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(np.mean(a_lens), color="red", linestyle="--", label=f"Ort={np.mean(a_lens):.1f}")
ax.set_title("Cevap Uzunluğu (kelime) Dağılımı")
ax.set_xlabel("Kelime Sayısı")
ax.legend()

# (3) Soru uzunluğu histogramı
ax = axes[1, 0]
ax.hist(q_lens, bins=20, color="#2ecc71", edgecolor="white", alpha=0.8)
ax.axvline(np.mean(q_lens), color="red", linestyle="--", label=f"Ort={np.mean(q_lens):.1f}")
ax.set_title("Soru Uzunluğu (kelime) Dağılımı")
ax.set_xlabel("Kelime Sayısı")
ax.legend()

# (4) En çok kapsanan ülkeler (top 15)
ax = axes[1, 1]
top_countries = country_dist.most_common(15)
if top_countries:
    c_names, c_counts = zip(*top_countries)
    ax.barh(c_names, c_counts, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(c_names))))
    ax.set_title("En Çok Kapsanan Ülkeler (Top 15)")
    ax.set_xlabel("QA Çifti Sayısı")

plt.suptitle("Sentetik Veri Kalite Analizi", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "synthetic_data_analysis.png"), dpi=150)
print("[OK] synthetic_data_analysis.png")
plt.close()

# Rapor kaydet
report = {
    "total_generated":    len(all_pairs),
    "after_filtering":    len(filtered_pairs),
    "removed":            removed,
    "unique_countries":   len(country_dist),
    "type_distribution":  dict(type_dist),
    "avg_q_length_words": round(np.mean(q_lens), 1),
    "avg_a_length_words": round(np.mean(a_lens), 1),
    "generation_method":  (f"{PROVIDER}:{NVIDIA_MODEL if PROVIDER=='nvidia' else GEMINI_MODEL}"
                            if LLM_READY else "demo_template"),
    "estimated_cost_usd": round(cost_total, 4),
    "output_files": {
        "raw":      "synthetic_qa_pairs.jsonl",
        "analysis": "synthetic_qa_analysis.csv",
        "training": "synthetic_training_data.jsonl",
    },
}
with open(os.path.join(OUTPUT_DIR, "synthetic_data_report.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print(" SENTETİK VERİ ÜRETİMİ TAMAMLANDI!")
print(f" Toplam: {len(filtered_pairs)} QA çifti")
print(f" API Maliyeti: ~${cost_total:.4f}")
print(f" Fine-tuning icin: {OUTPUT_DIR}/synthetic_training_data.jsonl")
print(" Bu veriyi fine-tuning scriptlerine 'ek veri' olarak kullanin.")
print("=" * 70)
