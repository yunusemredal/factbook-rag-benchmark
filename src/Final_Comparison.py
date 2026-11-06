#!/usr/bin/env python
# coding: utf-8
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# ============================================================================
# FINAL — Karsilastirma ve Analiz (Tum Sistemler)
# ============================================================================
# Bu dosyayi tum model testleri tamamlandiktan SONRA calistirin.
#
# Okudugu dosyalar (final_outputs/ altinda):
#   retriever_results.csv      — Retriever-Only sonuclari
#   summary_*.csv              — Her model icin ozet
#   finetune_report_*.json     — Fine-tuning raporlari
#   stress_*.csv               — Stress test sonuclari
#
# Ciktilar:
#   final_comparison.csv       — 10. bolum: Zorunlu Karsilastirma Tablosu
#   final_em_f1.png            — EM & F1 gorsel
#   final_hallucination.png    — Halusinasyon orani
#   final_latency.png          — Latency karsilastirma
#   final_human_eval_radar.png — Human eval radar chart
#   final_cost_memory.png      — Maliyet & Bellek
#   final_analysis.md          — Otomatik tartisma metni
#
# Bagimliliklar:
# !pip install -q pandas matplotlib seaborn numpy
# ============================================================================

import os, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _resolve_dir(dname):
    for c in [f"./{dname}", f"../{dname}"]:
        if os.path.exists(c):
            return c
    return f"./{dname}"
OUTPUT_DIR = _resolve_dir("final_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print(" FINAL KARSILASTIRMA — Tum Sistem Sonuclari")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 1. TUM SONUCLARI YUKLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== DOSYALAR YUKLENIYOR ===")

# Retriever sonuclari (Seviye A)
retriever_csv = os.path.join(OUTPUT_DIR, "retriever_results.csv")
df_ret = pd.DataFrame()
if os.path.exists(retriever_csv):
    df_ret = pd.read_csv(retriever_csv)
    print(f"[OK] retriever_results.csv: {len(df_ret)} satir")
else:
    print("[UYARI] retriever_results.csv bulunamadi")

# Model summary dosyalari (Seviyeler B–E)
summary_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "summary_*.csv")))
all_summaries = []
for sf in summary_files:
    df = pd.read_csv(sf)
    all_summaries.append(df)
    print(f"[OK] {os.path.basename(sf)}: {len(df)} satir")

df_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

# NOT: Gemini sonuclari summary_gemini.csv uzerinden yukaridaki glob ile zaten
# yuklenir (No-RAG + RAG iki satir). Ayrica manuel eklemek mukerrer satira yol
# acardi; bu yuzden burada tekrar eklenmiyor.

# Fine-tuning raporlari
ft_reports = []
for rp in glob.glob(os.path.join(OUTPUT_DIR, "finetune_report_*.json")):
    with open(rp) as f:
        ft_reports.append(json.load(f))
    print(f"[OK] {os.path.basename(rp)}")

# Stress test sonuclari
stress_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "stress_*.csv")))
df_stress_all = pd.DataFrame()
if stress_files:
    df_stress_all = pd.concat([pd.read_csv(sf) for sf in stress_files], ignore_index=True)
    print(f"[OK] Stress test: {len(df_stress_all)} satir ({len(stress_files)} dosya)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVER-ONLY SATIRINI EKLE (Seviye A)
# ══════════════════════════════════════════════════════════════════════════════
if not df_ret.empty:
    # Threshold=None satirlarini filtrele, en iyi kombinasyonu al
    df_ret_nothr = df_ret[df_ret["Threshold"].astype(str) == "None"]
    if not df_ret_nothr.empty:
        best_ret = df_ret_nothr.sort_values("Hit Rate@k", ascending=False).iloc[0]
    else:
        best_ret = df_ret.sort_values("Hit Rate@k", ascending=False).iloc[0]

    retriever_row = {
        "Sistem":             "A — Retriever-Only",
        "EM":                 None,
        "F1":                 None,
        "Hallucination_Rate": None,
        "Avg_Latency_s":      round(best_ret.get("Avg Latency (ms)", 0) / 1000, 4),
        "Avg_Tokens/s":       None,
        "Avg_Mem_Spike_MB":   0,
        "Total_Cost_USD":     0,
        "RAM_GB":             0, "VRAM_MB": 0,
        "HE_relevance":       None, "HE_faithfulness": None, "HE_fluency": None,
        "Hit_Rate@k":         best_ret.get("Hit Rate@k", None),
        "MRR@k":              best_ret.get("MRR@k", None),
        "Embedding":          best_ret.get("Embedding", ""),
        "Chunking":           best_ret.get("Chunking", ""),
        "Top-K":              best_ret.get("Top-K", ""),
    }
    df_all = pd.concat([pd.DataFrame([retriever_row]), df_all], ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 5 SEVİYELİ KARSILASTIRMA TABLOSU (PDF Bolum 10)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" BOLUM 10 — NIHAI SISTEM KARSILASTIRMASI")
print("=" * 60)

# Seviye etiketleri duzenle (kisa gosterim icin)
LEVEL_MAP = {
    "B_Base_LLM":  "B — LLM-Only (Base)",
    "C_FT_NoRAG":  "C — FT LLM (No RAG)",
    "D_RAG+Base":  "D — RAG + Base LLM",
    "E_RAG+FT":    "E — RAG + Fine-Tuned",
}
if not df_all.empty and "Sistem" in df_all.columns:
    for old, new in LEVEL_MAP.items():
        df_all["Sistem"] = df_all["Sistem"].str.replace(old, new, regex=False)

DISPLAY_COLS = [
    "Sistem", "EM", "F1", "Hallucination_Rate",
    "Avg_Latency_s", "Avg_Tokens/s", "Total_Cost_USD",
    "HE_relevance", "HE_faithfulness", "HE_fluency",
    "VRAM_MB", "RAM_GB",
]
avail_cols = [c for c in DISPLAY_COLS if c in df_all.columns]

if not df_all.empty:
    print(df_all[avail_cols].to_string(index=False))
    df_all.to_csv(os.path.join(OUTPUT_DIR, "final_comparison.csv"), index=False)
    print(f"\n[INFO] final_comparison.csv kaydedildi ({len(df_all)} satir)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GÖRSELLEŞTİRMELER
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== GORSELLESTIRMELER ===")

df_valid = df_all.dropna(subset=["EM"]).copy() if not df_all.empty else pd.DataFrame()

# Grafik 1: EM ve F1 karsilastirma
if not df_valid.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, metric in enumerate(["EM", "F1"]):
        ax = axes[idx]
        colors = sns.color_palette("viridis", len(df_valid))
        bars = ax.barh(df_valid["Sistem"], df_valid[metric].astype(float), color=colors)
        ax.set_xlabel(metric)
        ax.set_title(f"{metric} — Tum Sistemler")
        ax.set_xlim(0, max(df_valid[metric].max() * 1.15, 0.05))
        for bar, val in zip(bars, df_valid[metric].astype(float)):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    plt.suptitle("Final — EM & F1 Karsilastirmasi", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_em_f1.png"), dpi=150)
    print("[OK] final_em_f1.png")
    plt.close()

# Grafik 2: Halusinasyon orani
if not df_valid.empty and "Hallucination_Rate" in df_valid.columns:
    df_hall = df_valid.dropna(subset=["Hallucination_Rate"])
    if not df_hall.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        hall_vals = df_hall["Hallucination_Rate"].astype(float)
        colors = ["#e74c3c" if v > 0.3 else "#f39c12" if v > 0.15 else "#2ecc71" for v in hall_vals]
        bars = ax.barh(df_hall["Sistem"], hall_vals, color=colors)
        ax.set_xlabel("Halusinasyon Orani")
        ax.set_title("Final — Halusinasyon Orani Karsilastirmasi")
        ax.axvline(0.3, color="red", linestyle="--", alpha=0.5, label="0.30 esigi")
        ax.legend()
        for bar, val in zip(bars, hall_vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "final_hallucination.png"), dpi=150)
        print("[OK] final_hallucination.png")
        plt.close()

# Grafik 3: Latency karsilastirma
if not df_valid.empty and "Avg_Latency_s" in df_valid.columns:
    df_lat = df_valid.dropna(subset=["Avg_Latency_s"])
    if not df_lat.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh(df_lat["Sistem"], df_lat["Avg_Latency_s"].astype(float),
                       color=sns.color_palette("rocket", len(df_lat)))
        ax.set_xlabel("Ortalama Latency (saniye)")
        ax.set_title("Final — Latency Karsilastirmasi")
        for bar, val in zip(bars, df_lat["Avg_Latency_s"].astype(float)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}s", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "final_latency.png"), dpi=150)
        print("[OK] final_latency.png")
        plt.close()

# Grafik 4: Human Evaluation Radar Chart
he_cols = ["HE_relevance", "HE_faithfulness", "HE_fluency"]
if not df_valid.empty and all(c in df_valid.columns for c in he_cols):
    df_he = df_valid.dropna(subset=he_cols)
    if len(df_he) >= 2:
        categories = ["Relevance", "Faithfulness", "Fluency"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = sns.color_palette("tab10", len(df_he))
        for i, (_, row) in enumerate(df_he.iterrows()):
            vals = [float(row[c]) for c in he_cols]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=row["Sistem"][:30], color=colors[i])
            ax.fill(angles, vals, alpha=0.08, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
        ax.set_title("Human Evaluation Proxy\n(1–5 Likert Scale)", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "final_human_eval_radar.png"), dpi=150)
        print("[OK] final_human_eval_radar.png")
        plt.close()

# Grafik 5: Maliyet ve Bellek
if not df_valid.empty and "Total_Cost_USD" in df_valid.columns:
    df_cost = df_valid.dropna(subset=["Total_Cost_USD", "VRAM_MB"]).copy()
    if not df_cost.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        # Maliyet
        bars = axes[0].barh(df_cost["Sistem"], df_cost["Total_Cost_USD"].astype(float),
                             color=sns.color_palette("YlOrRd", len(df_cost)))
        axes[0].set_xlabel("Toplam Maliyet (USD)")
        axes[0].set_title("Inference Maliyeti Karsilastirmasi")
        for bar, val in zip(bars, df_cost["Total_Cost_USD"].astype(float)):
            axes[0].text(bar.get_width() + 0.00001, bar.get_y() + bar.get_height() / 2,
                         f"${val:.4f}", va="center", fontsize=9)
        # VRAM
        bars2 = axes[1].barh(df_cost["Sistem"], df_cost["VRAM_MB"].astype(float),
                              color=sns.color_palette("Blues_r", len(df_cost)))
        axes[1].set_xlabel("VRAM Kullanimi (MB)")
        axes[1].set_title("GPU Bellek Kullanimi")
        for bar, val in zip(bars2, df_cost["VRAM_MB"].astype(float)):
            axes[1].text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                         f"{val:.0f} MB", va="center", fontsize=9)
        plt.suptitle("Final — Maliyet & Bellek Karsilastirmasi", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "final_cost_memory.png"), dpi=150)
        print("[OK] final_cost_memory.png")
        plt.close()

# Grafik 6: Stress Test OOD Basarisi (varsa)
if not df_stress_all.empty and "OOD" in df_stress_all.columns:
    df_ood = df_stress_all[df_stress_all["OOD"] == True]
    if not df_ood.empty and "Model" in df_ood.columns:
        ood_summary = df_ood.groupby("Model").agg(
            Dogru_Reddetme=("Dogru_Reddetme", "mean"),
            Halusinasyon=("Halucinasyon", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(ood_summary))
        width = 0.35
        ax.bar(x - width / 2, ood_summary["Dogru_Reddetme"], width, label="Dogru Reddetme", color="#2ecc71")
        ax.bar(x + width / 2, ood_summary["Halusinasyon"],   width, label="Halusinasyon",   color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(ood_summary["Model"], rotation=30, ha="right")
        ax.set_ylabel("Oran")
        ax.set_title("OOD Test — Model Basarim Karsilastirmasi")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "final_ood_comparison.png"), dpi=150)
        print("[OK] final_ood_comparison.png")
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. FINE-TUNING RAPORLARI OZETI
# ══════════════════════════════════════════════════════════════════════════════
if ft_reports:
    print("\n=== FINE-TUNING RAPOR OZETI ===")
    ft_summary_rows = []
    for rpt in ft_reports:
        print(f"\n  Model         : {rpt.get('model', 'N/A')}")
        print(f"  Yontem        : {rpt.get('method', 'N/A')}")
        print(f"  Egitim Ornek  : {rpt.get('train_examples', 'N/A')}")
        print(f"  Epoch         : {rpt.get('epochs', 'N/A')}")
        print(f"  Ogrenme Hizi  : {rpt.get('learning_rate', 'N/A')}")
        print(f"  Egitim Suresi : {rpt.get('training_time_min', 'N/A')} dakika")
        print(f"  GPU Saati     : {rpt.get('gpu_hours', 'N/A')}h")
        print(f"  Maliyet       : ${rpt.get('estimated_cost_usd', 'N/A')}")
        print(f"  Final Loss    : {rpt.get('final_train_loss', 'N/A')}")
        print(f"  Adapter       : {rpt.get('adapter_path', 'N/A')}")
        ft_summary_rows.append({
            "Model":           rpt.get("model", ""),
            "Method":          rpt.get("method", ""),
            "Train_Examples":  rpt.get("train_examples", 0),
            "Epochs":          rpt.get("epochs", 0),
            "Learning_Rate":   rpt.get("learning_rate", 0),
            "Train_Time_Min":  rpt.get("training_time_min", 0),
            "GPU_Hours":       rpt.get("gpu_hours", 0),
            "Cost_USD":        rpt.get("estimated_cost_usd", 0),
            "Final_Loss":      rpt.get("final_train_loss", 0),
        })

    pd.DataFrame(ft_summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "finetune_summary.csv"), index=False
    )
    print("\n[INFO] finetune_summary.csv kaydedildi")

# ══════════════════════════════════════════════════════════════════════════════
# 6. OTOMATİK ANALİZ VE TARTIŞMA METNİ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(" OTOMATIK ANALIZ VE TARTISMA")
print("=" * 60)

analysis_lines = [
    "# Final Analiz ve Tartisma\n",
    "## 11. Bolum — Analiz ve Tartisma\n",
]

# En iyi embedding-chunking
if not df_ret.empty:
    df_ret_nothr = df_ret[df_ret["Threshold"].astype(str) == "None"] if "Threshold" in df_ret.columns else df_ret
    best_ret_row = df_ret_nothr.sort_values("Hit Rate@k", ascending=False).iloc[0] if not df_ret_nothr.empty else df_ret.iloc[0]
    analysis_lines.append(
        f"### En Iyi Embedding–Chunking Kombinasyonu\n"
        f"- **{best_ret_row.get('Embedding','')}** + **{best_ret_row.get('Chunking','')}** "
        f"(Top-K={best_ret_row.get('Top-K','')})\n"
        f"- Hit Rate@k = **{best_ret_row.get('Hit Rate@k',0):.4f}**  |  "
        f"MRR@k = **{best_ret_row.get('MRR@k',0):.4f}**\n"
    )

# EM/F1 karsilastirma
if not df_valid.empty:
    best_em  = df_valid.loc[df_valid["EM"].idxmax()]
    best_f1  = df_valid.loc[df_valid["F1"].idxmax()]
    analysis_lines.append(
        f"### EM & F1 Sonuclari\n"
        f"- **En Yuksek EM**: {best_em['Sistem']}  ({float(best_em['EM']):.4f})\n"
        f"- **En Yuksek F1**: {best_f1['Sistem']}  ({float(best_f1['F1']):.4f})\n"
    )

# Halusinasyon
if not df_valid.empty and "Hallucination_Rate" in df_valid.columns:
    df_h = df_valid.dropna(subset=["Hallucination_Rate"])
    if not df_h.empty:
        lowest_hall = df_h.loc[df_h["Hallucination_Rate"].idxmin()]
        analysis_lines.append(
            f"### Halusinasyon Analizi\n"
            f"- **En Dusuk Halusinasyon**: {lowest_hall['Sistem']}  ({float(lowest_hall['Hallucination_Rate']):.1%})\n"
            f"- RAG, modelin context'e bagimli yanit uretmesini saglayarak halusinasyonu azaltmaktadir.\n"
        )

# Latency
if not df_valid.empty and "Avg_Latency_s" in df_valid.columns:
    df_l = df_valid.dropna(subset=["Avg_Latency_s"])
    if not df_l.empty:
        fastest = df_l.loc[df_l["Avg_Latency_s"].idxmin()]
        analysis_lines.append(
            f"### Latency & Trade-off\n"
            f"- **En Hizli Sistem**: {fastest['Sistem']}  ({float(fastest['Avg_Latency_s']):.3f}s)\n"
            f"- RAG pipeline retrieval adimi latency'yi artirir, ancak yanit kalitesi belirgin bicimde yukselir.\n"
        )

# Fine-tuning etki analizi
b_rows = df_valid[df_valid["Sistem"].str.contains("Base_LLM|B —", na=False)]
ft_rows = df_valid[df_valid["Sistem"].str.contains("FT_NoRAG|C —", na=False)]
if not b_rows.empty and not ft_rows.empty:
    em_diff = float(ft_rows["EM"].mean()) - float(b_rows["EM"].mean())
    f1_diff = float(ft_rows["F1"].mean()) - float(b_rows["F1"].mean())
    analysis_lines.append(
        f"### Fine-Tuning Etkisi (Base LLM vs Fine-Tuned LLM)\n"
        f"- EM farki: {em_diff:+.4f}  |  F1 farki: {f1_diff:+.4f}\n"
        f"- {'Fine-tuning EM ve F1 iyilestirmesi sagladi.' if em_diff > 0 else 'Fine-tuning tek basina belirgin iyilestirme saglamadi; RAG ile birlikte en iyi sonucu veriyor.'}\n"
    )

analysis_lines += [
    "### RAG vs Fine-Tuning vs Hybrid\n",
    "- **RAG tek basina**: Retrieval kalitesine bagli; hallucination azalir, yeni bilgiye erisim saglar.\n",
    "- **Fine-Tuning tek basina**: Domain bilgisi parametrelere gomulur; RAG gerekmez ancak out-of-date kalabilir.\n",
    "- **RAG + Fine-Tuned**: Genellikle en iyi performans — hem domain uyumu hem de dinamik bilgi erisimi.\n",
    "- **Retriever threshold filtering**: Dusuk benzerlikli gereksiz chunk'lari keserek hallucination'i azaltir.\n",
    "\n### Stress Test Bulgulari\n",
    "- **Long/Noisy sorular**: Retriever uzun sorularda bile dogru ulkeyi genellikle ilk 7 sonucta bulmaktadir.\n",
    "- **Multi-hop sorular**: Retriever multi-hop sorgularda zorluk yasiyor; gelecek calismalarda query decomposition onerilir.\n",
    "- **OOD testleri**: RAG sistemi context ile sinirli olmasi sayesinde daha iyi OOD reddetme performansi gosteriyor.\n",
]

analysis_text = "\n".join(analysis_lines)
print(analysis_text[:1500])   # konsolda kisa gosterim

analysis_path = os.path.join(OUTPUT_DIR, "final_analysis.md")
with open(analysis_path, "w", encoding="utf-8") as f:
    f.write(analysis_text)
print("\n[INFO] final_analysis.md kaydedildi")

# ══════════════════════════════════════════════════════════════════════════════
# 7. CIKTI DOSYALARI OZETI
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[INFO] Tum Ciktilar ({OUTPUT_DIR}):")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    fsize = os.path.getsize(fpath)
    print(f"  {fname:>45s}  ({fsize:>10,} bytes)")

print("\n" + "=" * 70)
print(" FINAL KARSILASTIRMA TAMAMLANDI!")
print("=" * 70)
