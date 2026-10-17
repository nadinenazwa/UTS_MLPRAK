"""Modul Perbandingan Model (Bagian Nadine)
Menganalisis hasil dari 6 eksperimen klasifikasi:
- nb_over, nb_under
- svm_over, svm_under
- dt_over, dt_under

Menghasilkan:
1. output/results/comparison.csv
2. output/results/best_model.txt
3. output/figures/comparison_metrics.png
4. output/figures/comparison_confusion_matrices.png
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils import banner, subbanner, save_fig

MODEL_NAMES = {
    "nb": "Naive Bayes",
    "svm": "Support Vector Machine",
    "dt": "Decision Tree",
}

SCENARIO_NAMES = {
    "over": "Oversampling (BorderlineSMOTE)",
    "under": "Undersampling (RandomUnderSampler)",
}


def load_results() -> tuple[list[dict], list[str]]:
    """Membaca file hasil evaluasi JSON dari folder output/results/."""
    results = []
    missing = []
    for exp_name in config.EXPERIMENTS:
        path = config.RESULTS_DIR / f"{exp_name}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                prefix, suffix = exp_name.split("_", 1)
                data["experiment"] = exp_name
                data["model_type"] = MODEL_NAMES.get(prefix, prefix.upper())
                data["scenario"] = SCENARIO_NAMES.get(suffix, suffix)
                results.append(data)
            except Exception as e:
                print(f"[Peringatan] Gagal membaca {path}: {e}")
                missing.append(exp_name)
        else:
            missing.append(exp_name)
    return results, missing


def build_comparison_dataframe(results: list[dict]) -> pd.DataFrame:
    """Menyusun dataframe perbandingan performa model."""
    records = []
    for r in results:
        records.append({
            "Eksperimen": r["experiment"],
            "Model": r["model_type"],
            "Skenario": r["scenario"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "TN": int(r["tn"]),
            "FP": int(r["fp"]),
            "FN": int(r["fn"]),
            "TP": int(r["tp"]),
        })
    df = pd.DataFrame(records)
    return df


def plot_metric_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Membuat grafik batang perbandingan 4 metrik evaluasi."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df_melt = pd.melt(
        df,
        id_vars=["Eksperimen"],
        value_vars=metrics,
        var_name="Metrik",
        value_name="Skor",
    )

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("muted", len(metrics))
    ax = sns.barplot(
        data=df_melt,
        x="Eksperimen",
        y="Skor",
        hue="Metrik",
        palette=palette,
    )
    plt.title("Perbandingan Kinerja Model Klasifikasi (6 Skenario)", fontsize=14, pad=15)
    plt.xlabel("Eksperimen", fontsize=11)
    plt.ylabel("Skor (0.0 - 1.0)", fontsize=11)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)

    # Tambahkan label angka di atas batang
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h*100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    save_fig(output_path)


def plot_combined_confusion_matrices(results: list[dict], output_path: Path) -> None:
    """Membuat grid 2x3 visualisasi confusion matrix untuk seluruh model."""
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["Normal (0)", "Gagal (1)"],
            yticklabels=["Normal (0)", "Gagal (1)"],
        )
        title = f"{r['experiment']} ({r['model_type']})\nAcc: {r['accuracy']:.2%} | F1: {r['f1']:.2%}"
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Prediksi", fontsize=10)
        ax.set_ylabel("Aktual", fontsize=10)

    # Kosongkan subplot jika ada sisa slot
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Matriks Kebingungan (Confusion Matrix) Seluruh Model", fontsize=16, y=1.02)
    plt.tight_layout()
    save_fig(output_path)


def run() -> pd.DataFrame:
    banner("PERBANDINGAN KINERJA MODEL (BAGIAN NADINE)")
    results, missing = load_results()

    if missing:
        subbanner("Catatan Ketersediaan Hasil Eksperimen")
        print(f"File hasil berikut belum ditemukan: {', '.join(missing)}")
        if not results:
            print("Tidak ada hasil eksperimen yang dapat dibandingkan!")
            return pd.DataFrame()

    df = build_comparison_dataframe(results)
    
    # Simpan tabel perbandingan CSV
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.COMPARISON_CSV, index=False)
    print(f"\n[Hasil] Tabel perbandingan disimpan ke: {config.COMPARISON_CSV}")
    
    # Cetak tabel perbandingan rapi
    subbanner("Tabel Perbandingan Kinerja")
    df_display = df.copy()
    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        df_display[m] = (df_display[m] * 100).map("{:.2f}%".format)
    print(df_display.to_string(index=False))

    # Tentukan model terbaik berdasarkan akurasi tertinggi
    best_row = df.loc[df["Accuracy"].idxmax()]
    best_model_name = best_row["Eksperimen"]
    config.BEST_MODEL_PATH.write_text(best_model_name, encoding="utf-8")
    
    subbanner("Model Terbaik (Akurasi Tertinggi)")
    print(f"Model Terbaik: {best_model_name} ({best_row['Model']} - {best_row['Skenario']})")
    print(f"Akurasi: {best_row['Accuracy']:.2%}")
    print(f"Precision: {best_row['Precision']:.2%}, Recall: {best_row['Recall']:.2%}, F1: {best_row['F1-Score']:.2%}")
    print(f"File model terbaik tercatat di: {config.BEST_MODEL_PATH}")

    # Buat grafik visualisasi
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric_comparison(df, config.FIGURES_DIR / "comparison_metrics.png")
    plot_combined_confusion_matrices(results, config.FIGURES_DIR / "comparison_confusion_matrices.png")

    return df


if __name__ == "__main__":
    run()
