import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

from src import config
from src.split_data import load_baseline
from src.utils import banner, subbanner, save_fig

def run():
    banner("5b. UNDERSAMPLING PADA TRAINING DATA")
    X_train, X_test, y_train, y_test = load_baseline()
    
    subbanner("Distribusi sebelum undersampling:")
    print(y_train.value_counts())
    
    # Undersampling process
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=config.RANDOM_STATE)
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
    
    subbanner("Distribusi sesudah undersampling:")
    print(y_train_under.value_counts())
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, data) in zip(axes, [("Sebelum Undersampling", y_train),
                                        ("Sesudah Undersampling", y_train_under)]):
        sns.countplot(x=data, ax=ax)
        ax.bar_label(ax.containers[0])
        ax.set_title(title)
    plt.tight_layout()
    save_fig(config.FIGURES_DIR / "distribusi_undersampling.png")
    
    banner("6. KLASIFIKASI DECISION TREE")
    # Konfigurasi disesuaikan dengan eksperimen oversampling anggota kelompok
    dt = DecisionTreeClassifier(criterion="entropy", random_state=config.RANDOM_STATE)
    print(f"Konfigurasi Model: {dt.get_params()}")
    dt.fit(X_train_under, y_train_under)
    y_pred = dt.predict(X_test)
    
    banner("7. EVALUASI (PADA DATA TESTING)")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal (0)", "Gagal (1)"]).plot(cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Confusion Matrix (Undersampling)")
    plt.tight_layout()
    save_fig(config.FIGURES_DIR / "confusion_matrix_undersampling.png")
    
    return dt, X_train_under, y_train_under, y_test, y_pred

if __name__ == "__main__":
    run()
