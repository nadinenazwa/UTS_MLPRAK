import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
# Menggunakan BorderlineSMOTE agar unik dibandingkan SMOTE biasa
from imblearn.over_sampling import BorderlineSMOTE

def jalankan_eksperimen_oversampling():
    print("\n" + "=" * 60)
    print("EKSPERIMEN OVERSAMPLING")
    print("=" * 60)

    # 1. Load Baseline Data
    # Mengambil data dari folder output/baseline/ yang sudah disiapkan
    print("Membaca data baseline training dan testing...")
    X_train = pd.read_csv("output/baseline/X_train.csv")
    y_train = pd.read_csv("output/baseline/y_train.csv").squeeze() # squeeze untuk ubah ke Series
    X_test = pd.read_csv("output/baseline/X_test.csv")
    y_test = pd.read_csv("output/baseline/y_test.csv").squeeze()

    # 2. Cek Distribusi Kelas Sebelum Resampling
    print("\nDistribusi kelas Target (Sebelum Oversampling):")
    print(y_train.value_counts())

    # 3. Terapkan Metode Oversampling
    # Menggunakan BorderlineSMOTE dengan sampling_strategy=0.5 (minoritas jadi 50% dari mayoritas)
    print("\nMenerapkan BorderlineSMOTE pada data training...")
    smote = BorderlineSMOTE(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 4. Cek Distribusi Kelas Sesudah Resampling
    print("\nDistribusi kelas Target (Sesudah Oversampling):")
    print(y_train_res.value_counts())

    # Visualisasi perbandingan distribusi
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x=y_train, ax=ax[0], palette="muted")
    ax[0].set_title("Sebelum Oversampling")
    sns.countplot(x=y_train_res, ax=ax[1], palette="muted")
    ax[1].set_title("Sesudah BorderlineSMOTE")
    plt.tight_layout()
    plt.savefig("output/figures/distribusi_oversampling.png", dpi=150)
    print("-> Grafik distribusi disimpan di output/figures/distribusi_oversampling.png")
    plt.close()

    # 5. Training Decision Tree
    # Menggunakan criterion="entropy" sebagai variasi model agar unik
    print("\nMelatih model Decision Tree dengan data hasil oversampling...")
    dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt.fit(X_train_res, y_train_res)

    # 6. Testing dan Evaluasi Model
    print("\nMelakukan prediksi pada data testing dan evaluasi...")
    y_pred = dt.predict(X_test)

    # Menghitung 4 Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Akurasi  : {acc:.4f}")
    print(f"Presisi  : {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Gagal (1)"]))

    # 7. Membuat dan Menyimpan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Gagal"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Oversampling")
    plt.tight_layout()
    plt.savefig("output/figures/confusion_matrix.png", dpi=150)
    print("-> Grafik confusion matrix disimpan di output/figures/confusion_matrix.png")
    plt.close()

    print("\nEksperimen Oversampling Selesai!\n")

if __name__ == "__main__":
    jalankan_eksperimen_oversampling()