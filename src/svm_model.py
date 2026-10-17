import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from src import config
from src.evaluate import evaluate_and_save
from src.resampling import oversample, undersample

def load_baseline():
    """Memuat data latih dan uji dari folder output/baseline."""
    X_train = pd.read_csv("output/baseline/X_train.csv")
    X_test = pd.read_csv("output/baseline/X_test.csv")
    y_train = pd.read_csv("output/baseline/y_train.csv").values.ravel()
    y_test = pd.read_csv("output/baseline/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name=""):
    """Mencetak hasil evaluasi model."""
    y_pred = model.predict(X_test)
    print(f"\n--- Hasil Evaluasi: {model_name} ---")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def run_svm():
    print("=== Memulai Eksperimen Model SVM ===")
    
    # 1. Muat data baseline dari output/baseline
    print("Memuat data baseline...")
    X_train, X_test, y_train, y_test = load_baseline()
    
    # 2. Resampling data training
    print("Menerapkan teknik resampling...")
    X_train_over, y_train_over = oversample(X_train, y_train)
    X_train_under, y_train_under = undersample(X_train, y_train)
    
    # 3. Inisialisasi Model SVM
    svm_model = SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
    
    # 4. Skenario 1: Oversampling (BorderlineSMOTE)
    print("\n[1] Melatih SVM dengan Oversampling (BorderlineSMOTE)...")
    start_time = time.time()
    svm_model.fit(X_train_over, y_train_over)
    print(f"Waktu pelatihan: {time.time() - start_time:.2f} detik")
    evaluate_model(svm_model, X_test, y_test, model_name="SVM + BorderlineSMOTE")
    evaluate_and_save(svm_model, X_test, y_test, "svm_over")
    
    # 5. Skenario 2: Undersampling (RandomUnderSampler)
    print("\n[2] Melatih SVM dengan Undersampling (RandomUnderSampler)...")
    start_time = time.time()
    svm_model.fit(X_train_under, y_train_under)
    print(f"Waktu pelatihan: {time.time() - start_time:.2f} detik")
    evaluate_model(svm_model, X_test, y_test, model_name="SVM + RandomUnderSampler")
    evaluate_and_save(svm_model, X_test, y_test, "svm_under")
    
    print("\n=== Eksekusi Selesai ===")

def run():
    run_svm()

if __name__ == "__main__":
    run_svm()