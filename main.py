import argparse
import joblib

from src import init_dataset, preprocessing, transformation, split_data, naive_bayes, decision_tree, compare, config
from src.svm_model import run_svm
from src.utils import quiet


STAGES = [
    "init",
    "preprocessing",
    "transformation",
    "split",
    "models",
    "compare",
]


def main(stage: str = "all") -> None:
    """Menjalankan pipeline sesuai stage yang dipilih."""

    # Menentukan tahap terakhir yang perlu dijalankan.
    if stage == "all":
        last = len(STAGES)
    else:
        last = STAGES.index(stage) + 1

    # Menentukan apakah output suatu tahap perlu ditampilkan.
    def show(name: str) -> bool:
        return stage == "all" or stage == name

    # Menjalankan tahap dengan atau tanpa output.
    def call_stage(name: str, fn, *args):
        if show(name):
            return fn(*args)

        return quiet(fn, *args)

    # =========================================================
    # 1. INIT DATASET
    # =========================================================
    df_raw = call_stage("init", init_dataset.run)

    if last == 1:
        return

    # =========================================================
    # 2. PREPROCESSING
    # =========================================================
    df_clean = call_stage(
        "preprocessing",
        preprocessing.run,
        df_raw,
    )

    if last == 2:
        return

    # =========================================================
    # 3. TRANSFORMATION
    # =========================================================
    X, y, scaler = call_stage(
        "transformation",
        transformation.run,
        df_clean,
    )

    # Simpan artefak preprocessing untuk UI/UX aplikasi (Bagian Nadine)
    bounds = preprocessing.compute_bounds(df_raw)
    preprocess_artifact = {
        "scaler": scaler,
        "bounds": bounds,
        "type_mapping": config.TYPE_MAPPING,
        "feature_order": [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocess_artifact, config.PREPROCESS_PATH)
    if show("transformation"):
        print(f"[Artefak] Preprocessing disimpan di: {config.PREPROCESS_PATH}")

    if last == 3:
        return

    # =========================================================
    # 4. SPLIT DATA
    # =========================================================
    call_stage(
        "split",
        split_data.run,
        X,
        y,
    )

    if last == 4:
        return

    # =========================================================
    # 5. EKSEKUSI MODEL & EKSPERIMEN
    # =========================================================
    if stage == "all" or stage == "models":
        print(
            "\nBaseline selesai: "
            "X_train, X_test, y_train, y_test siap dipakai.\n"
        )
        
        # 1. Jalankan eksperimen Naive Bayes 
        print("=== Menjalankan Eksperimen Naive Bayes ===")
        if hasattr(naive_bayes, "run"):
            naive_bayes.run()
        elif hasattr(naive_bayes, "run_naive_bayes"):
            naive_bayes.run_naive_bayes()
        
        # 2. Jalankan eksperimen SVM 
        print("\n=== Menjalankan Eksperimen SVM ===")
        run_svm()

        # 3. Jalankan eksperimen Decision Tree (UTS)
        print("\n=== Menjalankan Eksperimen Decision Tree ===")
        decision_tree.run()

    if last == 5:
        return

    # =========================================================
    # 6. PERBANDINGAN MODEL & REKOMENDASI (Bagian Nadine)
    # =========================================================
    if stage == "all" or stage == "compare":
        print("\n=== Menjalankan Modul Perbandingan (Bagian Nadine) ===")
        compare.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Klasifikasi Predictive Maintenance."
    )

    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Tahap pipeline yang ingin ditampilkan.",
    )

    args = parser.parse_args()

    main(args.stage)