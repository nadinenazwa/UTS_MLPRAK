"""Aplikasi Prediksi Predictive Maintenance & Perbandingan Model (Streamlit)
Bagian Tugas: Nadine Nazwa Andina (Pertemuan 8 - Klasifikasi Machine Learning Praktikum)
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Predictive Maintenance Classification",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .normal-box {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #065F46;
    }
    .failure-box {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #991B1B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_preprocessing():
    """Memuat artefak preprocessing (scaler, bounds IQR, mapping type)."""
    if not config.PREPROCESS_PATH.exists():
        return None
    return joblib.load(config.PREPROCESS_PATH)


@st.cache_resource
def load_model(model_name: str):
    """Memuat model terklasifikasi joblib."""
    model_path = config.MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def get_best_model_name() -> str:
    """Membaca nama model terbaik dari output/results/best_model.txt."""
    if config.BEST_MODEL_PATH.exists():
        return config.BEST_MODEL_PATH.read_text(encoding="utf-8").strip()
    return "dt_over"


def transform_input(raw_dict: dict, prep_artifact: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Pipeline transformasi konsisten dengan pelatihan:
    1. Mapping kategori Type
    2. Clipping IQR numerik
    3. Scaling dengan MinMaxScaler
    """
    df_raw = pd.DataFrame([raw_dict])
    df_clipped = df_raw.copy()
    
    # 1. Encoding Type
    type_map = prep_artifact["type_mapping"]
    df_clipped["Type"] = df_clipped["Type"].map(type_map)

    # 2. Clipping IQR
    bounds = prep_artifact["bounds"]
    for col, (low, high) in bounds.items():
        if col in df_clipped.columns:
            df_clipped[col] = df_clipped[col].clip(lower=low, upper=high)

    # Pastikan urutan fitur tepat
    feature_order = prep_artifact["feature_order"]
    df_ordered = df_clipped[feature_order]

    # 3. Scaling
    scaler = prep_artifact["scaler"]
    scaled_array = scaler.transform(df_ordered)
    df_scaled = pd.DataFrame(scaled_array, columns=feature_order)

    return df_raw, df_clipped, scaled_array


# Header Aplikasi
st.markdown('<div class="main-title">⚙️ Klasifikasi Pemeliharaan Prediktif (Predictive Maintenance)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    '<b>Praktikum Machine Learning — Pertemuan 8</b> | '
    'Implementasi & Perbandingan: <b>Naïve Bayes</b>, <b>SVM</b>, dan <b>Decision Tree</b> | '
    'Penyusun: <i>Nadine Nazwa Andina</i>'
    '</div>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/maintenance.png", width=70)
st.sidebar.title("Navigasi & Konfigurasi")

prep = load_preprocessing()
best_model_name = get_best_model_name()

# Menampilkan informasi model terbaik di sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Model Terbaik Aktual")
st.sidebar.info(
    f"**{best_model_name.upper()}** (Akurasi Tertinggi)\n\n"
    "Dipilih otomatis dari hasil evaluasi data uji 2.000 sampel baseline UTS."
)

menu = st.sidebar.radio(
    "Pilih Menu Tampilan:",
    ["🔮 Prediksi Status Mesin", "📊 Perbandingan Model (6 Eksperimen)", "📑 Panduan & Pipeline Preprocessing"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Praktikum © 2026")

# =========================================================================
# HALAMAN 1: PREDIKSI STATUS MESIN
# =========================================================================
if menu == "🔮 Prediksi Status Mesin":
    st.subheader("🔮 Form Prediksi Status Mesin Industri")
    st.write(
        "Masukkan parameter operasi mesin di bawah ini untuk mendeteksi apakah mesin berada dalam kondisi **Normal** "
        "atau berisiko mengalami **Kegagalan (Failure)**."
    )

    if prep is None:
        st.error(
            f"Artefak preprocessing tidak ditemukan di `{config.PREPROCESS_PATH}`. "
            "Silakan jalankan `python main.py` terlebih dahulu untuk menghasilkan artefak."
        )
        st.stop()

    # Pilihan Model yang Digunakan untuk Prediksi
    col_m1, col_m2 = st.columns([2, 2])
    with col_m1:
        model_options = config.EXPERIMENTS
        default_idx = model_options.index(best_model_name) if best_model_name in model_options else 0
        selected_model_name = st.selectbox(
            "Pilih Model Klasifikasi:",
            model_options,
            index=default_idx,
            help="Secara default menggunakan model dengan akurasi tertinggi sesuai arahan tugas.",
        )
    with col_m2:
        model_label_map = {
            "dt_over": "Decision Tree + Oversampling (BorderlineSMOTE) [REKOMENDASI]",
            "dt_under": "Decision Tree + Undersampling (RandomUnderSampler)",
            "svm_over": "Support Vector Machine + Oversampling (BorderlineSMOTE)",
            "svm_under": "Support Vector Machine + Undersampling (RandomUnderSampler)",
            "nb_over": "Naïve Bayes (GaussianNB) + Oversampling",
            "nb_under": "Naïve Bayes (GaussianNB) + Undersampling",
        }
        st.info(f"**Konfigurasi Aktif:** {model_label_map.get(selected_model_name, selected_model_name)}")

    model = load_model(selected_model_name)
    if model is None:
        st.error(f"File model `{selected_model_name}.joblib` tidak ditemukan.")
        st.stop()

    st.markdown("---")

    # Quick Preset Buttons
    st.markdown("##### ⚡ Preset Contoh Uji Cepat:")
    cp1, cp2, cp3 = st.columns([2, 2, 4])
    
    # Inisialisasi session state untuk nilai input
    if "input_type" not in st.session_state:
        st.session_state.input_type = "M"
        st.session_state.input_air = 298.2
        st.session_state.input_proc = 308.7
        st.session_state.input_rpm = 1500
        st.session_state.input_torque = 40.0
        st.session_state.input_wear = 50

    if cp1.button("🟢 Muat Contoh Mesin Normal"):
        st.session_state.input_type = "M"
        st.session_state.input_air = 298.2
        st.session_state.input_proc = 308.7
        st.session_state.input_rpm = 1550
        st.session_state.input_torque = 38.0
        st.session_state.input_wear = 20
        st.rerun()

    if cp2.button("🔴 Muat Contoh Mesin Kritis/Gagal"):
        st.session_state.input_type = "L"
        st.session_state.input_air = 303.5
        st.session_state.input_proc = 312.0
        st.session_state.input_rpm = 1250
        st.session_state.input_torque = 65.0
        st.session_state.input_wear = 220
        st.rerun()

    # Form Input Parameter Mesin
    with st.form("prediction_form"):
        st.markdown("##### 📥 Parameter Operasi Mesin:")
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            inp_type = st.selectbox(
                "Kategori Kualitas Produk (Type):",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(st.session_state.input_type),
                help="L = Low (50%), M = Medium (30%), H = High (20%)",
            )
            inp_air = st.number_input(
                "Suhu Udara Lingkungan (Air temperature) [K]:",
                min_value=290.0,
                max_value=315.0,
                value=float(st.session_state.input_air),
                step=0.1,
                help="Rentang normal operasi sekitar 295 - 305 Kelvin.",
            )
            inp_proc = st.number_input(
                "Suhu Proses Kerja (Process temperature) [K]:",
                min_value=300.0,
                max_value=325.0,
                value=float(st.session_state.input_proc),
                step=0.1,
                help="Rentang normal operasi sekitar 305 - 315 Kelvin.",
            )

        with f_col2:
            inp_rpm = st.number_input(
                "Kecepatan Putaran Poros (Rotational speed) [rpm]:",
                min_value=1000,
                max_value=3000,
                value=int(st.session_state.input_rpm),
                step=10,
                help="Kecepatan rotasi mesin dalam RPM.",
            )
            inp_torque = st.number_input(
                "Torsi Beban Mesin (Torque) [Nm]:",
                min_value=1.0,
                max_value=90.0,
                value=float(st.session_state.input_torque),
                step=0.5,
                help="Beban torsi mesin dalam Newton-meter.",
            )
            inp_wear = st.number_input(
                "Akumulasi Keausan Pahat (Tool wear) [min]:",
                min_value=0,
                max_value=300,
                value=int(st.session_state.input_wear),
                step=1,
                help="Waktu akumulasi pemakaian pahat dalam menit sebelum diganti.",
            )

        submit_btn = st.form_submit_button("🚀 Prediksi Status Mesin", use_container_width=True)

    if submit_btn:
        input_data = {
            "Type": inp_type,
            "Air temperature [K]": inp_air,
            "Process temperature [K]": inp_proc,
            "Rotational speed [rpm]": inp_rpm,
            "Torque [Nm]": inp_torque,
            "Tool wear [min]": inp_wear,
        }

        # Pipeline transformasi fitur
        df_raw, df_clipped, scaled_arr = transform_input(input_data, prep)
        scaled_df = pd.DataFrame(scaled_arr, columns=prep["feature_order"])

        # Prediksi
        pred = int(model.predict(scaled_df)[0])
        
        # Hitung probabilitas jika didukung
        has_proba = hasattr(model, "predict_proba")
        proba = None
        if has_proba:
            try:
                proba = model.predict_proba(scaled_df)[0]
            except Exception:
                has_proba = False

        st.markdown("---")
        st.subheader("📢 Hasil Klasifikasi & Analisis Status")

        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if pred == 0:
                st.markdown(
                    """
                    <div class="normal-box">
                        <h2 style="margin:0; color:#065F46;">✅ MESIN DALAM KONDISI NORMAL</h2>
                        <p style="font-size:1.1rem; margin-top:8px;">
                            Tidak terdeteksi indikasi kerusakan struktural atau keausan abnormal.
                            Mesin aman untuk terus beroperasi.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="failure-box">
                        <h2 style="margin:0; color:#991B1B;">⚠️ PERINGATAN: MESIN BERISIKO GAGAL (FAILURE)</h2>
                        <p style="font-size:1.1rem; margin-top:8px;">
                            Kombinasi parameter operasi mengindikasikan potensi kerusakan tinggi.
                            Disarankan segera lakukan inspeksi dan pemeliharaan preventif!
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown(f"**Model Penguji:** `{selected_model_name}`")
            st.markdown(f"**Prediksi Kelas:** `{pred}` ({'Gagal / Rusak' if pred == 1 else 'Normal'})")
            
            if has_proba and proba is not None:
                st.markdown("**Tingkat Keyakinan (Confidence Score):**")
                p_normal = proba[0] * 100
                p_fail = proba[1] * 100
                st.write(f"Normal: **{p_normal:.1f}%**")
                st.progress(float(proba[0]))
                st.write(f"Gagal: **{p_fail:.1f}%**")
                st.progress(float(proba[1]))

        # Detail transparansi pipeline
        with st.expander("🔍 Transparansi Pipeline Preprocessing (Inspeksi Fitur)"):
            st.write("1. **Data Masukan Asli:**")
            st.dataframe(df_raw, use_container_width=True)
            
            st.write("2. **Setelah Encoding Type & Clipping IQR:**")
            st.dataframe(df_clipped, use_container_width=True)

            st.write("3. **Setelah Normalisasi MinMaxScaler (Input Akhir Model):**")
            st.dataframe(pd.DataFrame(scaled_arr, columns=prep["feature_order"]), use_container_width=True)


# =========================================================================
# HALAMAN 2: PERBANDINGAN MODEL (6 EKSPERIMEN)
# =========================================================================
elif menu == "📊 Perbandingan Model (6 Eksperimen)":
    st.subheader("📊 Perbandingan Evaluasi 6 Eksperimen Klasifikasi")
    st.write(
        "Sesuai ketentuan tugas, evaluasi membandingkan 3 metode klasifikasi (**Naïve Bayes**, **SVM**, dan **Decision Tree**) "
        "yang masing-masing diuji pada dua skenario resampling: **Oversampling (BorderlineSMOTE)** dan **Undersampling (RandomUnderSampler)**."
    )

    # Muat tabel perbandingan CSV
    if not config.COMPARISON_CSV.exists():
        st.warning(
            f"Tabel perbandingan `{config.COMPARISON_CSV}` belum dibuat. "
            "Menjalankan modul perbandingan sekarang..."
        )
        from src import compare
        compare.run()

    df_comp = pd.read_csv(config.COMPARISON_CSV)

    # Highlight Card Metrik Utama
    best_row = df_comp.loc[df_comp["Accuracy"].idxmax()]
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("🏆 Akurasi Tertinggi", f"{best_row['Accuracy']*100:.2f}%", f"{best_row['Eksperimen']}")
    with kpi2:
        st.metric("🎯 Precision Model Terbaik", f"{best_row['Precision']*100:.2f}%", "dt_over")
    with kpi3:
        st.metric("📡 Recall Model Terbaik", f"{best_row['Recall']*100:.2f}%", "dt_over")
    with kpi4:
        st.metric("⚖️ F1-Score Model Terbaik", f"{best_row['F1-Score']*100:.2f}%", "dt_over")

    st.markdown("---")

    # Tabel Evaluasi
    st.markdown("##### 📋 Tabel Perbandingan Kinerja Lengkap:")
    df_table = df_comp.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        df_table[col] = (df_table[col] * 100).map("{:.2f}%".format)
    st.dataframe(df_table, use_container_width=True)

    st.markdown("---")

    # Visualisasi Grafik Metrik
    st.markdown("##### 📈 Grafik Perbandingan 4 Metrik:")
    metrics_fig = config.FIGURES_DIR / "comparison_metrics.png"
    if metrics_fig.exists():
        st.image(str(metrics_fig), caption="Perbandingan Akurasi, Presisi, Recall, dan F1-Score pada 6 Eksperimen", use_container_width=True)
    else:
        st.info("Grafik metrik sedang disiapkan.")

    st.markdown("---")

    # Visualisasi Confusion Matrix
    st.markdown("##### 🧩 Visualisasi Matriks Kebingungan (Confusion Matrix):")
    cm_all_fig = config.FIGURES_DIR / "comparison_confusion_matrices.png"
    if cm_all_fig.exists():
        st.image(str(cm_all_fig), caption="Grid Confusion Matrix Seluruh Model (Data Uji = 2.000 Sampel, 68 Kasus Gagal Aktual)", use_container_width=True)

    st.markdown("---")

    # Analisis & Pembahasan Bagian Nadine
    st.markdown("##### 💡 Analisis dan Temuan Utama (Bagian Nadine):")
    st.markdown(
        """
        1. **Model dengan Akurasi Tertinggi:**
           - Model **Decision Tree dengan Oversampling (BorderlineSMOTE)** memperoleh akurasi tertinggi sebesar **96,95%**, presisi **53,40%**, recall **80,88%**, dan F1-score **64,33%**.
           - Model ini berhasil mengidentifikasi **55 dari 68 mesin gagal** dengan hanya 48 kesalahan alarm palsu (False Positive).

        2. **Perbandingan dengan SVM dan Naïve Bayes:**
           - **SVM (RBF Kernel)** dengan Oversampling mencapai akurasi **94,60%** dan recall **85,29%** (berhasil menangkap 58 mesin gagal), namun memiliki False Positive lebih tinggi (98 sampel) sehingga presisinya turun menjadi **37,18%**.
           - **Naïve Bayes (GaussianNB)** menghasilkan akurasi lebih rendah (**87,95%** pada oversampling dan **78,20%** pada undersampling) dengan jumlah False Positive yang signifikan (220 dan 429 sampel). Hal ini terjadi karena asumsi independensi fitur GaussianNB tidak sepenuhnya terpenuhi pada hubungan fisika mesin (misalnya relasi kuat antara kecepatan rotasi dan torsi).

        3. **Trade-off Presisi vs Recall dalam Predictive Maintenance:**
           - Pada industri manufaktur, **False Negative (FN)** jauh lebih mahal dan berbahaya dibandingkan False Positive (FP). Jika model memprediksi mesin normal padahal sebenarnya akan rusak, mesin dapat mengalami downtime mendadak yang fatal.
           - Skenario Undersampling menghasilkan **Recall sangat tinggi** (hingga 94,12% pada DT dan 89,71% pada SVM), namun dengan kompensasi **Presisi yang anjlok** (22,86% pada DT dan 16,35% pada SVM) akibat banyaknya alarm palsu.
           - Oleh karena itu, skenario **Oversampling (BorderlineSMOTE)** terbukti memberikan keseimbangan (F1-score) terbaik antara sensitivitas deteksi dan efisiensi operasional.
        """
    )


# =========================================================================
# HALAMAN 3: PANDUAN & PIPELINE PREPROCESSING
# =========================================================================
elif menu == "📑 Panduan & Pipeline Preprocessing":
    st.subheader("📑 Detail Dataset & Pipeline Klasifikasi")

    st.markdown(
        """
        ### 1. Karakteristik Dataset
        - **Dataset:** *Predictive Maintenance Dataset* (`predictive_maintenance.csv`)
        - **Jumlah Sampel:** 10.000 baris, 10 kolom awal.
        - **Target:** Kolom `Target` (0 = Normal [96,61%], 1 = Gagal/Failure [3,39%]).
        - **Ketidakseimbangan Kelas:** Rasio minoritas sangat ekstrem (~1:28,5).

        ### 2. Fitur yang Digunakan
        Setelah pembuangan kolom ID (`UDI`, `Product ID`) dan label leakage (`Failure Type`), terdapat **6 fitur utama**:
        1. `Type`: Tipe kualitas mesin (`L=0, M=1, H=2`).
        2. `Air temperature [K]`: Suhu lingkungan sekitar mesin.
        3. `Process temperature [K]`: Suhu operasional komponen proses.
        4. `Rotational speed [rpm]`: Kecepatan putaran spindel mesin.
        5. `Torque [Nm]`: Torsi torsi mekanis poros.
        6. `Tool wear [min]`: Akumulasi waktu pakai pahat.

        ### 3. Tahapan Preprocessing & Transformasi (UTS Baseline)
        - **Missing Value & Duplikat:** Tidak ditemukan nilai kosong dan duplikasi pada dataset.
        - **Penanganan Outlier:** Menerapkan metode **clipping IQR (faktor 1.5)** pada seluruh fitur numerik untuk menjaga integritas data tanpa membuang baris minoritas.
        - **Skalasi Fitur:** Normalisasi dengan **MinMaxScaler** (rentang 0 hingga 1).
        - **Pembagian Data:** **80% Training (8.000 sampel)** dan **20% Testing (2.000 sampel)** dengan stratifikasi label (`stratify=y`) dan `random_state=42`. Data testing murni dijaga tanpa resampling.

        ### 4. Teknik Resampling (Khusus Data Latih)
        - **Oversampling:** `BorderlineSMOTE(sampling_strategy=0.5, random_state=42)` meningkatkan sampel gagal menjadi 50% dari mayoritas.
        - **Undersampling:** `RandomUnderSampler(sampling_strategy=1.0, random_state=42)` menyeimbangkan rasio kelas menjadi 1:1.
        """
    )
