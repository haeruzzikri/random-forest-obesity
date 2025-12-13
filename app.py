import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import joblib
from database import create_connection, create_table, save_prediction, load_history, delete_all_history, delete_history_by_id

# =========================================
# LOAD MODEL + ENCODER + FEATURE LIST
# =========================================
model = joblib.load("model_rf.pkl")
encoders = joblib.load("encoders.pkl")         # dict: includes categorical encoders + 'target'
feature_list = joblib.load("feature_list.pkl") # urutan fitur sesuai training

target_encoder = encoders["NObeyesdad"]   # FIXED

# =========================================
# STREAMLIT CONFIG
# =========================================
st.set_page_config(page_title="Klasifikasi Resiko Obesitas Random Forest", layout="wide")

conn = create_connection()
create_table(conn)

# =========================================
# NORMALISASI & PREPROCESSING
# =========================================
def preprocess(df: pd.DataFrame):
    df = df.copy()

    required = feature_list[:]     # target tidak ada di feature_list

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required feature: {col}")

    # encode ONLY categorical (do not encode 'target')
    for col in encoders:
        if col == "target":
            continue
        if col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))

    df = df[required]  # urutan harus sama seperti training
    return df


# =========================================
# REKOMENDASI
# =========================================
def get_rekomendasi(label):
    rekom = {
        "Insufficient_Weight": [
            "Tingkatkan asupan kalori harian.",
            "Konsumsi makanan tinggi protein.",
            "Periksa status nutrisi secara berkala."
        ],
        "Normal_Weight": [
            "Pertahankan pola makan sehat.",
            "Tetap rutin berolahraga.",
            "Cek berat badan tiap 2–4 minggu."
        ],
        "Overweight_Level_I": [
            "Kurangi makanan tinggi gula.",
            "Tingkatkan aktivitas fisik harian.",
        ],
        "Overweight_Level_II": [
            "Kurangi kalori 300–500 kkal.",
            "Olahraga teratur.",
        ],
        "Obesity_Type_I": [
            "Kurangi kalori 500–700 kkal/hari.",
            "Olahraga aerobik 150 menit/minggu.",
        ],
        "Obesity_Type_II": [
            "Mulai program diet terstruktur.",
            "Konsultasi ahli gizi."
        ],
        "Obesity_Type_III": [
            "Penanganan intensif oleh medis.",
            "Waspadai risiko hipertensi/diabetes."
        ]
    }
    return rekom.get(label, ["Tidak ada rekomendasi."])


# =========================================
# MENU
# =========================================
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Prediksi Data Massal", "Prediksi Data Satuan", "Feature Importance", "Evaluasi Model", "Riwayat Prediksi"]

)

# ================================================================
# 1. PREDIKSI UPLOAD DATASET
# ================================================================
if menu == "Prediksi Data Massal":

    st.title("Prediksi Obesitas (Upload CSV)")

    file = st.file_uploader("Upload file CSV (separator ,)", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file, sep=',')
        except:
            st.error("File CSV tidak valid")
            st.stop()

        # Samakan nama kolom dengan feature_list
        df_columns_lower = {col.lower(): col for col in df.columns}
        mapped = {}

        for f in feature_list:
            if f.lower() in df_columns_lower:
                mapped[df_columns_lower[f.lower()]] = f
            else:
                st.error(f"Dataset tidak memiliki fitur wajib: {f}")
                st.stop()

        df = df.rename(columns=mapped)

        st.subheader("Dataset Awal")
        st.dataframe(df.head())

        # Preprocess data
        try:
            df_proc = preprocess(df.copy())
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        # Prediksi tanpa evaluasi
        preds = model.predict(df_proc)
        pred_labels = target_encoder.inverse_transform(preds)
        df["Prediction"] = pred_labels

        # ============================================================
        #   PREVIEW & DOWNLOAD HASIL
        # ============================================================
        st.subheader("Preview Hasil Prediksi")
        st.dataframe(df.head())

        st.download_button(
            "Download Hasil Prediksi",
            df.to_csv(index=False).encode(),
            "hasil_prediksi.csv",
            "text/csv"
        )

# ================================================================
# 2. INPUT MANUAL
# ================================================================
elif menu == "Prediksi Data Satuan":

    st.title("Prediksi Obesitas (Input Manual)")
    st.info("Isi seluruh data sesuai kondisi Anda. Semua field wajib diisi.")

    input_data = {}

    with st.form("form_prediksi_manual"):

        # ===== URUTAN INPUT SESUAI PERMINTAAN =====
        input_data["Age"] = st.number_input(
            "Age (Usia)",
            min_value=1,
            max_value=100
        )

        input_data["Gender"] = st.selectbox(
            "Gender",
            encoders["Gender"].classes_
        )

        input_data["Height"] = st.number_input(
            "Height (meter)",
            min_value=1.0,
            max_value=2.5,
            help="Contoh: 1.62"
        )

        input_data["Weight"] = st.number_input(
            "Weight (kg)",
            min_value=10.0,
            max_value=300.0,
            help="Contoh: 64"
        )

        input_data["CALC"] = st.selectbox(
            "CALC (Konsumsi alkohol)",
            encoders["CALC"].classes_,
            help="Frekuensi konsumsi alkohol"
        )

        input_data["FAVC"] = st.selectbox(
            "FAVC (Sering konsumsi makanan tinggi kalori)",
            encoders["FAVC"].classes_
        )

        input_data["FCVC"] = st.slider(
            "FCVC (Konsumsi sayur)",
            1, 3,
            help="1 = jarang, 3 = sering"
        )

        input_data["NCP"] = st.slider(
            "NCP (Jumlah makan utama per hari)",
            1, 4
        )

        input_data["SCC"] = st.selectbox(
            "SCC (Monitoring kalori)",
            encoders["SCC"].classes_
        )

        input_data["SMOKE"] = st.selectbox(
            "SMOKE (Merokok)",
            encoders["SMOKE"].classes_
        )

        input_data["CH2O"] = st.slider(
            "CH2O (Konsumsi air)",
            1, 3,
            help="1 = sedikit, 3 = banyak"
        )

        input_data["family_history_with_overweight"] = st.selectbox(
            "Riwayat keluarga overweight",
            encoders["family_history_with_overweight"].classes_
        )

        input_data["FAF"] = st.slider(
            "FAF (Aktivitas fisik)",
            0, 3,
            help="0 = tidak pernah, 3 = sering"
        )

        input_data["TUE"] = st.slider(
            "TUE (Waktu penggunaan teknologi)",
            min_value=0,
            max_value=2,
            help="TUE = Time Using Technology, yaitu waktu penggunaan layar (HP, laptop, TV) per hari.\n\n0 = < 1 jam\n1 = 1–3 jam\n2 = > 3 jam"
        )

        input_data["CAEC"] = st.selectbox(
            "CAEC (Makan di luar waktu makan)",
            encoders["CAEC"].classes_
        )

        input_data["MTRANS"] = st.selectbox(
            "MTRANS (Transportasi)",
            encoders["MTRANS"].classes_
        )

        submit = st.form_submit_button("Prediksi")

    # ================= VALIDASI & PREDIKSI =================
    if submit:

        # Validasi wajib
        if input_data["Height"] <= 0 or input_data["Weight"] <= 0:
            st.error("❌ Height dan Weight wajib diisi dengan benar")
            st.stop()

        # Hitung BMI otomatis
        input_data["BMI"] = input_data["Weight"] / (input_data["Height"] ** 2)

        df_input = pd.DataFrame([input_data])
        df_proc = preprocess(df_input.copy())

        pred = model.predict(df_proc)[0]
        probas = model.predict_proba(df_proc)[0]

        pred_label = target_encoder.inverse_transform([pred])[0]

        st.success(f"**HASIL PREDIKSI: {pred_label}**")

        prob_df = pd.DataFrame({
            "Kelas": target_encoder.classes_,
            "Probabilitas": probas
        }).sort_values("Probabilitas", ascending=False)

        st.subheader("Probabilitas Prediksi")
        st.dataframe(prob_df)

        rekom = get_rekomendasi(pred_label)
        st.subheader("Rekomendasi")
        for r in rekom:
            st.write("•", r)

        save_prediction(
            conn,
            input_dict=input_data,
            probabilitas=float(max(probas)),
            prediksi=pred_label,
            rekomendasi="; ".join(rekom)
        )

        st.success("Hasil prediksi berhasil disimpan ke database!")


# ================================================================
# 3. FEATURE IMPORTANCE
# ================================================================
elif menu == "Feature Importance":

    st.title("Feature Importance")

    fi = model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_list,
        "Importance": fi
    }).sort_values("Importance", ascending=False)

    st.bar_chart(df_imp.set_index("Feature"))
    st.dataframe(df_imp)

# ================================================================
# 4. RIWAYAT
# ================================================================
elif menu == "Riwayat Prediksi":

    st.title("Riwayat Prediksi")

    rows = load_history(conn)

    if len(rows) == 0:
        st.info("Belum ada riwayat prediksi.")
        st.stop()

    # Tombol hapus semua
    if st.button("🗑️ Hapus Semua Riwayat"):
        delete_all_history(conn)
        st.success("Semua riwayat berhasil dihapus.")
        st.rerun()

    # Format ulang data riwayat
    formatted = []
    for id, input_json, prob, pred, rekom, ts in rows:
        try:
            inp = json.loads(input_json) 
        except Exception:
            inp = {}                       

        formatted.append({
            "ID": id,
            **inp,
            "Probabilitas": prob,
            "Prediksi": pred,
            "Rekomendasi": rekom,
            "Waktu": ts
        })

    df_hist = pd.DataFrame(formatted)

    # =========================================
    # 🔽 Download XLSX (letakkan sebelum tabel)
    # =========================================
    output = io.BytesIO()
    df_hist.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="⬇️ Download Semua Riwayat (Excel)",
        data=output,
        file_name="riwayat_prediksi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # =========================================
    # Tampilkan tabel
    # =========================================
    st.subheader("Data Riwayat")
    st.dataframe(df_hist, use_container_width=True)

    # =========================================
    # Hapus berdasarkan ID (AMAN)
    # =========================================
    st.subheader("Hapus Riwayat")

    selected_id = st.selectbox(
        "Pilih ID yang akan dihapus",
        df_hist["ID"].tolist()
    )

    if st.button("🗑️ Hapus Riwayat Terpilih"):
        delete_history_by_id(conn, int(selected_id))
        st.success(f"Riwayat dengan ID {selected_id} berhasil dihapus.")
        st.rerun()

# ================================================================
# 5. EVALUASI MODEL (BARU)
# ================================================================
elif menu == "Evaluasi Model":

    st.title("Evaluasi Model")

    try:
        evaluation = joblib.load("evaluation.pkl")
    except:
        st.error("File evaluation.pkl tidak ditemukan. Pastikan sudah dibuat saat training.")
        st.stop()

    # ================================
    # Tampilkan Metric Utama
    # ================================
    st.subheader("Ringkasan Metrik")

    metrik_cols = st.columns(4)
    with metrik_cols[0]:
        st.metric("Accuracy", f"{evaluation.get('accuracy', 0):.4f}")
    with metrik_cols[1]:
        st.metric("Precision", f"{evaluation.get('precision', 0):.4f}")
    with metrik_cols[2]:
        st.metric("Recall", f"{evaluation.get('recall', 0):.4f}")
    with metrik_cols[3]:
        st.metric("F1-Score", f"{evaluation.get('f1', 0):.4f}")

    if "auc" in evaluation:
        st.metric("AUC", f"{evaluation['auc']:.4f}")

    # ================================
    # Classification Report
    # ================================
    st.subheader("Classification Report")
    if "classification_report" in evaluation:
        cr_df = pd.DataFrame(evaluation["classification_report"]).transpose()
        st.dataframe(cr_df)
    else:
        st.info("Tidak ada classification_report di evaluation.pkl")

    # ================================
    # Confusion Matrix (Heatmap)
    # ================================
    st.subheader("Confusion Matrix")

    if "confusion_matrix" in evaluation:

        cm = evaluation["confusion_matrix"]

        # Ambil class names jika ada
        classes = evaluation.get("class_names", [])

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=classes if len(classes) > 0 else "auto",
            yticklabels=classes if len(classes) > 0 else "auto",
            linewidths=.5,
            linecolor='black'
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix", fontsize=14, pad=10)

        st.pyplot(fig)

    else:
        st.info("Tidak ada confusion_matrix di evaluation.pkl")

    # ================================
    # ROC Curve
    # ================================
    st.subheader("ROC Curve (One-vs-Rest)")

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, class_name in enumerate(evaluation["class_names"]):
        ax.plot(
            evaluation["roc_curve"]["fpr"][i],
            evaluation["roc_curve"]["tpr"][i],
            label=class_name
        )

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    st.pyplot(fig)





