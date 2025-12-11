import streamlit as st
import pandas as pd
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
st.set_page_config(page_title="Obesity Risk Classifier", layout="wide")

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
    ["Prediksi Upload Dataset", "Prediksi Satuan", "Feature Importance", "Riwayat Prediksi"]
)

# ================================================================
# 1. PREDIKSI UPLOAD DATASET
# ================================================================
if menu == "Prediksi Upload Dataset":

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

        # Prediksi
        preds = model.predict(df_proc)
        pred_labels = target_encoder.inverse_transform(preds)
        df["Prediction"] = pred_labels

        # ===================================================================
        #   EVALUASI OPSIONAL → HANYA BERJALAN JIKA LABEL ADA DI DATASET
        # ===================================================================
        st.subheader("Evaluasi (Jika Label Ada)")

        if "NObeyesdad" in df.columns:

            # Coba encode label asli jika tersedia
            try:
                if "NObeyesdad" in encoders:
                    true_encoded = encoders["NObeyesdad"].transform(df["NObeyesdad"].astype(str))
                else:
                    # Buat encoder label sementara, aman dan tidak crash
                    from sklearn.preprocessing import LabelEncoder
                    lbl_tmp = LabelEncoder()
                    true_encoded = lbl_tmp.fit_transform(df["NObeyesdad"].astype(str))

                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

                # Accuracy
                acc = accuracy_score(true_encoded, preds)
                st.write(f"**Akurasi:** {acc:.4f}")

                # Classification report
                st.write("**Classification Report:**")
                report = classification_report(true_encoded, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # Confusion matrix
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(true_encoded, preds)
                st.dataframe(pd.DataFrame(cm))

                st.subheader("Distribusi Prediksi")
                st.bar_chart(df["Prediction"].value_counts())

            except Exception as e:
                st.error(f"Gagal menghitung evaluasi: {e}")

        else:
            st.info("Dataset tidak memiliki kolom label NObeyesdad → Evaluasi dilewati.")

        # ===================================================================
        #   DOWNLOAD HASIL
        # ===================================================================
        st.subheader("Preview Hasil")
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
elif menu == "Prediksi Satuan":

    st.title("Prediksi Obesitas (Input Manual)")
    input_data = {}

    # Bentuk input form otomatis dari feature_list
    for col in feature_list:
        if col in encoders and col != "target":
            input_data[col] = st.selectbox(col, list(encoders[col].classes_))
        else:
            input_data[col] = st.number_input(col, format="%.2f")

    if st.button("Prediksi"):
        df_input = pd.DataFrame([input_data])
        df_proc = preprocess(df_input.copy())

        pred = model.predict(df_proc)[0]
        probas = model.predict_proba(df_proc)[0]

        pred_label = target_encoder.inverse_transform([pred])[0]  # FIXED

        st.success(f"HASIL: **{pred_label}**")

        prob_df = pd.DataFrame({
            "Kelas": target_encoder.classes_,
            "Probabilitas": probas
        })
        st.dataframe(prob_df)

        rekom = get_rekomendasi(pred_label)
        for r in rekom:
            st.write("- ", r)

        save_prediction(
            conn,
            input_dict=input_data,
            probabilitas=float(max(probas)),
            prediksi=pred_label,
            rekomendasi="; ".join(rekom)
        )

        st.success("Disimpan ke database!")

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
        inp = json.loads(input_json)
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
    import io

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
    # Tombol hapus per baris
    # =========================================
    st.subheader("Hapus Berdasarkan ID")

    for i, row in df_hist.iterrows():
        col1, col2 = st.columns([10, 2])

        with col1:
            st.write(f"ID: {row['ID']} — Prediksi: {row['Prediksi']} — Waktu: {row['Waktu']}")

        with col2:
            if st.button("🗑️ Hapus", key=f"hapus_{row['ID']}"):
                delete_history_by_id(conn, int(row['ID']))
                st.success(f"Riwayat dengan ID {row['ID']} berhasil dihapus.")
                st.rerun()



