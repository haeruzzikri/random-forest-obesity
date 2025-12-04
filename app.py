import streamlit as st
import pandas as pd
import io
import joblib
from sklearn.preprocessing import LabelEncoder
from database import create_connection, create_table, save_prediction, load_history


st.set_page_config(page_title="Prediksi Risko Obesitas Random Forest", layout="wide")

# ============================
# Load Model + Encoder + Feature List
# ============================
model = joblib.load("model_rf.pkl")
encoders = joblib.load("encoders.pkl")  # dict, e.g. {"gender": LabelEncoder()}
feature_list = joblib.load("feature_list.pkl")  # e.g. ['age','gender','height','weight','bmi']

conn = create_connection()
create_table(conn)


# ============================
# Helper: normalize gender values to match encoder classes
# ============================
def normalize_gender_series(s: pd.Series, encoder):
    """
    Normalize typical user inputs to encoder classes.
    - If encoder.classes_ contain readable gender labels (have letters), map case-insensitive.
    - If encoder.classes_ are numeric-like (e.g. '1','2'), map 'male'->classes_[0], 'female'->classes_[1]
      and show a warning to the user about the assumed mapping.
    """
    s = s.astype(str).str.strip()
    # Lowercase for initial normalization
    s_low = s.str.lower().replace({
        'male.': 'male', 'female.': 'female'
    })

    encoder_classes = list(encoder.classes_)
    encoder_classes_lower = [str(c).lower() for c in encoder_classes]

    # If encoder classes contain alphabetic characters, try to map case-insensitive
    has_alpha = any(any(ch.isalpha() for ch in str(c)) for c in encoder_classes)

    # mapping function
    def map_val(v):
        if pd.isna(v):
            return v
        v_l = str(v).lower()
        if has_alpha:
            # try direct case-insensitive match to encoder classes
            if v_l in encoder_classes_lower:
                return encoder_classes[encoder_classes_lower.index(v_l)]
            # common synonyms
            if v_l in ['male', 'm']:
                for c in encoder_classes:
                    if str(c).lower().startswith('m'):
                        return c
            if v_l in ['female', 'f']:
                for c in encoder_classes:
                    if str(c).lower().startswith('f'):
                        return c
            # fallback: return original (will be validated later)
            return v
        else:
            # encoder classes are numeric-like (e.g., '1','2') -> assume first=male, second=female
            # Inform user via warning outside this function
            if v_l in ['male', 'm']:
                return encoder_classes[0] if len(encoder_classes) >= 1 else v
            if v_l in ['female', 'f']:
                return encoder_classes[1] if len(encoder_classes) >= 2 else v
            # if user provided a numeric-like value that matches encoder, return it
            if v in encoder_classes:
                return v
            return v

    mapped = s_low.apply(map_val)
    return mapped

# ============================
# Preprocessing Function (robust)
# ============================
def preprocess(df: pd.DataFrame, show_warnings=False):
    df = df.copy()

    # Drop label if present
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Ensure all required columns exist
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. "
                         f"Required columns: {feature_list}")

    # Normalize gender if present and encoder exists
    if "gender" in df.columns and "gender" in encoders:
        enc = encoders["gender"]
        # If encoder classes numeric-like, inform the user about assumed mapping
        has_alpha = any(any(ch.isalpha() for ch in str(c)) for c in enc.classes_)
        if not has_alpha and show_warnings:
            st.warning(
                "Warning: encoder untuk kolom 'gender' berisi kelas: "
                f"{list(enc.classes_)}. Aplikasi akan mengasumsikan mapping "
                f"'male' -> {enc.classes_[0]} dan 'female' -> {enc.classes_[1]}."
            )
        df["gender"] = normalize_gender_series(df["gender"], enc)

    # Apply encoders safely (validate unseen labels and give clear message)
    for col, enc in encoders.items():
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in input dataframe.")
        # Check unseen labels
        unique_vals = set(df[col].astype(str).unique())
        allowed = set([str(c) for c in enc.classes_])
        unseen = unique_vals - allowed
        if unseen:
            # try case-insensitive match
            allowed_lower_map = {str(c).lower(): c for c in enc.classes_}
            replacements = {}
            for u in list(unseen):
                if u.lower() in allowed_lower_map:
                    replacements[u] = allowed_lower_map[u.lower()]
                    unseen.remove(u)
            if replacements:
                df[col] = df[col].astype(str).replace(replacements)
                unique_vals = set(df[col].astype(str).unique())
                unseen = unique_vals - allowed

            if unseen:
                raise ValueError(
                    f"Kolom '{col}' mengandung nilai yang tidak dikenali: {sorted(list(unseen))}. "
                    f"Harus salah satu dari: {sorted(list(allowed))}."
                )
        # transform
        df[col] = enc.transform(df[col].astype(str))

    # Reorder columns to match training order and keep only feature_list
    df = df[feature_list]

    # Ensure numeric types where appropriate (convert if possible)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    return df

# ============================
# Sidebar Menu
# ============================
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Prediksi Upload Dataset", "Prediksi Satuan", "Feature Importance", "Riwayat Prediksi"]
)

# ============================
# MENU 1: Prediksi File Dataset
# ============================
if menu == "Prediksi Upload Dataset":
    st.title("Prediksi Obesitas (Upload Dataset)")
    st.write("Upload file CSV Anda (dipisahkan dengan `;`). Pastikan file hanya berisi fitur berikut: age, gender, height, weight, dan bmi")
    st.write("Pastikan kolom gender pada data anda berisi 1 dan 2, dimana 1 adalah male dan 2 adalah female!")

    file = st.file_uploader("Upload CSV disini", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file, sep=';')
        except Exception:
            st.error("Gagal membaca file. Pastikan format CSV benar dan separator adalah ';'")
            st.stop()

        st.subheader("Preview data (5 baris pertama):")
        st.dataframe(df.head(), use_container_width=True)

        # Drop label if present (we don't want it sent into model)
        if "label" in df.columns:
            df = df.drop(columns=["label"])

        # Preprocess with warnings allowed
        try:
            df_processed = preprocess(df.copy(), show_warnings=True)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        # Predict
        try:
            preds = model.predict(df_processed)
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            st.stop()

        # Attach predictions to original (non-preprocessed) dataframe
        out = df.copy()
        out["Prediction"] = preds
        st.success("Prediksi selesai!")
        st.subheader("Hasil Prediksi (preview):")
        st.dataframe(out.head(), use_container_width=True)

        # Download full results
        st.download_button(
            label="Download Hasil Prediksi (CSV)",
            data=out.to_csv(index=False).encode(),
            file_name="hasil_prediksi.csv",
            mime="text/csv",
        )

# ============================
# MENU 2: Prediksi Satuan
# ============================
if menu == "Prediksi Satuan":
    st.title("Prediksi Obesitas (Form Input)")

    st.write("Masukkan nilai dari setiap fitur berikut:")

    input_data = {}

    for col in feature_list:
        if col == "gender":
            input_data[col] = st.selectbox("Gender", ["male", "female"])
        else:
            input_data[col] = st.number_input(col, format="%.2f")

    if st.button("Prediksi"):
        df_input = pd.DataFrame([input_data])

        # Preprocessing
        df_processed = preprocess(df_input.copy())

        # Predict
        pred = model.predict(df_processed)[0]
        probas = model.predict_proba(df_processed)[0]

        st.success(f"Hasil Prediksi: **{pred}**")

        # ======================
        # Probabilitas
        # ======================
        st.subheader("Probabilitas Tiap Kelas")
        prob_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas": probas
        })
        st.dataframe(prob_df, use_container_width=True)

        # ======================
        # Rekomendasi Gaya Hidup
        # ======================
        st.subheader("Rekomendasi Gaya Hidup")

        rekom = {
            "Underweight": [
                "Tingkatkan asupan kalori harian.",
                "Fokus pada makanan tinggi protein dan karbohidrat kompleks.",
                "Perbanyak frekuensi makan (5–6 kali/hari)."
            ],
            "Normal Weight": [
                "Pertahankan pola makan seimbang.",
                "Lakukan aktivitas fisik minimal 30 menit/hari.",
                "Monitor berat badan setiap 2 minggu."
            ],
            "Overweight": [
                "Kurangi makanan tinggi gula dan lemak.",
                "Tingkatkan aktivitas fisik seperti jalan cepat atau jogging.",
                "Perbanyak konsumsi sayur dan buah."
            ],
            "Obese": [
                "Kurangi konsumsi kalori 500–1000 kalori/hari.",
                "Lakukan olahraga teratur (150 menit per minggu).",
                "Konsultasikan rencana diet dengan ahli gizi.",
                "Kurangi minuman manis dan makanan olahan."
            ],
            "Extremely Obese": [
                "Pertimbangkan program penurunan berat badan yang diawasi tenaga medis.",
                "Pantau tekanan darah dan kadar gula secara rutin.",
                "Kombinasikan diet, olahraga, dan konsultasi profesional."
            ]
        }

        if pred in rekom:
            for r in rekom[pred]:
                st.write(f"- {r}")
        else:
            st.write("Tidak ada rekomendasi untuk kategori ini.")

        st.write("DEBUG INPUT:", input_data)


        # Rekomendasi final untuk dimasukkan ke DB
        rekom_text = "\n".join(rekom[pred])

        save_prediction(
            conn,
            age=input_data.get("age", None),
            gender=input_data.get("gender", None),
            height=input_data.get("height", None),
            weight=input_data.get("weight", None),
            bmi=input_data.get("bmi", None),
            probabilitas=float(max(probas)),
            prediksi=pred,
            rekomendasi=rekom_text
        )
        st.success("Hasil prediksi disimpan ke database.")
# ============================
# MENU 3: Feature Importance
# ============================
elif menu == "Feature Importance":
    st.title("Feature Importance (Random Forest)")

    importance = model.feature_importances_
    feature_names = feature_list  

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    # Grafik
    st.subheader("Grafik Feature Importance")
    st.bar_chart(df_imp.set_index("Feature"))

    # Tabel detail
    st.subheader("Detail Nilai Importance")
    st.dataframe(df_imp, use_container_width=True)

elif menu == "Riwayat Prediksi":
    st.title("Riwayat Prediksi Obesitas")

    rows = load_history(conn)

    df_history = pd.DataFrame(rows, columns=[
        "riwayat_id", "user_id", "age", "gender", "height", "weight", 
        "bmi", "probability", "prediction", "recommendation", "timestamp"
    ])

    df_display = df_history.drop(columns=["riwayat_id", "user_id"])
    df_display.insert(0, "No", range(1, len(df_display) + 1))

    df_display = df_display.rename(columns={
        "age": "Age",
        "gender": "Gender",
        "height": "Height",
        "weight": "Weight",
        "bmi": "BMI",
        "probability": "Probability",
        "prediction": "Prediction",
        "recommendation": "Recommendation",
        "timestamp": "Waktu Prediksi"
    })
    st.dataframe(df_display, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_display.to_excel(writer, index=False, sheet_name='Riwayat Prediksi')

    st.download_button(
        label="Download Excel",
        data=buffer.getvalue(),
        file_name="riwayat_prediksi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



