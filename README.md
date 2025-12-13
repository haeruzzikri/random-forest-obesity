# 🧠 Obesity Risk Prediction System  
**Machine Learning–Based Obesity Classification with Streamlit**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Deskripsi Proyek
Proyek ini adalah **sistem prediksi tingkat obesitas** berbasis *machine learning* menggunakan **Random Forest Classifier**.  
Aplikasi dibangun dengan **Streamlit** sehingga dapat digunakan secara interaktif oleh pengguna non-teknis.

Sistem ini:
- Memprediksi **kategori obesitas**
- Menampilkan **metrik evaluasi lengkap**
- Menyediakan **riwayat prediksi**
- Ramah pengguna dengan **penjelasan setiap fitur input**

---

## 🎯 Tujuan
- Membantu pengguna memahami **risiko obesitas**
- Menyediakan contoh **end-to-end ML pipeline**
- Menjadi referensi implementasi **ML + Streamlit + SQLite**

---

## 📊 Dataset
Dataset: **Obesity Dataset**  
Fitur mencakup:
- Data demografis
- Kebiasaan makan
- Aktivitas fisik
- Penggunaan teknologi
- Riwayat keluarga

---

## ⚙️ Metodologi
1. **Preprocessing**
   - Penanganan missing value (median & mode)
   - Encoding variabel kategorikal
   - Feature engineering (BMI) -> Ini bisa ditambahkan bisa tidak
   - SMOTE untuk data imbalance

2. **Model**
   - Random Forest Classifier
   - Cross Validation (5-fold)

3. **Evaluasi**
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1-score
   - ROC–AUC (multiclass OVR)
   - Confusion Matrix

---

## 🧪 Hasil Evaluasi Model
| Metric | Score |
|------|------|
| Accuracy | **0.9905** |
| Precision | **0.9906** |
| Recall | **0.9905** |
| F1-Score | **0.9905** |
| AUC | **0.9999** |

Model menunjukkan performa **sangat baik dan stabil**.

---

## 🖥️ Fitur Aplikasi
### 🔹 1. Prediksi Data Satuan
- Input manual dengan **penjelasan setiap fitur**
- Validasi input (tidak boleh kosong)
- Output probabilitas tiap kelas
- Rekomendasi berbasis hasil prediksi

### 🔹 2. Evaluasi Model
- Ringkasan metrik utama
- Classification Report
- Confusion Matrix visual
- ROC Curve multiclass

### 🔹 3. Riwayat Prediksi
- Penyimpanan otomatis ke SQLite
- Download riwayat ke Excel
- Hapus data (per ID / semua)

---

## 🧾 Penjelasan Singkatan Fitur
| Kode | Deskripsi |
|----|----|
| FCVC | Frekuensi konsumsi sayur |
| NCP | Jumlah makan utama per hari |
| CH2O | Konsumsi air harian |
| FAF | Frekuensi aktivitas fisik |
| TUE | Waktu penggunaan teknologi |
| CAEC | Konsumsi makanan di antara waktu makan |
| CALC | Konsumsi alkohol |
| MTRANS | Moda transportasi |

---

## 🚀 Cara Menjalankan Aplikasi
### 1️⃣ Clone Repository
```bash
git clone https://github.com/username/obesity-prediction-system.git
cd obesity-prediction-system
pip install -r requirements.txt
python train_model.py
streamlit run app.py

📦 obesity-prediction-system
 ┣ 📜 app.py
 ┣ 📜 train_model.py
 ┣ 📜 database.py
 ┣ 📜 evaluation.pkl
 ┣ 📜 model_rf.pkl
 ┣ 📜 encoders.pkl
 ┣ 📜 feature_list.pkl
 ┣ 📜 obesity.db
 ┣ 📜 requirements.txt
 ┗ 📜 README.md

🛠️ Teknologi yang Digunakan
Python
Scikit-learn
Imbalanced-learn (SMOTE)
Streamlit
SQLite
Pandas, NumPy
Matplotlib & Seaborn

📌 Catatan
Proyek ini ditujukan untuk edukasi dan penelitian, bukan sebagai alat diagnosis medis.

👨‍💻 Author
Haeruzzikri
📍 Indonesia
📧 Feel free to connect and contribute!



