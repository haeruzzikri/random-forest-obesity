import sqlite3

# ============================
# Create Connection
# ============================
def create_connection():
    conn = sqlite3.connect("obesity.db", check_same_thread=False)
    return conn

# ============================
# Create Table
# ============================
def create_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_prediksi (
            riwayat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER,
            gender INTEGER,
            height INTEGER,
            weight INTEGER,
            bmi FLOAT,
            probabilitas REAL,
            prediksi TEXT,
            rekomendasi TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

# ============================
# Save Prediction
# ============================
def save_prediction(conn, age, gender, height, weight, bmi, probabilitas, prediksi, rekomendasi):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO riwayat_prediksi 
        (user_id, age, gender, height, weight, bmi, probabilitas, prediksi, rekomendasi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (1, age, gender, height, weight, bmi, probabilitas, prediksi, rekomendasi))
    conn.commit()

# ============================
# Load History
# ============================
def load_history(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM riwayat_prediksi ORDER BY riwayat_id DESC")
    rows = cursor.fetchall()
    return rows
