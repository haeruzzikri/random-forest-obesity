import sqlite3
import json

DB_NAME = "obesity.db"

def create_connection():
    """Membuat koneksi database SQLite."""
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def create_table(conn):
    """Membuat tabel jika belum ada dan memperbaiki kolom jika tabel lama belum lengkap."""
    cursor = conn.cursor()

    # Buat tabel dasar (jika belum ada)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_prediksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_json TEXT,
            probabilitas REAL,
            prediksi TEXT,
            rekomendasi TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Periksa kolom yang sudah ada
    cursor.execute("PRAGMA table_info(riwayat_prediksi)")
    existing_cols = [row[1] for row in cursor.fetchall()]

    # Kolom yang wajib ada
    required_columns = {
        "input_json": "TEXT",
        "probabilitas": "REAL",
        "prediksi": "TEXT",
        "rekomendasi": "TEXT",
        "timestamp": "DATETIME"
    }

    # Tambahkan kolom yang belum ada
    for col, col_type in required_columns.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE riwayat_prediksi ADD COLUMN {col} {col_type}")

    conn.commit()


def save_prediction(conn, input_dict, probabilitas, prediksi, rekomendasi):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO riwayat_prediksi (input_json, probabilitas, prediksi, rekomendasi)
        VALUES (?, ?, ?, ?)
    """, (
        json.dumps(input_dict, default=str),
        float(probabilitas),
        str(prediksi),
        str(rekomendasi)
    ))
    conn.commit()


def load_history(conn):
    """Memuat seluruh riwayat prediksi."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, input_json, probabilitas, prediksi, rekomendasi, timestamp
        FROM riwayat_prediksi
        ORDER BY id DESC
    """)
    return cursor.fetchall()

def delete_history_by_id(conn, row_id):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM riwayat_prediksi WHERE id = ?", (row_id,))
    conn.commit()

def delete_all_history(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM riwayat_prediksi")
    conn.commit()

