import sqlite3
import pandas as pd
import os

DB_NAME = "solar_shield.db"

class DatabaseHandler:
    def __init__(self, db_path=DB_NAME):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                risk_score REAL,
                status TEXT,
                bz REAL,
                speed REAL,
                kp REAL,
                bx REAL,
                by REAL,
                imf REAL,
                density REAL,
                visual_risk REAL DEFAULT 0,
                image_url TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON measurements(timestamp)')
        conn.commit()
        conn.close()

    def save_measurement(self, data, prediction, visual_risk=0, image_url=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO measurements (timestamp, risk_score, status, bz, speed, kp, bx, by, imf, density, visual_risk, image_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['datetime'],
                prediction['risk_score'],
                prediction['status'],
                data.get('Bz', 0),
                data.get('Flow_Speed', 0),
                data.get('Kp_x10', 0)/10,
                data.get('Bx', 0),
                data.get('By', 0),
                data.get('IMF_Magnitude', 0),
                data.get('Proton_Density', 0),
                visual_risk,
                image_url
            ))
            conn.commit()
        except sqlite3.Error as e:
            print(f"❌ DB Error: {e}")
        finally:
            conn.close()

    def get_recent_history(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM measurements ORDER BY id DESC LIMIT ?"
        try:
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df.iloc[::-1].reset_index(drop=True)
        finally:
            conn.close()

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT count(*), max(risk_score), max(visual_risk) FROM measurements")
            res = cursor.fetchone()
            # Защита от None, если база пуста
            return {
                "total_records": res[0] if res[0] else 0, 
                "max_risk_ever": res[1] if res[1] else 0.0,
                "max_visual_ever": res[2] if res[2] else 0.0
            }
        finally:
            conn.close()