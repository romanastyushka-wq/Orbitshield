import time
import requests
import numpy as np
import os
from datetime import datetime
from collections import deque
from predict import predict_batch
from predict_vision import SolarVision
from db_handler import DatabaseHandler

# === КОНФИГУРАЦИЯ ===
NOAA_URLS = {
    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json",
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
}
SDO_URL = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg"
BAD_VALUES = [-999.9, 999.9, -9999, 9999, -1.00e+31]

class SolarMonitor:
    def __init__(self):
        self.history_buffer = deque(maxlen=72)
        self.last_valid_sample = None
        self.db = DatabaseHandler()
        self.vision = SolarVision()
        print("🛰️ Multi-Modal Monitor Started.")

    def fetch_json(self, url):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except: return None

    def _clean_value(self, value, key):
        try:
            val = float(value)
            if any(abs(val - bad) < 0.1 for bad in BAD_VALUES): raise ValueError
            return val
        except:
            if self.last_valid_sample: return self.last_valid_sample.get(key)
            return None

    def get_latest_data(self):
        mag = self.fetch_json(NOAA_URLS["mag"])
        plasma = self.fetch_json(NOAA_URLS["plasma"])
        kp = self.fetch_json(NOAA_URLS["kp"])
        
        if not (mag and plasma and kp): return None
        
        try:
            raw = {
                "IMF_Magnitude": mag[-1][6], 
                "Bx": mag[-1][1], 
                "By": mag[-1][2], 
                "Bz": mag[-1][3],
                "Proton_Density": plasma[-1][1], 
                "Flow_Speed": plasma[-1][2], 
                "Kp_x10": float(kp[-1][1]) * 10
            }
            cleaned = {k: self._clean_value(v, k) for k, v in raw.items()}
            if None in cleaned.values(): return None
            cleaned["datetime"] = mag[-1][0]
            self.last_valid_sample = cleaned
            return cleaned
        except: return None

    def run(self, interval=60):
        print(f"🔄 Polling NOAA & SDO every {interval}s...")
        while True:
            data = self.get_latest_data()
            if data:
                self.history_buffer.append(data)
                if len(self.history_buffer) >= 2:
                    try:
                        pred = predict_batch(list(self.history_buffer))
                        vis_risk, img_obj = self.vision.analyze_url(SDO_URL)
                        
                        if img_obj: 
                            img_obj.save("latest_sun.jpg")
                        
                        self.db.save_measurement(data, pred, visual_risk=vis_risk, image_url="latest_sun.jpg")
                        
                        # --- ОБНОВЛЕННЫЙ ВЫВОД ПАРАМЕТРОВ ---
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        speed = data['Flow_Speed']
                        density = data['Proton_Density']
                        bz = data['Bz']
                        kp = data['Kp_x10'] / 10
                        risk = pred['risk_score']

                        print(f"[{timestamp}] ✅ Data Saved")
                        print(f"   📊 Wind: {speed:.1f} km/s | Density: {density:.1f} p/cm³ | Bz: {bz:.2f} nT | Kp: {kp:.1f}")
                        print(f"   🧠 AI Risk: {risk:.2f}% | Vision Risk: {vis_risk:.2f}%")
                        print("-" * 60)
                        
                    except Exception as e: 
                        print(f"🔥 Error: {e}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💤 Waiting for data...")
            
            time.sleep(interval)

if __name__ == "__main__":
    SolarMonitor().run()