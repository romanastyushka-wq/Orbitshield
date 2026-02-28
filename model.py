import numpy as np
import joblib

class PhysicsAwareModel:
    def __init__(self, input_dim):
        self.mean = np.zeros(input_dim)
        self.std = np.ones(input_dim)
        self.weights = np.ones(input_dim)
        self.threshold = 3.0 # Сигма для алерта

    def fit(self, X_scaled):
        # X_scaled - это pandas DataFrame
        self.mean = X_scaled.mean(axis=0).values
        self.std = X_scaled.std(axis=0).values + 1e-6
        
        # --- Расчет весов важности ---
        # Мы хотим, чтобы модель паниковала от Bz_South и Dynamic_Pressure
        feature_names = X_scaled.columns.tolist()
        weights = np.ones(len(feature_names))
        
        for i, name in enumerate(feature_names):
            if "Bz_South" in name:
                weights[i] = 5.0  # Критический вес
            elif "Kp" in name:
                weights[i] = 3.0
            elif "Pressure" in name:
                weights[i] = 2.0
            elif "lag" in name:
                weights[i] = 1.5
                
        self.weights = weights
        print(f"✅ Model fitted. Critical weights assigned to: {[f for f, w in zip(feature_names, weights) if w > 1]}")

    def get_anomaly_score(self, x_vector):
        # Махаланобис-подобная дистанция с весами
        z_score = (x_vector - self.mean) / self.std
        
        # Квадратичная ошибка взвешенная
        weighted_dist = np.sqrt(np.sum(self.weights * (z_score ** 2)))
        return weighted_dist