import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

# Константы физики
BAD_VALUES = [9999, 99999, -9999, -99999]
RAW_FEATURES = ["IMF_Magnitude", "Bx", "By", "Bz", "Proton_Density", "Flow_Speed", "Kp_x10"]

class SolarPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, history_window=3):
        self.scaler = RobustScaler()
        self.medians = None
        self.history_window = history_window  # Сколько часов назад смотрим
        self.feature_names = []

    def fit(self, X, y=None):
        # 1. Очистка и расчет медиан (только на TRAIN)
        X_clean = X[RAW_FEATURES].replace(BAD_VALUES, np.nan)
        self.medians = X_clean.median()
        
        # Заполняем для фиттинга скейлера
        X_filled = X_clean.fillna(self.medians)
        
        # 2. Генерируем инженерные фичи для скейлера
        X_eng = self._engineer_features(X_filled)
        
        self.scaler.fit(X_eng)
        self.feature_names = X_eng.columns.tolist()
        return self

    def transform(self, X):
        # 1. Очистка
        X_clean = X[RAW_FEATURES].replace(BAD_VALUES, np.nan)
        X_filled = X_clean.fillna(self.medians) # Используем медианы из TRAIN!
        
        # 2. Инжиниринг
        X_eng = self._engineer_features(X_filled)
        
        # 3. Скейлинг
        X_scaled = self.scaler.transform(X_eng)
        
        # Возвращаем DataFrame для удобства отладки
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    def _engineer_features(self, df):
        df_eng = df.copy()
        
        # --- Физика ---
        # Асимметрия Bz: Южное направление (отрицательное) открывает магнитосферу
        df_eng["Bz_South"] = df_eng["Bz"].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Энергия солнечного ветра (простой прокси: плотность * скорость^2)
        df_eng["Dynamic_Pressure"] = df_eng["Proton_Density"] * (df_eng["Flow_Speed"]**2)

        # --- Динамика (Lags & Deltas) ---
        # В продакшене мы ожидаем, что df имеет историю. 
        # Если это одиночный запрос, эти поля будут NaN (нужна буферизация в проде)
        for lag in [1, 3]:
            for col in ["Bz", "Flow_Speed", "Kp_x10"]:
                col_name = f"{col}_lag{lag}"
                df_eng[col_name] = df_eng[col].shift(lag)
                
                # Заполняем пропуски от сдвига текущими значениями (bfill/ffill), 
                # чтобы не терять первые строки в тесте
                df_eng[col_name] = df_eng[col_name].fillna(method='bfill').fillna(method='ffill')

        # Градиенты (резкие изменения опаснее плавных)
        df_eng["Bz_Grad"] = df_eng["Bz"].diff().fillna(0).abs()
        
        return df_eng