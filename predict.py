import numpy as np
import pandas as pd
import joblib

# Загрузка артефактов
try:
    model = joblib.load("solarshield_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
except:
    print("⚠️ Models not found. Run train.py first.")

def interpret_risk(score):
    if score < 2.0:
        return "GREEN (Normal)", score
    elif score < 5.0:
        return "YELLOW (Unsettled)", score
    elif score < 10.0:
        return "ORANGE (Warning)", score
    else:
        return "RED (CRITICAL STORM)", score

def predict_batch(data_list):
    """
    Принимает список словарей (историю наблюдений).
    Последний элемент списка - это 'текущий момент', который мы предсказываем.
    """
    # Превращаем в DataFrame, чтобы сработали лаги и shift
    df = pd.DataFrame(data_list)
    
    # Трансформация (создаст лаги, дельты и т.д.)
    # Важно: preprocessor использует медианы из обучения, утечки нет
    X_scaled = preprocessor.transform(df)
    
    # Берем ПОСЛЕДНЮЮ точку (текущий момент)
    current_vector = X_scaled.iloc[-1].values
    
    score = model.get_anomaly_score(current_vector)
    status, final_score = interpret_risk(score)
    
    return {
        "status": status,
        "risk_score": round(final_score, 2),
        "input_processed": X_scaled.iloc[-1].to_dict() # для дебага
    }