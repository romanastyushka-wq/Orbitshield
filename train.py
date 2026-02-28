from data_loader import load_txt_data
from preprocess import SolarPreprocessor
from model import PhysicsAwareModel
import joblib
import numpy as np
import pandas as pd

def main():
    print("🚀 Loading data...")
    df = load_txt_data("data/solar_data.txt")
    
    # --- Strict Time Split ---
    # Нельзя перемешивать (shuffle) временные ряды!
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Training on range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")

    # --- Preprocessing ---
    print("🛠 Preprocessing...")
    preprocessor = SolarPreprocessor()
    
    # Fit только на Train!
    preprocessor.fit(train_df)
    X_train = preprocessor.transform(train_df)
    
    # --- Model Training ---
    print("🧠 Training Physics Model...")
    model = PhysicsAwareModel(input_dim=X_train.shape[1])
    model.fit(X_train)

    # --- Validation Loop (Simple) ---
    # Проверка на тесте
    X_test = preprocessor.transform(test_df)
    scores = [model.get_anomaly_score(row) for row in X_test.values]
    
    # Статистика
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    print(f"Test Stats -> Avg Risk: {mean_score:.2f}, Max Risk: {max_score:.2f}")

    # --- Saving ---
    joblib.dump(model, "solarshield_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("✅ System saved.")

if __name__ == "__main__":
    main()