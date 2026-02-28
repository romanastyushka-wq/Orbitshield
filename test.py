from predict import predict_batch
import pandas as pd

# 1. Создаем имитацию потока данных (последние 4 часа)
# Допустим, мы видим нарастание скорости ветра и поворот Bz на юг
history_data = [
    # T-3 hours (Спокойно)
    {"IMF_Magnitude": 5.0, "Bx": 2, "By": 1, "Bz": 2, "Proton_Density": 5, "Flow_Speed": 400, "Kp_x10": 20},
    # T-2 hours (Начало возмущения)
    {"IMF_Magnitude": 8.0, "Bx": 3, "By": -2, "Bz": -5, "Proton_Density": 10, "Flow_Speed": 450, "Kp_x10": 30},
    # T-1 hours (Усиление)
    {"IMF_Magnitude": 12.0, "Bx": 4, "By": -5, "Bz": -10, "Proton_Density": 15, "Flow_Speed": 550, "Kp_x10": 45},
    # T-0 (CURRENT MOMENT - Шторм!)
    {"IMF_Magnitude": 25.0, "Bx": 5, "By": -10, "Bz": -20, "Proton_Density": 30, "Flow_Speed": 800, "Kp_x10": 80},
]

print("--- Running SolarShield Simulation ---")

# Пробуем предсказать риск для последнего момента, учитывая историю
result = predict_batch(history_data)

print(f"\nPREDICTION RESULT:")
print(f"Status: {result['status']}")
print(f"Anomaly Score: {result['risk_score']}")
print("\nProcessed Features (Debug):")
# Показываем, какие фичи реально зашли в модель (обрати внимание на Bz_South и лаги)
for k, v in result['input_processed'].items():
    if abs(v) > 0.1: # Показываем только значимые
        print(f"  {k}: {v:.3f}")