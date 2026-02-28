import numpy as np
from joblib import load
a = int()
b = int()
a==0
b==0
model = load("solarshield_model.pkl")

# пример строки из твоих данных (НОРМАЛЬНАЯ)
x_normal = np.array([[14, 15, 99990.7, -99990.2, 999910.0, 999295.0, 99993]])

a = model.predict(x_normal)
b = model.predict(x_normal)

print("Normal case:")
print("Risk probability:", a)
print("Anomaly score:", b)
