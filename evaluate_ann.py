import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from model_ann import Net
import pandas as pd

# Charger le scaler
scaler = joblib.load("models/scaler.pkl")

# Charger les données test (même préparation que dans train_ann)
# [...]

# Charger modèle
model = Net()
model.load_state_dict(torch.load("models/ann_model_lp.pt"))
model.eval()

# Préparation des données test
X_scaled = scaler.transform(X)
X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Prédictions
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

y_pred = y_pred_tensor.numpy()

# RMSE et MSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
mse = mean_squared_error(y, y_pred)
print(f"📉 MSE: {mse:.4f}")
print(f"📈 RMSE: {rmse:.4f}")
