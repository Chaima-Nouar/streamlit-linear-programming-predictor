import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model_ann import Net

# Charger les données
df = pd.read_csv("data/LP_modele.csv")

# Prétraitement
df[['c1', 'c2']] = df['coefficients_objectif'].str.strip('[]').str.split(',', expand=True).astype(float)
df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']] = df['matrice_contraintes'].str.strip('[]').str.split(',', expand=True).astype(float)
df[['b1', 'b2', 'b3']] = df['borne_contraintes'].str.strip('[]').str.split(',', expand=True).astype(float)
df[['x1', 'x2']] = df['solution'].str.strip('[]').str.split(',', expand=True).astype(float)

X = df[['c1', 'c2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'b1', 'b2', 'b3']].values
y = df[['x1', 'x2']].values

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Modèle
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 500

# Entraînement
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Sauvegarde modèle + scaler
torch.save(model.state_dict(), "models/ann_model.pth")
import joblib
joblib.dump(scaler, "models/scaler.pkl")

# Plot
plt.plot(losses)
plt.xlabel("Épochs")
plt.ylabel("Loss (MSE)")
plt.title("Courbe de convergence du modèle RNA")
plt.grid()
plt.show()
