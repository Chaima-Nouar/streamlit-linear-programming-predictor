import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Créer le dossier s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Charger les données
df = pd.read_csv("data/LP_modele.csv")

# Extraction des colonnes nécessaires
df[['c1', 'c2']] = df['coefficients_objectif'].str.strip('[]').str.split(',', expand=True).astype(float)
df[['a1', 'a2', 'a3', 'a4', 'a5', 'a6']] = df['matrice_contraintes'].str.strip('[]').str.split(',', expand=True).astype(float)
df[['b1', 'b2', 'b3']] = df['borne_contraintes'].str.strip('[]').str.split(',', expand=True).astype(float)

X = df[['c1', 'c2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'b1', 'b2', 'b3']].values

# Recréer le scaler et l'entraîner
scaler = StandardScaler()
scaler.fit(X)

# Sauvegarder le scaler
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Scaler sauvegardé avec succès dans models/scaler.pkl")
