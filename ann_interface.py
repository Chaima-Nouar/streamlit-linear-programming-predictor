import streamlit as st
import torch
import numpy as np
import joblib
from model_ann import Net

# Charger le modèle et le scaler
model = Net()
model.load_state_dict(torch.load("models/ann_model_lp.pt"))
model.eval()
scaler = joblib.load("models/scaler.pkl")

st.title("🔮 Prédiction de la solution (x1, x2) et valeur de z")

st.markdown("### 🎯 Entrez les valeurs pour les 11 variables d'entrée :")

# Objectif
c1 = st.number_input("c1 (objectif 1)", min_value=1, max_value=100, value=50)
c2 = st.number_input("c2 (objectif 2)", min_value=1, max_value=100, value=50)

# Contraintes A
a1 = st.number_input("a1", min_value=1, max_value=100, value=50)
a2 = st.number_input("a2", min_value=1, max_value=100, value=50)
a3 = st.number_input("a3", min_value=1, max_value=100, value=50)
a4 = st.number_input("a4", min_value=1, max_value=100, value=50)
a5 = st.number_input("a5", min_value=1, max_value=100, value=50)
a6 = st.number_input("a6", min_value=1, max_value=100, value=50)

# Bornes B
b1 = st.number_input("b1", min_value=1, max_value=150, value=75)
b2 = st.number_input("b2", min_value=1, max_value=150, value=75)
b3 = st.number_input("b3", min_value=1, max_value=150, value=75)

# Prédiction
if st.button("🔍 Prédire"):
    user_input = np.array([[c1, c2, a1, a2, a3, a4, a5, a6, b1, b2, b3]])
    user_input_scaled = scaler.transform(user_input)
    input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).numpy()[0]

    # Arrondi à une décimale
    x1_rounded = f"{prediction[0]:.1f}"
    x2_rounded = f"{prediction[1]:.1f}"

    # Calcul de la fonction objectif
    z = float(x1_rounded) * c1 + float(x2_rounded) * c2
    z = f"{z:.1f}"

    # Affichage
    st.success(f"✅ Prédiction : x1 = {x1_rounded}, x2 = {x2_rounded}")
    st.info(f"📈 Valeur de la fonction objectif : z = {z}")
