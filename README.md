# 🔮 Prédiction de solutions pour problèmes linéaires

Cette application Streamlit prédit les variables de décision `(x1, x2)` d’un problème de programmation linéaire, à partir des coefficients et contraintes fournis (problèmes avec 2 solutions et 3 contraintes).

## 🧠 Modèle

Le modèle est un réseau de neurones artificiels (ANN) entraîné avec PyTorch sur des données générées aléatoirement.

## 🖥️ Interface

Développée avec [Streamlit](https://streamlit.io/), elle permet de saisir :
- Les coefficients de la fonction objectif `c1`, `c2`
- Les coefficients de la matrice de contraintes `a1` à `a6`
- Les bornes des contraintes `b1` à `b3`

Et affiche :
- La solution `(x1, x2)`
- La valeur de la fonction objectif `z = c1·x1 + c2·x2`

## 📁 Structure

```
.
├── app.py                # Interface Streamlit
├── train_ann.py          # Entraînement de modèle
├── model_ann.py          # Modèle ANN
├── models/
│   ├── ann_model.pth     # Modèle entraîné
│   └── scaler.pkl        # Scaler sauvegardé
├── data/
│   └── LP_modele.csv     # Données d'entraînement 
├── requirements.txt      # Dépendances Python
```
