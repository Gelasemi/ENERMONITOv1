import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import xgboost as xgb
import plotly.express as px

# -----------------------
# 1. Chargement des données et du scaler
# -----------------------
try:
    df = pd.read_csv("data.csv", parse_dates=["Date"])
except Exception:
    st.error("❌ data.csv introuvable. Ajoutez un fichier historique.")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
except Exception:
    st.error("❌ scaler.pkl introuvable. Ajoutez votre scaler entraîné.")
    st.stop()

# -----------------------
# 2. UI Streamlit
# -----------------------
st.set_page_config(page_title="Energy Forecast La Réunion", layout="centered")
st.title("⚡ Prédiction de Consommation Énergétique - La Réunion")

# Choix du modèle
model_choice = st.sidebar.selectbox("📊 Choisir le modèle", ["LSTM", "XGBoost"])

# -----------------------
# 3. Chargement du modèle
# -----------------------
if model_choice == "LSTM":
    try:
        model = load_model("lstm_model.h5")
        look_back = 14 * 24  # 14 jours * 24 heures
    except Exception:
        st.error("❌ lstm_model.h5 introuvable. Ajoutez votre modèle LSTM entraîné.")
        st.stop()
else:
    try:
        model = xgb.XGBRegressor()
        model.load_model("xgboost_model.json")
    except Exception:
        st.error("❌ xgboost_model.json introuvable. Ajoutez votre modèle XGBoost.")
        st.stop()

# -----------------------
# 4. Entrées utilisateur
# -----------------------
st.subheader("⚙️ Paramètres utilisateur")

# Exemple de variables exogènes (ajoute les autres plus tard)
temp = st.slider("🌡 Température (°C)", -10, 40, 25)
event = st.selectbox("🎉 Événement spécial ?", [0, 1])  # 0=non, 1=oui

# -----------------------
# 5. Préparation des données
# -----------------------
features = ["Consommation (MW)", "Température (°C)", "Jour_semaine", "Mois", "Heure", "Weekend", "Vacances"]

if model_choice == "LSTM":
    scaled_data = scaler.transform(df[features])
    last_sequence = scaled_data[-look_back:].reshape(1, look_back, -1)

    prediction_scaled = model.predict(last_sequence)
    # Reprojection (inverse scaling sur la 1ère variable: consommation)
    dummy = np.zeros((1, scaled_data.shape[1]))
    dummy[0, 0] = prediction_scaled[0, 0]
    prediction = scaler.inverse_transform(dummy)[0, 0]
else:
    last_row = df.iloc[-1].copy()
    # On met à jour avec input utilisateur
    new_features = pd.DataFrame({
        "Consommation (MW)": [last_row["Consommation (MW)"]],
        "Température (°C)": [temp],
        "Jour_semaine": [last_row["Jour_semaine"]],
        "Mois": [last_row["Mois"]],
        "Heure": [last_row["Heure"]],
        "Weekend": [last_row["Weekend"]],
        "Vacances": [last_row["Vacances"]]
    })

    features_scaled = scaler.transform(new_features)
    prediction = model.predict(features_scaled)[0]

# -----------------------
# 6. Résultat
# -----------------------
st.success(f"🔮 Consommation prédite: **{prediction:.2f} MW**")

# -----------------------
# 7. Graphe historique + prévision
# -----------------------
fig = px.line(df.tail(100), x="Date", y="Consommation (MW)", title="Historique de Consommation")
fig.add_scatter(x=[df["Date"].iloc[-1] + pd.Timedelta(hours=1)],
                y=[prediction],
                mode="markers+text",
                name="Prévision",
                text=["Prévision"],
                textposition="top center")
st.plotly_chart(fig, use_container_width=True)
