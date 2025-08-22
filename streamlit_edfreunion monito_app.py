import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="EnerMonito v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #1f77b4;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .model-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("<h1 class='main-header'>⚡ EnerMonito v2</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Prédiction de consommation énergétique à La Réunion</p>", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/data.csv', parse_dates=['date'])
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Veuillez ajouter le fichier data.csv dans le dossier data/")
        return None

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        # Charger le modèle LSTM
        lstm_model = load_model('models/lstm_model.h5')
        # Charger le scaler pour LSTM
        scaler_lstm = joblib.load('models/scaler_lstm.pkl')
        
        # Charger le modèle XGBoost
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model('models/xgboost_model.json')
        
        # Charger le scaler général
        scaler = joblib.load('models/scaler.pkl')
        
        return lstm_model, scaler_lstm, xgb_model, scaler
    except FileNotFoundError as e:
        st.error(f"Fichier modèle non trouvé: {e}")
        return None, None, None, None

# Charger les données et modèles
df = load_data()
lstm_model, scaler_lstm, xgb_model, scaler = load_models()

if df is not None and lstm_model is not None:
    # Sidebar
    st.sidebar.header("Paramètres de prédiction")
    
    # Sélection du modèle
    model_choice = st.sidebar.selectbox(
        "Choisir le modèle",
        ["LSTM (R²≈0.96)", "XGBoost (R²≈0.94)"],
        index=0
    )
    
    # Date de prédiction
    prediction_date = st.sidebar.date_input(
        "Date de prédiction",
        min_value=df['date'].min() + timedelta(days=15),
        max_value=df['date'].max() + timedelta(days=30),
        value=df['date'].max() + timedelta(days=1)
    )
    
    # Paramètres météo
    st.sidebar.subheader("Paramètres météo")
    temperature = st.sidebar.slider(
        "Température (°C)",
        min_value=15.0,
        max_value=35.0,
        value=25.0,
        step=0.1
    )
    
    humidity = st.sidebar.slider(
        "Humidité (%)",
        min_value=30,
        max_value=90,
        value=65
    )
    
    # Paramètres de consommation
    st.sidebar.subheader("Paramètres de consommation")
    population = st.sidebar.number_input(
        "Population",
        min_value=700000,
        max_value=1000000,
        value=int(df['Population'].max()),
        step=1000
    )
    
    # Affichage des métriques clés
    st.subheader("Métriques clés")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Consommation moyenne", f"{df['Consommation Totale'].mean():,.0f} kWh")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Population", f"{df['Population'].max():,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Énergies renouvelables", f"{df['renewable_ratio'].mean()*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Dernière consommation", f"{df['Consommation Totale'].iloc[-1]:,.0f} kWh")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Section de prédiction
    st.subheader("Prédiction de consommation")
    
    # Bouton de prédiction
    predict_button = st.button("Lancer la prédiction", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calcul en cours..."):
            if model_choice == "LSTM (R²≈0.96)":
                # Prédiction avec LSTM
                window = 14  # Fenêtre de 14 jours
                
                # Récupérer les 14 derniers jours avant la date de prédiction
                start_date = prediction_date - timedelta(days=window)
                mask = (df['date'] >= start_date) & (df['date'] < prediction_date)
                sequence_data = df.loc[mask, ['Consommation Totale', 'temperature', 'Population', 'is_weekend', 'Résidentiel', 'Tertiaire']].values
                
                # Vérifier si nous avons assez de données
                if len(sequence_data) < window:
                    st.error(f"Pas assez de données historiques pour la date sélectionnée. Disponible: {len(sequence_data)} jours, Requis: {window} jours")
                else:
                    # Normaliser avec le scaler LSTM
                    sequence_scaled = scaler_lstm.transform(sequence_data)
                    
                    # Reshape pour le LSTM (1 séquence de 14 jours avec 6 features)
                    sequence_reshaped = sequence_scaled.reshape(1, window, 6)
                    
                    # Faire la prédiction
                    prediction_scaled = lstm_model.predict(sequence_reshaped)
                    
                    # Dénormaliser la prédiction
                    dummy = np.zeros((1, 6))
                    dummy[0, 0] = prediction_scaled[0, 0]
                    prediction = scaler_lstm.inverse_transform(dummy)[0, 0]
                    
                    # Afficher le résultat
                    st.success(f"Consommation prédite pour le {prediction_date.strftime('%d/%m/%Y')}: **{prediction:,.2f} kWh**")
                    
                    # Visualisation
                    fig = go.Figure()
                    
                    # Ajouter les données historiques (30 derniers jours)
                    hist_data = df[df['date'] >= prediction_date - timedelta(days=30)]
                    fig.add_trace(go.Scatter(
                        x=hist_data['date'],
                        y=hist_data['Consommation Totale'],
                        mode='lines+markers',
                        name='Historique',
                        line=dict(color='blue')
                    ))
                    
                    # Ajouter la prédiction
                    fig.add_trace(go.Scatter(
                        x=[prediction_date],
                        y=[prediction],
                        mode='markers',
                        name='Prédiction LSTM',
                        marker=dict(color='red', size=12)
                    ))
                    
                    fig.update_layout(
                        title="Prédiction de consommation (LSTM)",
                        xaxis_title="Date",
                        yaxis_title="Consommation (kWh)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # XGBoost
                # Préparer les features pour XGBoost
                features = pd.DataFrame({
                    'Année': [prediction_date.year],
                    'Energie Fossile': [df['Energie Fossile'].mean()],
                    'Photo Voltaique': [df['Photo Voltaique'].mean()],
                    'Eolien': [df['Eolien'].mean()],
                    'Production MGWH': [df['Production MGWH'].mean()],
                    'RES': [df['RES'].mean()],
                    'ENT_PRO': [df['ENT_PRO'].mean()],
                    'ENT': [df['ENT'].mean()],
                    'Population': [population],
                    'year': [prediction_date.year],
                    'lag_1': [df['Consommation Totale'].iloc[-1]],
                    'lag_7': [df['Consommation Totale'].iloc[-7] if len(df) >= 7 else df['Consommation Totale'].mean()],
                    'lag_30': [df['Consommation Totale'].iloc[-30] if len(df) >= 30 else df['Consommation Totale'].mean()],
                    'rolling_mean_7': [df['Consommation Totale'].rolling(window=7).mean().iloc[-1]],
                    'rolling_mean_30': [df['Consommation Totale'].rolling(window=30).mean().iloc[-1]],
                    'ema_7': [df['Consommation Totale'].ewm(span=7).mean().iloc[-1]],
                    'event_impact': [0],  # Par défaut, pas d'événement spécial
                    'res_ratio': [df['RES'].mean() / (df['Consommation Totale'].mean() + 1e-3)],
                    'ent_pro_ratio': [df['ENT_PRO'].mean() / (df['Consommation Totale'].mean() + 1e-3)],
                    'ent_ratio': [df['ENT'].mean() / (df['Consommation Totale'].mean() + 1e-3)]
                })
                
                # Normaliser les features
                features_scaled = scaler.transform(features)
                
                # Faire la prédiction
                prediction = xgb_model.predict(features_scaled)[0]
                
                # Afficher le résultat
                st.success(f"Consommation prédite pour le {prediction_date.strftime('%d/%m/%Y')}: **{prediction:,.2f} kWh**")
                
                # Visualisation
                fig = go.Figure()
                
                # Ajouter les données historiques (30 derniers jours)
                hist_data = df[df['date'] >= prediction_date - timedelta(days=30)]
                fig.add_trace(go.Scatter(
                    x=hist_data['date'],
                    y=hist_data['Consommation Totale'],
                    mode='lines+markers',
                    name='Historique',
                    line=dict(color='blue')
                ))
                
                # Ajouter la prédiction
                fig.add_trace(go.Scatter(
                    x=[prediction_date],
                    y=[prediction],
                    mode='markers',
                    name='Prédiction XGBoost',
                    marker=dict(color='green', size=12)
                ))
                
                fig.update_layout(
                    title="Prédiction de consommation (XGBoost)",
                    xaxis_title="Date",
                    yaxis_title="Consommation (kWh)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Section d'analyse
    st.subheader("Analyse des données")
    
    # Sélection de la période
    col1, col2 = st.columns(2)
    with col1:
        start_period = st.date_input("Date de début", value=df['date'].min())
    with col2:
        end_period = st.date_input("Date de fin", value=df['date'].max())
    
    # Filtrer les données
    mask = (df['date'] >= start_period) & (df['date'] <= end_period)
    filtered_data = df.loc[mask]
    
    # Graphique de consommation
    fig_consumption = px.line(
        filtered_data,
        x='date',
        y='Consommation Totale',
        title="Consommation énergétique historique",
        labels={'Consommation Totale': 'Consommation (kWh)', 'date': 'Date'}
    )
    st.plotly_chart(fig_consumption, use_container_width=True)
    
    # Graphique de corrélation
    st.subheader("Corrélations")
    
    # Sélectionner les colonnes numériques pertinentes
    corr_cols = ['Consommation Totale', 'temperature', 'Population', 'is_weekend', 'Résidentiel', 'Tertiaire']
    corr_data = filtered_data[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_data,
        text_auto=True,
        aspect="auto",
        title="Matrice de corrélation",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Développé avec ❤️ pour EnerMonito v2 | Données mises à jour en temps réel")
else:
    st.error("Impossible de charger les données ou les modèles. Vérifiez que tous les fichiers sont présents.")
