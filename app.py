import streamlit as st
import pandas as pd
import joblib

# Configuration visuelle
st.set_page_config(page_title="IA Salaire Predictor", layout="centered")

# Chargement du pack (Modèle + Scaler + Encoders)
@st.cache_resource
def load_all():
    # Le fichier doit être dans le même dossier sur GitHub
    return joblib.load('bundle_final.pkl')

data = load_all()
model = data['model']
scaler = data['scaler']
encoders = data['encoders']
features = data['feature_names']

st.title("💰 Prédiction de Salaire par Intelligence Artificielle")
st.write("Ce modèle utilise un Réseau de Neurones Artificiels entraîné sur 250 000 données.")

# Création du formulaire interactif
with st.form("prediction_form"):
    st.subheader("Entrez les détails du profil")
    inputs = {}
    
    # On génère les menus dynamiquement
    for col in features:
        if col in encoders:
            # Récupère toutes les villes, métiers, etc. du CSV
            options = sorted(encoders[col].classes_.tolist())
            inputs[col] = st.selectbox(f"{col}", options)
        else:
            # Pour l'expérience ou les certifications
            inputs[col] = st.number_input(f"{col}", min_value=0, max_value=50, value=5)
            
    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    # Transformation des entrées en DataFrame
    df_input = pd.DataFrame([inputs])
    
    # Encodage automatique (Traduction texte -> nombre)
    for col in encoders:
        df_input[col] = encoders[col].transform(df_input[col])
    
    # Normalisation et Prédiction
    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)
    
    # Résultat final
    st.success(f"### Salaire annuel estimé : {prediction[0]:,.2f} €")
    st.info("Cette estimation est basée sur les tendances du marché analysées par le Data Mining.")