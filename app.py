import streamlit as st
import pandas as pd
import joblib  # Utilisation standard de joblib

# Configuration visuelle
st.set_page_config(page_title="IA Salaire Predictor", layout="centered")

# Chargement du pack (Modèle + Scaler + Encoders)
@st.cache_resource
def load_all():
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
    
    for col in features:
        if col in encoders:
            options = sorted(encoders[col].classes_.tolist())
            inputs[col] = st.selectbox(f"{col}", options)
        else:
            # Saisie numérique pour l'expérience, compétences et certifications
            inputs[col] = st.number_input(f"{col}", min_value=0, max_value=100, value=5)
            
    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    # --- LA LOI DE 30 (SÉCURITÉ ET FIABILITÉ) ---
    LIMITE = 30
    colonnes_a_limiter = ['experience_years', 'skills_count', 'certifications']
    
    # On vérifie si une alerte est nécessaire
    if any(inputs.get(c, 0) > LIMITE for c in colonnes_a_limiter):
        st.warning(f"⚠️ **Note de fiabilité :** Pour garantir la précision du modèle, les valeurs supérieures à {LIMITE} sont plafonnées (Loi de saturation).")
    
    # On applique le plafonnement technique sur les données d'entrée
    for c in colonnes_a_limiter:
        if c in inputs:
            inputs[c] = min(inputs[c], LIMITE)
    # ---------------------------------------------

    # Transformation des entrées en DataFrame
    df_input = pd.DataFrame([inputs])
    
    # S'assurer que les colonnes sont dans le bon ordre (celui de features)
    df_input = df_input[features]
    
    # Encodage automatique (Texte -> Nombre)
    for col in encoders:
        df_input[col] = encoders[col].transform(df_input[col])
    
    # Normalisation et Prédiction
    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)
    
    # Résultat final
    st.success(f"### Salaire annuel estimé : {prediction[0]:,.2f} €")
    st.info("Cette estimation est basée sur les tendances du marché analysées par le Data Mining.")
