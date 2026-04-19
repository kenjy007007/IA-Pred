import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT DES OUTILS ---
# Charge le bundle (modèle, scaler, encoders)
bundle = joblib.load('bundle_final.pkl')
model = bundle['model']
scaler = bundle['scaler']
encoders = bundle['encoders']

st.set_page_config(page_title="Prédicteur de Salaire", layout="centered")
st.title("💰 Estimation de Salaire")

# --- 2. SAISIE DES DONNÉES (Widgets) ---
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Métier", options=encoders['job_title'].classes_)
    experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
    education_level = st.selectbox("Niveau d'études", options=encoders['education_level'].classes_)
    skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)

with col2:
    industry = st.selectbox("Secteur", options=encoders['industry'].classes_)
    company_size = st.selectbox("Taille entreprise", options=encoders['company_size'].classes_)
    location = st.selectbox("Ville", options=encoders['location'].classes_)
    remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
    certifications = st.number_input("Certifications", min_value=0, value=2)

# --- 3. CRÉATION DU DICTIONNAIRE (Étape clé) ---
# On crée l'objet AVANT de lui appliquer la loi
input_row = {
    'job_title': job_title,
    'experience_years': experience_years,
    'education_level': education_level,
    'skills_count': skills_count,
    'industry': industry,
    'company_size': company_size,
    'location': location,
    'remote_work': remote_work,
    'certifications': certifications
}

# --- 4. LA LOI DE 30 & PRÉVENTION ---
LIMITE = 30

if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ **Note :** Pour garantir la fiabilité, les calculs sont plafonnés à {LIMITE} (Loi de saturation).")

# Application technique de la saturation
input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 5. PRÉDICTION ---
if st.button("Calculer le salaire estimé"):
    # On transforme le dictionnaire en DataFrame
    df_pred = pd.DataFrame([input_row])
    
    # Encodage des textes
    for col, enc in encoders.items():
        df_pred[col] = enc.transform(df_pred[col])
        
    # Scaling et Prédiction
    final_data = scaler.transform(df_pred)
    resultat = model.predict(final_data)
    
    st.success(f"### Salaire estimé : {resultat[0]:,.2f} € / an")
