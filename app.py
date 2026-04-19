import streamlit as st
import pandas as pd
import joblib


# --- 1. COLLECTE DES DONNÉES (Widgets Streamlit) ---
# Assure-toi que ces noms correspondent à tes widgets
job_title = st.selectbox("Métier", options=encoders['job_title'].classes_)
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=encoders['education_level'].classes_)
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur d'activité", options=encoders['industry'].classes_)
company_size = st.selectbox("Taille de l'entreprise", options=encoders['company_size'].classes_)
location = st.selectbox("Localisation", options=encoders['location'].classes_)
remote_work = st.selectbox("Télétravail", options=[0, 1])
certifications = st.number_input("Nombre de certifications", min_value=0, value=2)

# --- 2. LA LOI DE 30 & PRÉVENTION ---
LIMITE = 30

# On crée le dictionnaire AVANT d'appliquer la loi
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

# MAINTENANT on applique la loi dans le dictionnaire
if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ **Note :** Valeurs plafonnées à {LIMITE} pour la précision.")
    
input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 3. PRÉDICTION ---
if st.button("Estimer le salaire"):
    df_pred = pd.DataFrame([input_row])
    # ... suite de ton code (encoders, scaler, model.predict)
