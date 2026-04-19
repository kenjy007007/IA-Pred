import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT (Toujours en haut) ---
bundle = joblib.load('bundle_final.pkl')
model = bundle['model']
scaler = bundle['scaler']
encoders = bundle['encoders']

st.title("💰 Prédicteur de Salaire Professionnel")

# --- 2. SAISIE DES DONNÉES (Widgets) ---
job_title = st.selectbox("Métier", options=encoders['job_title'].classes_)
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=encoders['education_level'].classes_)
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur", options=encoders['industry'].classes_)
company_size = st.selectbox("Taille entreprise", options=encoders['company_size'].classes_)
location = st.selectbox("Localisation", options=encoders['location'].classes_)
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Certifications", min_value=0, value=2)

# --- 3. CONSTRUCTION ET LOI DE SATURATION ---
# ÉTAPE A : On crée le dictionnaire input_row d'abord
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

# ÉTAPE B : On applique la loi de 30 (SÉCURITÉ)
LIMITE = 30

if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ **Note de fiabilité :** Valeurs plafonnées à {LIMITE} pour ce calcul.")

input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 4. PRÉDICTION ---
if st.button("Estimer le salaire"):
    # On transforme en DataFrame
    df_pred = pd.DataFrame([input_row])
    
    # Encodage
    for col, enc in encoders.items():
        df_pred[col] = enc.transform(df_pred[col])
        
    # Scaling et Prédiction
    final_data = scaler.transform(df_pred)
    prediction = model.predict(final_data)
    
    st.success(f"### Salaire estimé : {prediction[0]:,.2f} €")
