import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT DU BUNDLE ---
bundle = joblib.load('bundle_final.pkl')
model = bundle['model']
scaler = bundle['scaler']
encoders = bundle['encoders']
feature_names = bundle['feature_names']

st.title("Prédicteur de Salaire Professionnel")

# --- 2. INTERFACE UTILISATEUR (Widgets) ---
# On crée les champs un par un en respectant l'ordre de ton dataset
job_title = st.selectbox("Métier", options=encoders['job_title'].classes_)
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=encoders['education_level'].classes_)
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur d'activité", options=encoders['industry'].classes_)
company_size = st.selectbox("Taille de l'entreprise", options=encoders['company_size'].classes_)
location = st.selectbox("Localisation", options=encoders['location'].classes_)
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Nombre de certifications", min_value=0, value=2)

# --- 3. LA LOI DE 30 & PRÉVENTION ---
LIMITE = 30

# On prépare les données pour l'IA en appliquant le plafonnement
exp_ia = min(experience_years, LIMITE)
skills_ia = min(skills_count, LIMITE)
certif_ia = min(certifications, LIMITE)

# Affichage du message si on dépasse la limite
if experience_years > LIMITE or skills_count > LIMITE or certifications > LIMITE:
    st.warning(f"⚠️ **Note de fiabilité :** Le système plafonne les valeurs à {LIMITE} pour rester cohérent avec les données réelles.")

# --- 4. CALCUL DE LA PRÉDICTION ---
if st.button("Estimer le salaire"):
    # On crée le dictionnaire avec les valeurs SATURÉES (exp_ia, etc.)
    input_data = {
        'job_title': job_title,
        'experience_years': exp_ia,
        'education_level': education_level,
        'skills_count': skills_ia,
        'industry': industry,
        'company_size': company_size,
        'location': location,
        'remote_work': remote_work,
        'certifications': certif_ia
    }
    
    # Transformation en DataFrame
    df_pred = pd.DataFrame([input_data])
    
    # Encodage des textes
    for col, enc in encoders.items():
        df_pred[col] = enc.transform(df_pred[col])
        
    # Scaling et Prédiction
    final_data = scaler.transform(df_pred)
    resultat = model.predict(final_data)
    
    # Affichage final
    st.success(f"### Le salaire estimé est de : {resultat[0]:,.0f} €")
