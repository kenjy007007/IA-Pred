import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT DES OUTILS ---
# Assure-toi que le fichier bundle_final.pkl est dans le même dossier
bundle = joblib.load('bundle_final.pkl')
model = bundle['model']
scaler = bundle['scaler']
encoders = bundle['encoders']
feature_names = bundle['feature_names']

st.title("Prédicteur de Salaire Professionnel")

# --- 2. SAISIE DES DONNÉES (Widgets) ---
# On crée les variables à partir des choix de l'utilisateur
job_title = st.selectbox("Métier", options=encoders['job_title'].classes_)
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=encoders['education_level'].classes_)
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur d'activité", options=encoders['industry'].classes_)
company_size = st.selectbox("Taille de l'entreprise", options=encoders['company_size'].classes_)
location = st.selectbox("Localisation", options=encoders['location'].classes_)
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Nombre de certifications", min_value=0, value=2)

# --- 3. CRÉATION DU DICTIONNAIRE ET LOI DE 30 ---
# ÉTAPE CRUCIALE : On crée input_row d'abord
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

# Maintenant on applique la loi (Saturation)
LIMITE = 30

if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ **Note de fiabilité :** Les valeurs sont plafonnées à {LIMITE} pour ce calcul.")

input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 4. PRÉDICTION ---
if st.button("Estimer le salaire"):
    # Transformation en DataFrame
    df_pred = pd.DataFrame([input_row])
    
    # Encodage des colonnes textuelles
    for col, enc in encoders.items():
        df_pred[col] = enc.transform(df_pred[col])
        
    # Normalisation et Prédiction finale
    final_data = scaler.transform(df_pred)
    resultat = model.predict(final_data)
    
    st.success(f"### Le salaire estimé est de : {resultat[0]:,.0f} €")
