import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT DES FICHIERS ---
# On charge le modèle et les outils d'encodage
bundle = joblib.load('bundle_final.pkl')
model = bundle['model']
scaler = bundle['scaler']
encoders = bundle['encoders']

st.title("💰 Prédicteur de Salaire Professionnel")

# --- 2. RÉCUPÉRATION DES SAISIES (Widgets) ---
# On crée les variables à partir des formulaires
job_title = st.selectbox("Métier", options=encoders['job_title'].classes_)
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=encoders['education_level'].classes_)
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur", options=encoders['industry'].classes_)
company_size = st.selectbox("Taille entreprise", options=encoders['company_size'].classes_)
location = st.selectbox("Localisation", options=encoders['location'].classes_)
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Certifications", min_value=0, value=2)

# --- 3. CRÉATION DU DICTIONNAIRE (Étape indispensable) ---
# ON CRÉE L'OBJET ICI POUR QU'IL EXISTE DANS LA SUITE DU CODE
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

# --- 4. APPLICATION DE LA LOI DE 30 (SÉCURITÉ) ---
LIMITE = 30

# On vérifie si on doit afficher l'alerte
if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ **Note de fiabilité :** Les valeurs sont plafonnées à {LIMITE} pour ce calcul.")

# On applique la saturation technique AVANT la prédiction
input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 5. PRÉDICTION ---
if st.button("Estimer le salaire"):
    # On transforme le dictionnaire en DataFrame
    df_pred = pd.DataFrame([input_row])
    
    # Encodage des colonnes texte
    for col, enc in encoders.items():
        df_pred[col] = enc.transform(df_pred[col])
        
    # Scaling et Prédiction finale
    final_data = scaler.transform(df_pred)
    prediction = model.predict(final_data)
    
    # Affichage du résultat
    st.success(f"### Le salaire estimé est de : {prediction[0]:,.2f} €")
