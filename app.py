import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURATION & CHARGEMENT ---
st.set_page_config(page_title="IA Salaire", layout="centered")

try:
    # On charge le bundle qui contient TOUT (modèle, scaler, encoders, noms des colonnes)
    bundle = joblib.load('bundle_final.pkl')
    model = bundle['model']
    scaler = bundle['scaler']
    encoders = bundle['encoders']
    feature_names = bundle['feature_names']
except Exception as e:
    st.error(f"Erreur de chargement du fichier .pkl : {e}")
    st.stop() # Arrête l'application si le fichier est introuvable

st.title("💰 Prédicteur de Salaire Professionnel")
st.write("Remplissez les informations ci-dessous pour obtenir une estimation.")

# --- 2. CRÉATION DES WIDGETS (Saisie) ---
# On utilise les .classes_ pour être sûr que l'IA connaisse les mots choisis
job_title = st.selectbox("Métier", options=list(encoders['job_title'].classes_))
experience_years = st.number_input("Années d'expérience", min_value=0, max_value=50, value=5)
education_level = st.selectbox("Niveau d'études", options=list(encoders['education_level'].classes_))
skills_count = st.number_input("Nombre de compétences", min_value=0, max_value=100, value=10)
industry = st.selectbox("Secteur d'activité", options=list(encoders['industry'].classes_))
company_size = st.selectbox("Taille de l'entreprise", options=list(encoders['company_size'].classes_))
location = st.selectbox("Localisation (Ville)", options=list(encoders['location'].classes_))
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Nombre de certifications", min_value=0, max_value=50, value=2)

# --- 3. CONSTRUCTION DU DICTIONNAIRE ---
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

# --- 4. LA LOI DE 30 (SÉCURITÉ) ---
LIMITE = 30

# Vérification pour l'affichage du message
if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ **Note :** Les valeurs sont plafonnées à {LIMITE} pour garantir la précision du modèle.")

# Application technique du plafonnement
input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 5. LOGIQUE DE PRÉDICTION ---
if st.button("Estimer le salaire"):
    try:
        # A. Créer le DataFrame
        df_pred = pd.DataFrame([input_row])
        
        # B. Forcer l'ordre des colonnes pour correspondre à l'entraînement
        df_pred = df_pred[feature_names]
        
        # C. Encodage des textes (Label Encoding)
        for col, enc in encoders.items():
            df_pred[col] = enc.transform(df_pred[col])
            
        # D. Normalisation (Scaling)
        final_data = scaler.transform(df_pred)
        
        # E. Prédiction finale
        prediction = model.predict(final_data)
        
        # Affichage du résultat avec mise en forme
        st.success(f"### Salaire estimé : {prediction[0]:,.0f} € / an")
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul : {e}")
