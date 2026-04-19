import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT ---
try:
    bundle = joblib.load('bundle_final.pkl')
    model = bundle['model']
    scaler = bundle['scaler']
    encoders = bundle['encoders']
except Exception as e:
    st.error(f"Erreur de chargement du fichier bundle_final.pkl : {e}")

st.title("💰 Prédicteur de Salaire Professionnel")

# --- 2. SAISIE DES DONNÉES ---
# On utilise directement les classes de l'encodeur pour éviter les "unseen labels"
job_title = st.selectbox("Métier", options=list(encoders['job_title'].classes_))
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=list(encoders['education_level'].classes_))
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur", options=list(encoders['industry'].classes_))
company_size = st.selectbox("Taille entreprise", options=list(encoders['company_size'].classes_))
location = st.selectbox("Localisation", options=list(encoders['location'].classes_))
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Certifications", min_value=0, value=2)

# --- 3. CONSTRUCTION & LOI DE 30 ---
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

LIMITE = 30
if input_row['experience_years'] > LIMITE or input_row['skills_count'] > LIMITE or input_row['certifications'] > LIMITE:
    st.warning(f"⚠️ Valeurs plafonnées à {LIMITE} pour la précision.")

input_row['experience_years'] = min(input_row['experience_years'], LIMITE)
input_row['skills_count'] = min(input_row['skills_count'], LIMITE)
input_row['certifications'] = min(input_row['certifications'], LIMITE)

# --- 4. PRÉDICTION SÉCURISÉE ---
if st.button("Estimer le salaire"):
    try:
        df_pred = pd.DataFrame([input_row])
        
        # Encodage avec gestion d'erreur
        for col, enc in encoders.items():
            df_pred[col] = enc.transform(df_pred[col])
            
        final_data = scaler.transform(df_pred)
        prediction = model.predict(final_data)
        
        st.success(f"### Salaire estimé : {prediction[0]:,.2f} €")
        
    except ValueError as e:
        st.error("❌ Erreur de données : Une des valeurs choisies n'est pas reconnue par l'IA.")
        st.info("Conseil : Assurez-vous d'utiliser les options proposées dans les listes.")
    except Exception as e:
        st.error(f"❌ Une erreur imprévue est survenue : {e}")
