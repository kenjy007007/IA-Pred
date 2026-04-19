import streamlit as st
import pandas as pd
import joblib

# --- 1. CHARGEMENT SÉCURISÉ ---
try:
    bundle = joblib.load('bundle_final.pkl')
    model = bundle['model']
    scaler = bundle['scaler']
    encoders = bundle['encoders']
    # On récupère l'ordre exact des colonnes utilisé à l'entraînement
    feature_names = bundle['feature_names']
except Exception as e:
    st.error(f"Erreur : Impossible de charger le bundle. Vérifiez le fichier .pkl. {e}")

st.title("💰 Prédicteur de Salaire Professionnel")

# --- 2. WIDGETS DE SAISIE ---
# On utilise directement les classes des encodeurs pour être 100% sûr des labels
job_title = st.selectbox("Métier", options=list(encoders['job_title'].classes_))
experience_years = st.number_input("Années d'expérience", min_value=0, value=5)
education_level = st.selectbox("Niveau d'études", options=list(encoders['education_level'].classes_))
skills_count = st.number_input("Nombre de compétences", min_value=0, value=10)
industry = st.selectbox("Secteur", options=list(encoders['industry'].classes_))
company_size = st.selectbox("Taille entreprise", options=list(encoders['company_size'].classes_))
location = st.selectbox("Localisation", options=list(encoders['location'].classes_))
remote_work = st.selectbox("Télétravail", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
certifications = st.number_input("Certifications", min_value=0, value=2)

# --- 3. PRÉPARATION ET LOI DE 30 ---
# On crée le dictionnaire avec les noms EXACTS de ton dataset
input_data = {
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

# Application de ta loi de saturation
LIMITE = 30
if any(input_data[k] > LIMITE for k in ['experience_years', 'skills_count', 'certifications']):
    st.warning(f"⚠️ Note : Valeurs plafonnées à {LIMITE} pour garantir la stabilité.")
    input_data['experience_years'] = min(input_data['experience_years'], LIMITE)
    input_data['skills_count'] = min(input_data['skills_count'], LIMITE)
    input_data['certifications'] = min(input_data['certifications'], LIMITE)

# --- 4. LOGIQUE DE PRÉDICTION ---
if st.button("Estimer le salaire"):
    try:
        # ÉTAPE A : Créer le DataFrame et FORCER l'ordre des colonnes
        df_pred = pd.DataFrame([input_data])
        df_pred = df_pred[feature_names] # <--- C'est ça qui règle souvent les erreurs !

        # ÉTAPE B : Encodage des textes
        for col, enc in encoders.items():
            df_pred[col] = enc.transform(df_pred[col].astype(str))
            
        # ÉTAPE C : Scaling et Prédiction
        final_data = scaler.transform(df_pred)
        prediction = model.predict(final_data)
        
        st.success(f"### Salaire estimé : {prediction[0]:,.2f} €")

    except ValueError as e:
        st.error(f"❌ Erreur de correspondance : {e}")
        st.info("L'IA n'a pas reconnu une des catégories choisies.")
    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
