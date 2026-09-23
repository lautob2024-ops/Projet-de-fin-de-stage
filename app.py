import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Configuration de la page
st.set_page_config(page_title="Prédiction Rendement Agricole Bénin", page_icon="🌽", layout="centered")

st.title("🌾 Application de Prédiction du Rendement Agricole")
st.write("Saisissez les informations de l'exploitation pour obtenir une prédiction du rendement.")

# 2. Chargement du modèle et du préprocesseur
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prep_path = os.path.join(base_dir, 'preprocessor.joblib')
    model_path = os.path.join(base_dir, 'model.joblib')
    
    prep = joblib.load(prep_path)
    model = joblib.load(model_path)
    return prep, model

prep, model = load_assets()

# 3. Listes des produits et des communes
LISTE_PRODUITS = sorted([
    'CRINCRIN', 'ANACARDE', 'PIMENT', "POIDS D'ANGOLE", 'MANIOC', 'RIZ', 
    'TOMATE', 'GOMBO', 'HARICOT VERT', 'CITULUS', 'ARACHIDE', 'CHOUX', 
    'CONCOMBRE', 'GOUSSI', 'NIEBE', 'POMME TERRE', 'FONIO', 'SOJA', 
    'COTON', 'VOANDZOU', 'PATATE DOUCE', 'PASTEQUE', 'GBOMA', 'MAIS', 
    'CAROTTE', 'PETIT MIL', 'SORGHO', 'IGNAME', 'TARO', 'SESAME', 
    'OIGNON', 'LAITUE', 'ANANAS', 'DOHI'
])

# Liste des communes du Bénin
LISTE_COMMUNES = sorted([
    "BANIKOARA", "KEROU", "GOGOUNOU", "KANDI", "BEMBEREKE", "SAVALOU", 
    "DASSA-ZOUME", "TANGUIETA", "MATERI", "NATITINGOU", "BOUKOMBE", 
    "N'DALI", "PERERE", "PARAKOU", "TCHAOUROU", "COBLY", "DJOUGOU", 
    "BASSILA", "KAPASSI", "MALANVILLE", "KARIMAMA", "SINENDE", "KALALE"
])

LISTE_ANNEES = [str(y) for y in range(2024, 2031)]

# 4. Formulaire de saisie des données
with st.form("form_prediction"):
    col1, col2 = st.columns(2)
    
    with col1:
        commune = st.selectbox("Commune", options=LISTE_COMMUNES)
        produit = st.selectbox("Produit / Culture", options=LISTE_PRODUITS)
        superficie = st.number_input("Superficie (en hectares)", min_value=0.1, value=1.0, step=0.5)
        annee = st.selectbox("Année", options=LISTE_ANNEES)

    with col2:
        st.subheader("Données Météo")
        haut_cumul = st.number_input("HAUT Cumul (Pluviométrie mm)", min_value=0.0, value=0.0)
        nj_cumul = st.number_input("Nombre de Jours de Pluie (NJ)", min_value=0, value=0)

    submitted = st.form_submit_button("Calculer le rendement")

# 5. Logique de prédiction
if submitted:
    # On fournit à la fois 'commune'/'communes' et 'HAUT_cumul'/'Haut_cumul' 
    # pour garantir la compatibilité exacte avec le préprocesseur
    donnees = pd.DataFrame([{
        'commune': str(commune),
        'communes': str(commune),
        'superficie': float(superficie),
        'HAUT_cumul': float(haut_cumul),
        'Haut_cumul': float(haut_cumul),
        'NJ_cumul': float(nj_cumul),
        'produit': str(produit),
        'annee': str(annee)
    }])

    try:
        # Transformation des données et prédiction
        donnees_traitees = prep.transform(donnees)
        pred_log = model.predict(donnees_traitees)
        
        # Inversion log (np.expm1)
        rendement_estime = round(np.expm1(pred_log)[0], 2)
        production_totale = round(rendement_estime * superficie, 2)
        
        st.success(f"🌾 **Rendement Estimé pour {produit} :** {rendement_estime} kg / ha")
        st.metric(label=f"Production totale estimée ({superficie} ha)", value=f"{production_totale} kg")
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")