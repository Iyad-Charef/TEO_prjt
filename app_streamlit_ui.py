# app_streamlit_ui.py
import streamlit as st
import requests
import pandas as pd

# --- Define options for selectboxes (these should be the ENGLISH keys expected by the FastAPI mapping dicts) ---
SEX_OPTIONS = ["Féminin", "Masculin"]
CP_TYPE_OPTIONS = ["Angine Typique (Type 0)", "Angine Atypique (Type 1)", "Douleur Non Angineuse (Type 2)", "Asymptomatique (Type 3)"]
FBS_OPTIONS = ["Non (<120 mg/dl)", "Oui (>120 mg/dl)"]
RESTECG_OPTIONS = ["Normal (Type 0)", "Anomalie ST-T (Type 1)", "HVG probable/définie (Type 2)"]
EXNG_OPTIONS = ["Non", "Oui"]
SLP_OPTIONS = ["Ascendante (Type 0)", "Plate (Type 1)", "Descendante (Type 2)"]
CAA_OPTIONS = ["0 Vaisseaux", "1 Vaisseau", "2 Vaisseaux", "3 Vaisseaux"] # Correspond à vsc_0 à vsc_3
THALL_OPTIONS = ["Normale (Thall-0)", "Défaut Fixe (Thall-1)", "Défaut Réversible (Thall-2)"] # Correspond à thall_0 à thall_2
# --- Feature Name Display Mapping (Short FRENCH Internal Name -> User-Friendly FRENCH Display Name) ---
# This map translates the short names received from the API explanation into readable French
MAP_AFFICHAGE_CARACTERISTIQUE = {
    'age': 'Âge',
    'sexe': 'Sexe', # This is the 0/1 feature itself
    'dt_0': 'Douleur Thoracique : Typique (Type 0)',
    'dt_1': 'Douleur Thoracique : Atypique (Type 1)',
    'dt_2': 'Douleur Thoracique : Non Angineuse (Type 2)',
    'dt_3': 'Douleur Thoracique : Asymptomatique (Type 3)',
    'par': 'Pression Artérielle Repos',
    'chol': 'Cholestérol Sérique',
    'gaj': 'Glycémie à Jeun > 120', # This is the 0/1 feature
    'ecg_0': 'ECG Repos : Normal (Type 0)',
    'ecg_1': 'ECG Repos : Anomalie ST-T (Type 1)',
    'ecg_2': 'ECG Repos : HVG (Type 2)',
    'fcmax': 'Fréquence Cardiaque Max',
    'angeff': 'Angine Induite par Effort', # This is the 0/1 feature
    'depst': 'Dépression ST (effort)',
    'pst_0': 'Pente ST : Ascendante (Type 0)',
    'pst_1': 'Pente ST : Plate (Type 1)',
    'pst_2': 'Pente ST : Descendante (Type 2)',
    'vsc_0': 'Vaisseaux Colorés : 0',
    'vsc_1': 'Vaisseaux Colorés : 1',
    'vsc_2': 'Vaisseaux Colorés : 2',
    'vsc_3': 'Vaisseaux Colorés : 3',
    'thall_0': 'Test Thallium : Normal (Type 0)', # Mapping internal thall_0
    'thall_1': 'Test Thallium : Défaut Fixe (Type 1)', # Mapping internal thall_1
    'thall_2': 'Test Thallium : Défaut Réversible (Type 2)' # Mapping internal thall_2
}


st.set_page_config(layout="wide", page_title="Prédicteur MC", page_icon="🩺") # Using French page title
st.title("Prédicteur de Risque de Maladie Cardiaque 🩺") # French Title
st.markdown("Entrez les détails du patient ci-dessous pour prédire la probabilité de maladie cardiaque.") # French Text
col1, col2, col3 = st.columns(3)

# Use French labels for all UI elements
with col1:
    st.subheader("Informations Démographiques") # French Subheader
    # Keep internal variable names English
    age_input = st.number_input("Âge (années)", min_value=20, max_value=100, value=55, key="age_input", help="Âge du patient en années.") # French Label & Help
    sexe_input_str = st.selectbox("Sexe", options=SEX_OPTIONS, index=1, key="sexe_input", help="Sexe biologique du patient.") # French Label & Help
    dt_input_str = st.selectbox("Type de Douleur Thoracique", options=CP_TYPE_OPTIONS, index=3, key="cp_type_input", help="Type de douleur thoracique ressentie.") # French Label & Help

with col2:
    st.subheader("Signes Vitaux & Analyses") # French Subheader
    par_input_val = st.number_input("Pression Artérielle au Repos (mmHg)", min_value=80, max_value=220, value=140, key="trestbps_input", help="Pression artérielle systolique au repos.") # French Label & Help
    chol_input_val = st.number_input("Cholestérol Sérique (mg/dl)", min_value=100, max_value=600, value=240, key="chol_input", help="Taux de cholestérol total.") # French Label & Help
    gaj_input_str = st.selectbox("Glycémie à Jeun > 120 mg/dl?", options=FBS_OPTIONS, key="fbs_input", help="La glycémie à jeun est-elle supérieure à 120 mg/dl?") # French Label & Help
    fcmax_input_val = st.number_input("Fréquence Cardiaque Maximale Atteinte", min_value=60, max_value=220, value=150, key="thalachh_input", help="Fréquence cardiaque la plus élevée pendant le test d'effort.") # French Label & Help

with col3:
    st.subheader("Tests d'Effort & ECG") # French Subheader
    ecg_input_str = st.selectbox("Résultat ECG au Repos", options=RESTECG_OPTIONS, key="restecg_type_input", help="Résultats de l'électrocardiogramme au repos.") # French Label & Help
    angeff_input_str = st.selectbox("Angine Induite par l'Effort?", options=EXNG_OPTIONS, key="exng_input", help="L'angine a-t-elle été induite par l'effort?") # French Label & Help
    depst_input_val = st.number_input("Dépression ST (induite par l'effort)", min_value=0.0, max_value=7.0, value=1.0, step=0.1, key="oldpeak_input", help="Dépression du segment ST par rapport au repos.") # French Label & Help
    pst_input_str = st.selectbox("Pente du Segment ST à l'Effort Max", options=SLP_OPTIONS, index=1, key="slp_type_input", help="La pente du segment ST pendant l'effort maximal.") # French Label & Help
    vsc_input_str = st.selectbox("Nombre de Vaisseaux Majeurs (Fluoroscopie)", options=CAA_OPTIONS, key="caa_value_input", help="Nombre de vaisseaux majeurs (0-3) colorés par fluoroscopie.") # French Label & Help
    thall_input_str = st.selectbox("Résultat Test au Thallium", options=THALL_OPTIONS, key="thall_type_input", help="Résultat du test d'effort au thallium.") # French Label & Help


if st.button("Prédire le Risque de Maladie Cardiaque", type="primary", use_container_width=True): # French Button Text
    # Prepare payload for FastAPI - keys here MUST match Pydantic model aliases in app_main.py
    # The *values* are the English strings selected by the user, which the API maps
    payload = {
        "age": age_input,
        "sexe": sexe_input_str,       # Alias is 'sexe', value is "Female" or "Male"
        "dt": dt_input_str,           # Alias is 'dt', value is "Typical Angina (Type 0)" etc.
        "par": par_input_val,         # Alias is 'par'
        "chol": chol_input_val,       # Alias is 'chol'
        "gaj": gaj_input_str,         # Alias is 'gaj'
        "ecg": ecg_input_str,         # Alias is 'ecg'
        "fcmax": fcmax_input_val,     # Alias is 'fcmax'
        "angeff": angeff_input_str,   # Alias is 'angeff'
        "depst": depst_input_val,     # Alias is 'depst'
        "pst": pst_input_str,         # Alias is 'pst'
        "vsc": vsc_input_str,         # Alias is 'vsc'
        "thall_type": thall_input_str # Alias is 'thall_type'
    }

    st.markdown("---")
    st.subheader("Analyse de la Prédiction") # French Subheader

    # API Call
    api_url = "http://127.0.0.1:8000/predire" # Endpoint name kept English for now
    try:
        with st.spinner("Analyse des données et génération de la prédiction..."): # French spinner text
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            results = response.json()

        if "erreur" in results: # Check for French error key from API
            st.error(f"Erreur de l'API : {results['erreur']}") # Display French error
        else:
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                prediction_label = results['prediction_label'] # Get French label from API
                # Use probability for class 1 (heart disease) - key kept English for robustness
                prob_hd = results['probability_class_1'] 

                if results['prediction_code'] == 1:
                    st.error(f"**Prédiction :** {prediction_label}") # Display French label
                    st.metric(label="Probabilité de Risque", value=f"{prob_hd:.1%}", delta_color="inverse") # French Label
                else:
                    st.success(f"**Prédiction :** {prediction_label}") # Display French label
                    st.metric(label="Probabilité de Risque", value=f"{prob_hd:.1%}", delta_color="normal") # French Label
                st.progress(prob_hd)

            # Explanation section
            if "explanation" in results and results["explanation"]["type"] == "coefficients":
                with res_col2:
                    st.write("**Principaux Facteurs Contributifs :**") # French Text
                    # Use English keys from API response structure
                    explanation_df = pd.DataFrame(results["explanation"]["top_features"]) 

                    for _, row in explanation_df.iterrows():
                        # Get the short French feature name from API response
                        feature_name_short_fr = row['feature'] 
                        # Map to user-friendly French display name using the dictionary
                        feature_display = MAP_AFFICHAGE_CARACTERISTIQUE.get(feature_name_short_fr, feature_name_short_fr.replace("_", " ").title())

                        original_val_display = row['original_input_value'] # Value is already user-friendly string/number
                        contribution = row['contribution_log_odds']
                        impact_color = "red" if contribution > 0 else "green"
                        # Using different emojis to avoid potential rendering issues, but kept concept
                        direction_arrow = "💘 Risque Augmenté" if contribution > 0 else "💖 Risque Diminué" 

                        st.markdown(f"""
                            <div style="margin-bottom: 5px; padding: 5px; border-left: 3px solid {impact_color};">
                                <small>
                                    <strong>{feature_display}</strong> (Votre entrée: <em>{original_val_display}</em>)<br>
                                    {direction_arrow}: <span style='color:{impact_color}; font-weight:bold;'>{contribution:+.3f}</span> (Impact Log-Odds)
                                </small>
                            </div>
                        """, unsafe_allow_html=True) # Text inside markdown is French

                    st.caption("<small>Contribution aux log-odds : Positif augmente le risque calculé, négatif le diminue. La magnitude indique la force de l'influence.</small>", unsafe_allow_html=True) # French caption
            else:
                with res_col2:
                    # French info message
                    st.info("L'explication détaillée n'a pas pu être générée ou une erreur s'est produite.")

    except requests.exceptions.ConnectionError:
         # French error message
        st.error(f"Erreur de Connexion : Impossible de joindre le service de prédiction à {api_url}. Assurez-vous que le backend est démarré.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur HTTP de l'API : {e.response.status_code} - {e.response.text}")
        st.write("Données envoyées à l'API (pour débogage) :")
        st.json(payload)
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite : {type(e).__name__} - {e}")
        if 'results' in locals() and isinstance(results, dict):
            st.write("Réponse brute de l'API (pour débogage) :")
            st.json(results)

st.markdown("---")
# French disclaimer
st.warning("🩺 **Avertissement :** Cet outil fournit des prédictions basées sur un modèle statistique et est destiné à des fins d'information uniquement. Il **ne remplace pas un avis médical professionnel, un diagnostic ou un traitement.** Consultez toujours un professionnel de la santé qualifié pour toute question médicale ou avant de prendre des décisions relatives à votre santé.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; font-size: 0.9em; color: #6c757d;">
        <em>Made by <strong>Iyad Charef</strong> and <strong>Nadjib Allioua</strong></em><br>
        <em><strong>@ESI.algiers</strong></em>
    </div>
    """,
    unsafe_allow_html=True
)
