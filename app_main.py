# app_main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Literal
import os

# --- Configuration ---

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the current file
MODEL_PATH = os.path.join(BASE_DIR, "Logistic_Regression.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "heart_disease_scaler.joblib")

# --- CRITICAL: This list uses the SHORT FRENCH feature names based on your updated CSV ---
FINAL_FEATURE_NAMES = [
    'age', 'sexe', 'dt_0', 'dt_1', 'dt_2', 'dt_3', 'par', 'chol', 'gaj',
    'ecg_0', 'ecg_1', 'ecg_2', 'fcmax', 'angeff', 'depst', 'pst_0', 'pst_1', 'pst_2',
    'vsc_0', 'vsc_1', 'vsc_2', 'vsc_3', 'thall_0', 'thall_1', 'thall_2'
]
# These were the original numerical features before any OHE (using short French names)
ORIGINAL_NUMERICAL_FEATURES = ['age', 'par', 'chol', 'fcmax', 'depst']


# --- Load Model and Scaler at Startup ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # Output translated
    print("Modèle et scaler chargés avec succès.")
except Exception as e:
    # Output translated
    print(f"ERREUR FATALE : Impossible de charger le modèle ou le scaler. L'application ne fonctionnera pas. Erreur : {e}")
    model = None
    scaler = None

app = FastAPI(
    title="API de Prédiction de Maladie Cardiaque",
    description="Une API pour prédire le risque de maladie cardiaque basée sur les données du patient.",
    version="1.0.0"
)

# --- MAPPING DICTIONARIES (Using ENGLISH descriptive keys from the provided code's input model) ---
# These map the descriptive English strings (that Streamlit will send based on the previous code)
# to the numerical values needed internally.
SEX_MAP = {"Féminin": 0, "Masculin": 1}
FBS_MAP = {"Non (<120 mg/dl)": 0, "Oui (>120 mg/dl)": 1}
EXNG_MAP = {"Non": 0, "Oui": 1}

CP_BASE_MAP = {"Angine Typique (Type 0)": 0, "Angine Atypique (Type 1)": 1, "Douleur Non Angineuse (Type 2)": 2, "Asymptomatique (Type 3)": 3}
_cp_categories = 4

RESTECG_BASE_MAP = {"Normal (Type 0)": 0, "Anomalie ST-T (Type 1)": 1, "HVG probable/définie (Type 2)": 2}
_restecg_categories = 3

SLP_BASE_MAP = {"Ascendante (Type 0)": 0, "Plate (Type 1)": 1, "Descendante (Type 2)": 2}
_slp_categories = 3

CAA_BASE_MAP = {"0 Vaisseaux": 0, "1 Vaisseau": 1, "2 Vaisseaux": 2, "3 Vaisseaux": 3}
_caa_categories = 4

THALL_BASE_MAP = {"Normale (Thall-0)":0, "Défaut Fixe (Thall-1)": 1, "Défaut Réversible (Thall-2)": 2}
_thall_categories = 3


class UserPatientInput(BaseModel): # Pydantic model definition remains mostly English internally
    # Field names are internal, aliases match expected keys from frontend
    # Title attributes are translated for API docs
    age: int = Field(..., example=55, ge=20, le=100, title="Âge")
    sex_str: Literal["Féminin", "Masculin"] = Field(..., example="Féminin", alias="sexe", title="Sexe")
    cp_type_str: Literal["Angine Typique (Type 0)", "Angine Atypique (Type 1)", "Douleur Non Angineuse (Type 2)", "Asymptomatique (Type 3)"] = Field(..., example="Asymptomatique (Type 3)", alias="dt", title="Type Douleur Thoracique")
    trestbps: int = Field(..., example=140, ge=80, le=220, alias="par", title="Pression Artérielle Repos") # Internal 'trestbps' maps to alias 'par'
    chol: int = Field(..., example=240, ge=100, le=600, alias="chol", title="Cholestérol Sérique") # Internal 'chol' maps to alias 'chol'
    fbs_str: Literal["Non (<120 mg/dl)", "Oui (>120 mg/dl)"] = Field(..., example="Non (<120 mg/dl)", alias="gaj", title="Glycémie à Jeun > 120?")
    restecg_type_str: Literal["Normal (Type 0)", "Anomalie ST-T (Type 1)", "HVG probable/définie (Type 2)"] = Field(..., example="Normal (Type 0)", alias="ecg", title="Résultat ECG Repos")
    thalachh: int = Field(..., example=150, ge=60, le=220, alias="fcmax", title="Fréq. Cardiaque Max Atteinte") # Internal 'thalachh' maps to alias 'fcmax'
    exng_str: Literal["Non", "Oui"] = Field(..., example="Non", alias="angeff", title="Angine Induite par Effort?")
    oldpeak: float = Field(..., example=1.0, ge=0.0, le=7.0, alias="depst", title="Dépression ST (effort)") # Internal 'oldpeak' maps to alias 'depst'
    slp_type_str: Literal["Ascendante (Type 0)", "Plate (Type 1)", "Descendante (Type 2)"] = Field(..., example="Plate (Type 1)", alias="pst", title="Pente ST Effort Max")
    caa_value_str: Literal["0 Vaisseaux", "1 Vaisseau", "2 Vaisseaux", "3 Vaisseaux"] = Field(..., example="0 Vaisseaux", alias="vsc", title="Nb Vaisseaux Majeurs (fluoro)")
    thall_type_str: Literal["Normale (Thall-0)", "Défaut Fixe (Thall-1)", "Défaut Réversible (Thall-2)"] = Field(..., example="Normale (Thall-0)", alias="thall_type", title="Résultat Test Thallium")

@app.post("/predire", summary="Prédire le risque de maladie cardiaque") 
async def predict_heart_disease(data: UserPatientInput): 
    if model is None or scaler is None:
        return {"erreur": "Modèle ou Scaler non chargé correctement. Veuillez vérifier les journaux du serveur."}

    input_dict_raw = data.model_dump(by_alias=True) # Use aliases to get keys (expecting sexe, dt, par etc.)
    processed_features = {}

    # 1. Directly assign original numerical features
    processed_features['age'] = input_dict_raw['age'] # input_dict_raw key matches alias 'age' implicitly
    processed_features['par'] = input_dict_raw['par'] # input_dict_raw key matches alias 'par'
    processed_features['chol'] = input_dict_raw['chol'] # input_dict_raw key matches alias 'chol'
    processed_features['fcmax'] = input_dict_raw['fcmax'] # input_dict_raw key matches alias 'fcmax'
    processed_features['depst'] = input_dict_raw['depst'] # input_dict_raw key matches alias 'depst'
    
    # Map simple binary features 
    processed_features['sexe'] = SEX_MAP[input_dict_raw['sexe']] # input_dict_raw key matches alias 'sexe'
    processed_features['gaj'] = FBS_MAP[input_dict_raw['gaj']] # input_dict_raw key matches alias 'gaj' (was fbs_str before)
    processed_features['angeff'] = EXNG_MAP[input_dict_raw['angeff']] # input_dict_raw key matches alias 'angeff' (was exng_str before)

    # 2. Prepare for features that are already one-hot encoded in FINAL_FEATURE_NAMES
    # User selects a category (English string), we map to its base index, then set the correct OHE column (short French name) to 1.
    
    cp_base_val = CP_BASE_MAP[input_dict_raw['dt']] # input_dict_raw key matches alias 'dt'
    for i in range(_cp_categories): processed_features[f'dt_{i}'] = 1 if cp_base_val == i else 0

    restecg_base_val = RESTECG_BASE_MAP[input_dict_raw['ecg']] # input_dict_raw key matches alias 'ecg'
    for i in range(_restecg_categories): processed_features[f'ecg_{i}'] = 1 if restecg_base_val == i else 0
        
    slp_base_val = SLP_BASE_MAP[input_dict_raw['pst']] # input_dict_raw key matches alias 'pst'
    for i in range(_slp_categories): processed_features[f'pst_{i}'] = 1 if slp_base_val == i else 0

    caa_base_val = CAA_BASE_MAP[input_dict_raw['vsc']] # input_dict_raw key matches alias 'vsc'
    for i in range(_caa_categories): processed_features[f'vsc_{i}'] = 1 if caa_base_val == i else 0
        
    thall_base_val = THALL_BASE_MAP[input_dict_raw['thall_type']] # input_dict_raw key matches alias 'thall_type'
    for i in range(_thall_categories): processed_features[f'thall_{i}'] = 1 if thall_base_val == i else 0

    # 3. Create DataFrame in the EXACT order of FINAL_FEATURE_NAMES
    try:
        input_features_list = [processed_features[name] for name in FINAL_FEATURE_NAMES]
        input_df_ordered = pd.DataFrame([input_features_list], columns=FINAL_FEATURE_NAMES)
    except KeyError as e:
         # Return error message in French
        return {"erreur": f"Erreur de construction de caractéristique : {e}. Vérifiez les mappages/logique OHE."}
    except Exception as e:
        return {"erreur": f"Erreur lors de la préparation du DataFrame d'entrée : {e}"}

    # 4. Scale the ENTIRE DataFrame
    try:
        input_scaled_array = scaler.transform(input_df_ordered)
        input_scaled_df = pd.DataFrame(input_scaled_array, columns=FINAL_FEATURE_NAMES)
    except ValueError as ve:
        # Return error message in French
        return {"erreur": f"Erreur de mise à l'échelle de l'entrée. Discordance probable des caractéristiques avec le scaler. Détails : {ve}."}
    except Exception as e:
        return {"erreur": f"Erreur de mise à l'échelle de l'entrée : {e}"}

    # 5. Predict
    prediction = model.predict(input_scaled_df)[0]
    probabilities = model.predict_proba(input_scaled_df)[0].tolist()

    # 6. Explain (Logistic Regression)
    explanation = {} # Keep variable names English
    if isinstance(model, LogisticRegression):
        coefficients = model.coef_[0]
        feature_contributions = []
        scaled_feature_values = input_scaled_df.iloc[0]

        for feature_name, coeff, scaled_value in zip(FINAL_FEATURE_NAMES, coefficients, scaled_feature_values):
            original_user_value_display = "N/A" # Placeholder
            # Map back to user-friendly input string for display (using input_dict_raw and aliases)
            if feature_name == 'age': original_user_value_display = input_dict_raw['age']
            elif feature_name == 'sexe': original_user_value_display = input_dict_raw['sexe']
            elif feature_name == 'par': original_user_value_display = input_dict_raw['par']
            elif feature_name == 'chol': original_user_value_display = input_dict_raw['chol']
            elif feature_name == 'gaj': original_user_value_display = input_dict_raw['gaj']
            elif feature_name == 'fcmax': original_user_value_display = input_dict_raw['fcmax']
            elif feature_name == 'angeff': original_user_value_display = input_dict_raw['angeff']
            elif feature_name == 'depst': original_user_value_display = input_dict_raw['depst']
            elif feature_name.startswith("dt_") and processed_features.get(feature_name) == 1: original_user_value_display = input_dict_raw['dt']
            elif feature_name.startswith("ecg_") and processed_features.get(feature_name) == 1: original_user_value_display = input_dict_raw['ecg']
            elif feature_name.startswith("pst_") and processed_features.get(feature_name) == 1: original_user_value_display = input_dict_raw['pst']
            elif feature_name.startswith("vsc_") and processed_features.get(feature_name) == 1: original_user_value_display = input_dict_raw['vsc']
            elif feature_name.startswith("thall_") and processed_features.get(feature_name) == 1: original_user_value_display = input_dict_raw['thall_type']
            elif processed_features.get(feature_name, 0) == 0 and not (feature_name in ORIGINAL_NUMERICAL_FEATURES or feature_name in ['sexe','gaj','angeff']):
                continue

            contribution_log_odds = coeff * scaled_value
            if abs(contribution_log_odds) > 0.001 or (feature_name in ORIGINAL_NUMERICAL_FEATURES or feature_name in ['sexe','gaj','angeff']) :
                # Keep explanation keys English for structure, but value is user input
                feature_contributions.append({
                    "feature": feature_name, # Short French feature name
                    "original_input_value": original_user_value_display, # English descriptive string from user
                    "scaled_value_fed_to_model": round(scaled_value, 3),
                    "coefficient": round(coeff, 3),
                    "contribution_log_odds": round(contribution_log_odds, 3)
                })
        
        sorted_contributions = sorted(feature_contributions, key=lambda item: abs(item["contribution_log_odds"]), reverse=True)
        # Keep explanation keys English
        explanation = {"type": "coefficients", "top_features": sorted_contributions[:len(sorted_contributions)]} 
    else:
         # Return error message in French
        explanation = {"type": "erreur_modele", "message": "La logique d'explication est pour la Régression Logistique. Le modèle chargé est peut-être différent."}

    # Return results with French labels for user-facing strings, but keep keys English
    return {
        "prediction_code": int(prediction),
        "prediction_label": "Maladie Cardiaque Probable" if prediction == 1 else "Maladie Cardiaque Improbable", # French label
        "probability_class_0": round(probabilities[0], 4), # Prob class 0
        "probability_class_1": round(probabilities[1], 4), # Prob class 1
        "explanation": explanation
    }

# To run: uvicorn app_main:app --reload
