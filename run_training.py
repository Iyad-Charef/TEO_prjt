# run_training.py
import pandas as pd
import numpy as np
import time
import os
import joblib

# Utilisation de pandas pour charger les données et joblib pour sauvegarder les modèles et le scaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import des modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# --- Configuration du projet ---
FULL_DATA_CSV = r"D:\ESI\1CP\Semester 2\TEO\ML\Data\big data fr.csv"  # Chemin du fichier CSV
TARGET_COLUMN = 'resultat'  # Colonne cible dans le CSV
OUTPUT_DIR = r"D:\ESI\1CP\Semester 2\TEO\ML\Models"  # Dossier pour enregistrer modèles et scaler
SCALER_FILENAME = os.path.join(OUTPUT_DIR, 'heart_disease_scaler.joblib')
RESULTS_CSV_FILENAME = os.path.join(OUTPUT_DIR, 'model_comparison_summary_fr.csv')
TEST_SPLIT_SIZE = 0.2  # 80% entraînement, 20% test
SPLIT_RANDOM_STATE = 42  # Graine pour la reproductibilité
LATENCY_TEST_RUNS = 1000  # Nombre de runs pour mesurer la latence

# Dictionnaire des modèles à entraîner
models_to_train = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=SPLIT_RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=SPLIT_RANDOM_STATE),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SPLIT_RANDOM_STATE)
}

# --- Chargement et division des données ---
def load_and_split_data(filename, target_col, test_size, random_state):
    # Affiche le chemin et charge le CSV
    print(f"--- Chargement des données depuis : {filename} ---")
    if not os.path.exists(filename):
        print(f"Erreur : Fichier non trouvé à {filename}")
        return None, None, None, None, None
    df = pd.read_csv(filename)
    print(f"Données chargées. Forme : {df.shape}")

    # Vérifie la présence de la colonne cible
    if target_col not in df.columns:
        print(f"Erreur : Colonne cible '{target_col}' introuvable.")
        return None, None, None, None, None

    # Sépare features et cible
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Affiche les noms des colonnes utilisées
    features = list(X.columns)
    print(f"Features pour entraînement : {features}")
    print(f"Forme X: {X.shape}, forme y: {y.shape}")

    # Divise en ensembles entraînement et test (stratifié)
    print(f"--- Division {int((1-test_size)*100)}% train / {int(test_size*100)}% test ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train X={X_train.shape}, y={y_train.shape}")
    print(f"Test  X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test, features

# --- Mise à l'échelle des données ---
def scale_data_and_save_scaler(X_train, X_test, scaler_path):
    print("--- Mise à l'échelle des données ---")
    scaler = StandardScaler()
    print("Fit du StandardScaler sur X_train...")
    X_train_scaled = scaler.fit_transform(X_train)
    print("Transform de X_test...")
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarde du scaler pour l'inférence
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler enregistré : {scaler_path}")
    return X_train_scaled, X_test_scaled

# --- Entraînement et sauvegarde des modèles ---
def train_and_save_models(models_dict, X_train_scaled, y_train, output_dir):
    print("--- Entraînement des modèles ---")
    os.makedirs(output_dir, exist_ok=True)
    trained_info = {}
    for name, model in models_dict.items():
        print(f"Entraînement de {name}...")
        start = time.time()
        model.fit(X_train_scaled, y_train)
        durée = time.time() - start
        print(f"Terminé en {durée:.2f}s. Sauvegarde...")

        # Chemin du fichier .joblib
        fname = os.path.join(output_dir, name.replace(' ', '_') + '.joblib')
        joblib.dump(model, fname)
        print(f"Modèle sauvegardé : {fname}")
        trained_info[name] = {'filename': fname, 'training_time_s': durée}
    return trained_info

# --- Évaluation des modèles ---
def evaluate_saved_models(trained_info, scaler_path, X_test, y_test, runs):
    print("--- Évaluation des modèles ---")
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    # Prêt pour mesurer latence sur une instance
    single = X_test_scaled[:1]

    results = []
    for name, info in trained_info.items():
        print(f"Évaluation de {name}...")
        model = joblib.load(info['filename'])

        # Taille et temps de chargement
        size_mb = os.path.getsize(info['filename'])/(1024*1024)
        # Prédiction batch
        start = time.time()
        y_pred = model.predict(X_test_scaled)
        batch_time = time.time() - start

        # Latence moyenne
        latencies = []
        for _ in range(runs):
            s = time.time(); model.predict(single); latencies.append(time.time()-s)
        latency_ms = np.mean(latencies)*1000

        # Calcul des métriques
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"F1={f1:.2f}, Acc={acc:.2f}, Temps prédiction={latency_ms:.2f}ms")
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Training Time (s)': info['training_time_s'],
            'Model Size (MB)': size_mb,
            'Predict Latency (ms)': latency_ms,
            'Predict Time Bulk (s)': batch_time
        })
    return results

# --- Point d'entrée ---
if __name__ == '__main__':
    print("=== Début entraînement & évaluation ===")
    X_train, X_test, y_train, y_test, features = load_and_split_data(
        FULL_DATA_CSV, TARGET_COLUMN, TEST_SPLIT_SIZE, SPLIT_RANDOM_STATE
    )
    if X_train is None:
        exit("Erreur chargement données.")

    X_train_scaled, X_test_scaled = scale_data_and_save_scaler(X_train, X_test, SCALER_FILENAME)
    trained = train_and_save_models(models_to_train, X_train_scaled, y_train, OUTPUT_DIR)
    final = evaluate_saved_models(trained, SCALER_FILENAME, X_test, y_test, LATENCY_TEST_RUNS)

    # Sauvegarde des résultats finaux
    df = pd.DataFrame(final)
    df = df.sort_values(by='F1 Score', ascending=False)
    df.to_csv(RESULTS_CSV_FILENAME, index=False)
    print(f"Résultats finaux enregistrés : {RESULTS_CSV_FILENAME}")
    print("=== Fin ===")
