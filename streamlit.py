import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.colors import ListedColormap
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, precision_score, recall_score, roc_auc_score, auc
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost
import sklearn
import gdown

# Chargement des données
# ID du fichier Google Drive (à récupérer dans l'URL de partage)
dffinal_ID = "1qgriVZfLRNW7ud1HgXLb4hYwRcel5RFo"
dffinal_file = "dffinal.csv"

# URL du fichier Google Drive
dffinal_URL = f"https://drive.google.com/uc?id={dffinal_ID}"

# Télécharger le fichier CSV
@st.cache_data
def load_data():
    gdown.download(dffinal_URL, dffinal_file, quiet=False)
    return pd.read_csv(dffinal_file)

# Charger les données
dffinal = load_data()

# ID du fichier Google Drive (à récupérer dans l'URL de partage)
df_model_ID = "1seaUfdCCc-0vY59l-Q_9X7dyukZPBqak"
df_model_file = "df_model.csv"

# URL du fichier Google Drive
df_model_URL = f"https://drive.google.com/uc?id={df_model_ID}"

# Télécharger le fichier CSV
@st.cache_data
def load_data_1():
    gdown.download(df_model_URL, df_model_file, quiet=False)
    return pd.read_csv(df_model_file)

# Charger les données
df_model = load_data_1()

# Prétraitement des données
df_model['dep'] = df_model['dep'].astype('object')
df_model["grav"] = df_model["grav"].map({'Accident léger': 0, 'Accident grave': 1, 'Accident mortel': 1})

# Séparation du jeu de données
X = df_model.drop("grav", axis=1)
y = df_model["grav"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
cat_cols = X_train.select_dtypes(include=["object"]).columns

# Imputation des valeurs manquantes
categorical_imputer = SimpleImputer(strategy='most_frequent')
cat_train_imputed = pd.DataFrame(categorical_imputer.fit_transform(X_train), columns=cat_cols)
cat_test_imputed = pd.DataFrame(categorical_imputer.transform(X_test), columns=cat_cols)

# Encodage
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
cat_train_encoded = pd.DataFrame(encoder.fit_transform(cat_train_imputed), columns=encoder.get_feature_names_out(cat_cols))
cat_test_encoded = pd.DataFrame(encoder.transform(cat_test_imputed), columns=encoder.get_feature_names_out(cat_cols))

# Rééchantillonnage SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(cat_train_encoded, y_train)

# Modèle XGBoost
scale_pos_weight = len(y_train) / (2 * sum(y_train == 1))
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Interface Streamlit
st.title("Prédiction et Analyse des Accidents Routiers en France métropolitaine: Un Outil d’Aide à la Décision")
st.sidebar.title("Sommaire")
pages = ["Exploration", "Analyse des facteurs de risque", "Outil de prédiction"]
page = st.sidebar.radio("Aller vers", pages)

if page == "Outil de prédiction":
    st.write("### Simulation de Prédiction de Gravité d'un Accident")

    user_input = {}
    
    for col in cat_cols:
        user_input[col] = st.selectbox(f"{col}", X_train[col].astype(str).unique())  # Convertir en string
    
    input_df = pd.DataFrame([user_input])
    
    # Vérifier et ajouter les colonnes manquantes
    missing_cols = set(cat_cols) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = "Unknown"  # Utiliser "Unknown" pour éviter les erreurs
    
    try:
        input_df = input_df.astype(str)  # Convertir toutes les valeurs en string
        input_encoded = encoder.transform(input_df)
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_cols))
        
        # Faire la prédiction avec probabilités
        prediction_prob = model.predict_proba(input_encoded_df)
        
        # Affichage du résultat
        st.subheader("Résultat de la prédiction")
        st.write(f"Probabilité d'un **Accident Léger** : {prediction_prob[0][0] * 100:.2f}%")
        st.write(f"Probabilité d'un **Accident Grave** : {prediction_prob[0][1] * 100:.2f}%")
        
    except Exception as e:
        st.error(f"Erreur lors de la transformation des données : {e}")
