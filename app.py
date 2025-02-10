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
import os

# Fonction pour charger les données depuis Google Drive et convertir en Parquet
@st.cache_data
def load_data(parquet_file, csv_file, url):
    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file)
    else:
        gdown.download(url, csv_file, quiet=False)
        df = pd.read_csv(csv_file, sep=',', on_bad_lines='skip', index_col=0)
        df.to_parquet(parquet_file, engine='fastparquet', compression='snappy')
        return df

# Définition des URLs des fichiers Google Drive
url_df_model = 'https://drive.google.com/uc?id=1-Fuva7dJ7evX8MSDtBTUuwSFmsaIf2Ow'
url_df_cleaned2023 = 'https://drive.google.com/uc?id=1g2dDfywtZK9BTWFp2u0i0MqO98U5S8NC'

# Chargement des fichiers en priorité depuis Parquet
df_model = load_data('df_model.parquet', 'df_model.csv', url_df_model)
dffinal2023 = load_data('df_model_2023.parquet', 'df_model_2023.csv', url_df_cleaned2023)
df_model = pd.concat([df_model, dffinal2023])

# Prétraitement
df_model['dep'] = df_model['dep'].astype('object')
df_model["grav"] = df_model["grav"].map({'Accident léger': 0, 'Accident grave': 1, 'Accident mortel': 1})

X = df_model.drop("grav", axis=1)
y = df_model["grav"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
cat_cols = X_train.select_dtypes(include=["object"]).columns

# Imputation et encodage
categorical_imputer = SimpleImputer(strategy='most_frequent')
cat_train_imputed = pd.DataFrame(categorical_imputer.fit_transform(X_train), columns=cat_cols)
cat_test_imputed = pd.DataFrame(categorical_imputer.transform(X_test), columns=cat_cols)

encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
cat_train_encoded = pd.DataFrame(encoder.fit_transform(cat_train_imputed), columns=encoder.get_feature_names_out(cat_cols))
cat_test_encoded = pd.DataFrame(encoder.transform(cat_test_imputed), columns=encoder.get_feature_names_out(cat_cols))

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(cat_train_encoded, y_train)

scale_pos_weight = len(y_train) / (2 * sum(y_train == 1))
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

st.title("Analyse et Prédiction des Accidents Routiers en France métropolitaine")
st.sidebar.title("Navigation")
pages = ["Exploration", "Analyse des facteurs de risque", "Outil de prédiction"]
page = st.sidebar.radio("Aller vers", pages)

if page == "Outil de prédiction":
    st.write("### Simulation de Prédiction de Gravité d'un Accident")
    user_input = {}

    for col in cat_cols:
        user_input[col] = st.selectbox(f"{col}", X_train[col].astype(str).unique())

    input_df = pd.DataFrame([user_input])
    missing_cols = set(cat_cols) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = "Unknown"

    try:
        input_df = input_df.astype(str)
        input_encoded = encoder.transform(input_df)
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(cat_cols))

        prediction_prob = model.predict_proba(input_encoded_df)

        st.subheader("Résultat de la prédiction")
        st.write(f"Probabilité d'un **Accident Léger** : {prediction_prob[0][0] * 100:.2f}%")
        st.write(f"Probabilité d'un **Accident Grave** : {prediction_prob[0][1] * 100:.2f}%")

    except Exception as e:
        st.error(f"Erreur lors de la transformation des données : {e}")
