# ml_prediction_section_prediction_interactive.py

# Ce script contient la section "Prédiction de Performance (ML)" avec l'interface interactive
# pour prédire le temps de réponse (RESPTI).
# Il est conçu pour être ajouté à un dashboard Streamlit existant.

# --- Importations nécessaires pour cette section ---
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Fonctions et données externes requises ---
# Pour que ce script fonctionne de manière autonome ou intégré,
# les éléments suivants doivent être disponibles dans l'environnement d'exécution :
# 1. 'dfs' : Un dictionnaire contenant les DataFrames chargés, notamment 'dfs["hitlist_db"]'.
# 2. 'clean_string_column' : Une fonction pour nettoyer les colonnes de type chaîne.
#
# Si ce script est exécuté seul pour test, vous pouvez décommenter les exemples ci-dessous.
# Pour l'intégration dans votre dashboard principal, assurez-vous que ces éléments
# sont définis avant l'appel à cette section.

# Exemple de fonction clean_string_column si ce script est exécuté seul (à décommenter pour test)
# import re
# def clean_string_column(series, default_value="Non défini"):
#     cleaned_series = series.astype(str).str.strip()
#     cleaned_series = cleaned_series.apply(lambda x: re.sub(r'[^\x20-\x7E\s]+', ' ', x).strip())
#     cleaned_series = cleaned_series.replace({'nan': default_value, '': default_value, ' ': default_value})
#     return cleaned_series

# Exemple de DataFrame 'dfs' si ce script est exécuté seul (à décommenter pour test)
# if 'dfs' not in locals():
#     # Création d'un DataFrame fictif pour la démonstration
#     data_ml = {
#         'RESPTI': np.random.rand(500) * 1000 + 10, # Temps de réponse en ms (ajout de +10 pour éviter log(0))
#         'FULL_DATETIME': pd.to_datetime(pd.date_range(start='2023-01-01', periods=500, freq='H')),
#         'CPUTI': np.random.rand(500) * 500, 'PROCTI': np.random.rand(500) * 800,
#         'QUEUETI': np.random.rand(500) * 100, 'ROLLWAITTI': np.random.rand(500) * 50,
#         'GUITIME': np.random.rand(500) * 200, 'GUICNT': np.random.randint(1, 100, 500),
#         'GUINETTIME': np.random.rand(500) * 100, 'DBP_COUNT': np.random.randint(1, 50, 500),
#         'DBP_TIME': np.random.rand(500) * 300, 'DSQLCNT': np.random.randint(1, 100, 500),
#         'QUECNT': np.random.randint(1, 20, 500), 'CPICCNT': np.random.randint(0, 10, 500),
#         'SLI_CNT': np.random.randint(1, 50, 500), 'READDIRTI': np.random.rand(500) * 10,
#         'READDIRBUF': np.random.rand(500) * 5, 'READDIRREC': np.random.randint(1, 20, 500),
#         'READSEQTI': np.random.rand(500) * 10, 'READSEQBUF': np.random.rand(500) * 5,
#         'READSEQREC': np.random.randint(1, 20, 500), 'INSCNT': np.random.randint(1, 10, 500),
#         'INSTI': np.random.rand(500) * 5, 'INSREC': np.random.randint(1, 10, 500),
#         'PHYINSCNT': np.random.randint(1, 10, 500), 'UPDCNT': np.random.randint(1, 10, 500),
#         'UPDTI': np.random.rand(500) * 5, 'UPDREC': np.random.randint(1, 10, 500),
#         'PHYUPDCNT': np.random.randint(1, 10, 500), 'DELCNT': np.random.randint(1, 5, 500),
#         'DELTI': np.random.rand(500) * 2, 'DELREC': np.random.randint(1, 5, 500),
#         'PHYDELCNT': np.random.randint(1, 5, 500), 'DBCALLS': np.random.randint(1, 100, 500),
#         'COMMITTI': np.random.rand(500) * 5, 'INPUTLEN': np.random.rand(500) * 1000,
#         'OUTPUTLEN': np.random.rand(500) * 1000, 'MAXROLL': np.random.rand(500) * 100,
#         'MAXPAGE': np.random.rand(500) * 50, 'ROLLINCNT': np.random.randint(1, 10, 500),
#         'ROLLINTI': np.random.rand(500) * 5, 'ROLLOUTCNT': np.random.randint(1, 10, 500),
#         'ROLLOUTTI': np.random.rand(500) * 5, 'PRIVSUM': np.random.rand(500) * 1000,
#         'USEDBYTES': np.random.rand(500) * 2000, 'MAXBYTES': np.random.rand(500) * 2500,
#         'MAXBYTESDI': np.random.rand(500) * 100, 'RFCRECEIVE': np.random.randint(0, 10, 500),
#         'RFCSEND': np.random.randint(0, 10, 500), 'RFCEXETIME': np.random.rand(500) * 50,
#         'RFCCALLTIM': np.random.rand(500) * 50, 'RFCCALLS': np.random.randint(0, 10, 500),
#         'VMC_CALL_COUNT': np.random.randint(0, 5, 500), 'VMC_CPU_TIME': np.random.rand(500) * 10,
#         'VMC_ELAP_TIME': np.random.rand(500) * 10,
#         'ACCOUNT': np.random.choice(['USER_A', 'USER_B', 'USER_C', 'USER_D', 'Non défini'], 500),
#         'REPORT': np.random.choice(['REPORT_X', 'REPORT_Y', 'REPORT_Z', 'Non défini'], 500)
#     })
#     dfs = {"hitlist_db": pd.DataFrame(data_ml)}


st.header("🔮 Prédiction du Temps de Réponse (RESPTI)")
st.write("Cette section utilise un modèle de Machine Learning pour prédire le temps de réponse (RESPTI) des transactions SAP.")

df_hitlist_ml = dfs.get("hitlist_db", pd.DataFrame())

if df_hitlist_ml.empty or 'RESPTI' not in df_hitlist_ml.columns or 'FULL_DATETIME' not in df_hitlist_ml.columns:
    st.info("Données 'hitlist_db' non disponibles ou les colonnes nécessaires (RESPTI, FULL_DATETIME) sont manquantes. Impossible d'effectuer la prédiction.")
else:
    # Préparation des données pour le modèle ML
    df_ml_data = df_hitlist_ml.copy()

    # Convertir RESPTI en numérique et gérer les NaNs pour la cible
    df_ml_data['RESPTI'] = pd.to_numeric(df_ml_data['RESPTI'], errors='coerce')
    df_ml_data = df_ml_data.dropna(subset=['RESPTI', 'FULL_DATETIME']) # Supprimer les lignes avec NaN dans la cible ou la date

    if df_ml_data.empty or df_ml_data['RESPTI'].sum() == 0:
        st.info("Pas de données valides pour la prédiction de RESPTI après nettoyage.")
    else:
        # Ingénierie des caractéristiques temporelles
        df_ml_data['hour_of_day'] = df_ml_data['FULL_DATETIME'].dt.hour
        df_ml_data['day_of_week'] = df_ml_data['FULL_DATETIME'].dt.dayofweek # Lundi=0, Dimanche=6

        # Sélection des caractéristiques (features)
        numerical_features = [
            'CPUTI', 'PROCTI', 'QUEUETI', 'ROLLWAITTI', 'GUITIME', 'GUICNT', 'GUINETTIME',
            'DBP_COUNT', 'DBP_TIME', 'DSQLCNT', 'QUECNT', 'CPICCNT', 'SLI_CNT',
            'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQTI',
            'READSEQBUF', 'READSEQREC', 'INSCNT', 'INSTI', 'INSREC', 'PHYINSCNT',
            'UPDCNT', 'UPDTI', 'UPDREC', 'PHYUPDCNT', 'DELCNT', 'DELTI', 'DELREC', 'PHYDELCNT',
            'DBCALLS', 'COMMITTI', 'INPUTLEN', 'OUTPUTLEN', 'MAXROLL', 'MAXPAGE', 'ROLLINCNT',
            'ROLLINTI', 'ROLLOUTCNT', 'ROLLOUTTI', 'PRIVSUM', 'USEDBYTES', 'MAXBYTES', 'MAXBYTESDI',
            'RFCRECEIVE', 'RFCSEND', 'RFCEXETIME', 'RFCCALLTIM', 'RFCCALLS', 'VMC_CALL_COUNT',
            'VMC_CPU_TIME', 'VMC_ELAP_TIME'
        ]
        categorical_features = ['ACCOUNT', 'REPORT']

        # Assurez-vous que toutes les colonnes sélectionnées existent dans le DataFrame
        features_to_use = []
        for col in numerical_features:
            if col in df_ml_data.columns:
                df_ml_data[col] = pd.to_numeric(df_ml_data[col], errors='coerce').fillna(0)
                features_to_use.append(col)
        for col in categorical_features:
            if col in df_ml_data.columns:
                df_ml_data[col] = clean_string_column(df_ml_data[col])
                features_to_use.append(col)
        
        features_to_use.extend(['hour_of_day', 'day_of_week'])
        features_to_use = [f for f in features_to_use if f in df_ml_data.columns]
        
        X = df_ml_data[features_to_use]
        y = np.log1p(df_ml_data['RESPTI'].clip(lower=0)) # Transformation logarithmique de la cible

        if X.empty or y.empty or len(X) < 2:
            st.info("Pas de données suffisantes pour entraîner le modèle après la sélection des caractéristiques.")
        else:
            # Stocker les valeurs uniques des catégories pour les selectbox de prédiction
            if 'unique_accounts' not in st.session_state:
                st.session_state.unique_accounts = sorted(df_ml_data['ACCOUNT'].unique().tolist())
            if 'unique_reports' not in st.session_state:
                st.session_state.unique_reports = sorted(df_ml_data['REPORT'].unique().tolist())

            # Créer le préprocesseur et le pipeline (seulement si pas déjà en session)
            if 'model_pipeline' not in st.session_state:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean'))
                ])
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, [f for f in features_to_use if f in numerical_features]),
                        ('cat', categorical_transformer, [f for f in features_to_use if f in categorical_features])
                    ],
                    remainder='passthrough'
                )
                model_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                ])

                # Entraîner le modèle et le stocker
                with st.spinner("Entraînement du modèle de prédiction de temps de réponse..."):
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model_pipeline.fit(X_train, y_train)
                        st.session_state.model_pipeline = model_pipeline
                        st.session_state.features_to_use = features_to_use # Stocker aussi les features utilisées
                        st.session_state.numerical_features = numerical_features
                        st.session_state.categorical_features = categorical_features
                        st.success("Modèle entraîné avec succès !")

                        # Évaluation du modèle
                        y_pred = model_pipeline.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        y_pred_original_scale = np.expm1(y_pred)
                        y_test_original_scale = np.expm1(y_test)
                        mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)

                        st.subheader("Performance du Modèle")
                        st.write(f"**Score R² (Coefficient de Détermination) :** `{r2:.4f}`")
                        st.write(f"**MAE (Erreur Absolue Moyenne) :** `{mae:.2f} ms`")
                        st.info("Un R² proche de 1 indique un bon ajustement du modèle. Un MAE faible indique que les prédictions sont proches des valeurs réelles.")

                        # Afficher les importances des caractéristiques
                        st.subheader("Importance des Caractéristiques")
                        regressor = model_pipeline.named_steps['regressor']
                        processed_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
                        feature_importances = pd.DataFrame({
                            'Feature': processed_feature_names,
                            'Importance': regressor.feature_importances_
                        }).sort_values(by='Importance', ascending=False)
                        st.dataframe(feature_importances.head(10))

                        # Visualisation des prédictions vs. réel
                        st.subheader("Prédictions vs. Réel (Échantillon)")
                        df_results = pd.DataFrame({'Réel': y_test_original_scale, 'Prédit': y_pred_original_scale})
                        fig_pred_vs_real = px.scatter(df_results.sample(min(500, len(df_results)), random_state=42),
                                                    x='Réel', y='Prédit',
                                                    title='Prédictions du Modèle vs. Valeurs Réelles',
                                                    labels={'Réel': 'Temps de Réponse Réel (ms)', 'Prédit': 'Temps de Réponse Prédit (ms)'})
                        fig_pred_vs_real.add_trace(px.line(x=[min(y_test_original_scale), max(y_test_original_scale)], y=[min(y_test_original_scale), max(y_test_original_scale)], 
                                                        color_discrete_sequence=['red']).data[0])
                        st.plotly_chart(fig_pred_vs_real, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement ou de l'évaluation du modèle : {e}")
                        st.info("Vérifiez la qualité de vos données et le nombre de lignes disponibles.")
            else:
                st.info("Le modèle a déjà été entraîné. Vous pouvez faire des prédictions ci-dessous.")
                st.subheader("Performance du Modèle (déjà entraîné)")
                # Ré-afficher les métriques si le modèle est déjà entraîné
                # (Vous pourriez stocker r2 et mae en session_state aussi si vous voulez les ré-afficher sans re-calcul)
                st.write("Le modèle est prêt pour la prédiction.")

            # --- Interface de Prédiction Interactive ---
            if 'model_pipeline' in st.session_state:
                st.subheader("Faire une Prédiction")
                st.markdown("Entrez les valeurs des métriques pour obtenir une prédiction du temps de réponse (RESPTI).")

                # Créer des champs de saisie pour chaque caractéristique
                input_data = {}
                
                # Récupérer les moyennes pour les valeurs par défaut
                df_stats = df_ml_data[st.session_state.numerical_features].mean().to_dict()

                # Champs pour les caractéristiques numériques
                cols = st.columns(3)
                col_idx = 0
                for feature in st.session_state.numerical_features:
                    with cols[col_idx % 3]:
                        default_value = df_stats.get(feature, 0.0)
                        input_data[feature] = st.number_input(
                            f"{feature} (ms/count)",
                            value=float(default_value),
                            format="%.2f",
                            key=f"input_{feature}"
                        )
                    col_idx += 1

                # Champs pour les caractéristiques catégorielles
                cols = st.columns(2)
                col_idx = 0
                if 'ACCOUNT' in st.session_state.categorical_features and st.session_state.unique_accounts:
                    with cols[col_idx % 2]:
                        input_data['ACCOUNT'] = st.selectbox(
                            "Compte (ACCOUNT)",
                            options=st.session_state.unique_accounts,
                            key="input_account"
                        )
                    col_idx += 1
                
                if 'REPORT' in st.session_state.categorical_features and st.session_state.unique_reports:
                    with cols[col_idx % 2]:
                        input_data['REPORT'] = st.selectbox(
                            "Rapport (REPORT)",
                            options=st.session_state.unique_reports,
                            key="input_report"
                        )
                    col_idx += 1

                # Champs pour les caractéristiques temporelles
                cols = st.columns(2)
                with cols[0]:
                    input_data['hour_of_day'] = st.slider(
                        "Heure de la journée (0-23)",
                        min_value=0, max_value=23, value=12, step=1,
                        key="input_hour_of_day"
                    )
                with cols[1]:
                    input_data['day_of_week'] = st.slider(
                        "Jour de la semaine (0=Lundi, 6=Dimanche)",
                        min_value=0, max_value=6, value=0, step=1,
                        key="input_day_of_week"
                    )

                # Bouton de prédiction
                if st.button("Prédire le Temps de Réponse"):
                    try:
                        # Créer un DataFrame à partir des entrées utilisateur
                        # Assurez-vous que l'ordre des colonnes est le même que X_train
                        input_df = pd.DataFrame([input_data])
                        
                        # S'assurer que toutes les colonnes attendues par le modèle sont présentes,
                        # même si elles n'ont pas été affichées comme input (remplir avec 0 ou moyenne)
                        # C'est important pour le preprocessor.
                        for col in st.session_state.numerical_features:
                            if col not in input_df.columns:
                                input_df[col] = df_stats.get(col, 0.0) # Utiliser la moyenne du training set
                        for col in st.session_state.categorical_features:
                            if col not in input_df.columns:
                                input_df[col] = "Non défini" # Ou la catégorie la plus fréquente
                        
                        # S'assurer que l'ordre des colonnes est correct
                        input_df = input_df[st.session_state.features_to_use]

                        # Effectuer la prédiction
                        predicted_log_respti = st.session_state.model_pipeline.predict(input_df)
                        predicted_respti = np.expm1(predicted_log_respti[0]) # Inverse la transformation

                        st.success(f"Le temps de réponse (RESPTI) prédit est de : **{predicted_respti:.2f} ms**")

                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {e}")
                        st.info("Veuillez vérifier les valeurs saisies.")
            else:
                st.info("Le modèle n'est pas encore entraîné ou n'a pas pu être chargé. Veuillez recharger la page ou vérifier les données.")
