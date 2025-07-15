elif st.session_state.current_section == "Prédiction de Performance (ML)":
    st.header("🔮 Prédiction de Performance (ML)")
    st.markdown("""
        Utilisez cette section pour entraîner un modèle de Machine Learning et prédire une métrique de performance.
        Le modèle utilise les autres colonnes comme caractéristiques (features) pour apprendre à prédire la colonne cible.
        """)

    # Sélection de la source de données
    ml_source_data_key = st.selectbox(
        "Sélectionner la source de données pour la ML :",
        options=["hitlist_db", "memory", "times", "usertcode", "performance", "sql_trace_summary"],
        index=0 # Default to hitlist_db
    )

    df_ml = dfs[ml_source_data_key].copy()

    if df_ml.empty:
        st.warning(f"La source de données '{ml_source_data_key}' est vide ou n'a pas pu être chargée. Impossible de procéder à la prédiction.")
    else:
        st.subheader("Préparation des Données pour la Prédiction")

        # Sélection de la colonne cible pour la prédiction
        numeric_cols_for_target = df_ml.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols_for_target:
            st.error(f"La source de données '{ml_source_data_key}' ne contient aucune colonne numérique pour la prédiction. Veuillez choisir une autre source.")
        else:
            target_ml_column = st.selectbox(
                "Sélectionner la colonne cible à prédire (doit être numérique) :",
                options=numeric_cols_for_target
            )

            # Préparation des caractéristiques (features) et de la cible
            X = df_ml.drop(columns=[target_ml_column], errors='ignore')
            y = df_ml[target_ml_column]

            # Supprimer les colonnes avec trop de valeurs manquantes ou non numériques pour X
            # (Exclure la colonne cible qui est déjà gérée)
            initial_columns = X.columns.tolist()
            X = X.dropna(axis=1, how='all') # Supprime les colonnes entièrement NaN
            X = X.select_dtypes(exclude=['datetime64[ns]']) # Exclure les colonnes datetime qui ne sont pas directement utilisables

            # Supprimer les colonnes avec une seule valeur unique (variance nulle)
            columns_to_drop_single_value = [col for col in X.columns if X[col].nunique() == 1]
            if columns_to_drop_single_value:
                X = X.drop(columns=columns_to_drop_single_value)
                st.info(f"Colonnes supprimées car elles ne contenaient qu'une seule valeur unique : {', '.join(columns_to_drop_single_value)}")

            # S'assurer que les jeux de données ne sont pas vides après le nettoyage
            if X.empty or y.empty:
                st.error("Les données nettoyées pour la prédiction sont vides. Veuillez vérifier votre sélection ou vos données brutes.")
            else:
                st.write(f"Nombre de caractéristiques utilisées : {X.shape[1]}")
                st.write(f"Premières lignes des caractéristiques (X) :")
                st.dataframe(X.head())
                st.write(f"Premières lignes de la cible (y) :")
                st.dataframe(y.head())

                # Division des données en ensembles d'entraînement et de test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                st.write(f"Taille de l'ensemble d'entraînement : {len(X_train)} échantillons")
                st.write(f"Taille de l'ensemble de test : {len(X_test)} échantillons")

                st.subheader("Entraînement et Évaluation du Modèle")

                try:
                    # Identifier les colonnes numériques et catégorielles après le nettoyage
                    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
                    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

                    # Créer les pipelines de prétraitement
                    numerical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ])

                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])

                    # Combiner les transformateurs avec ColumnTransformer
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numerical_transformer, numerical_features),
                            ('cat', categorical_transformer, categorical_features)
                        ],
                        remainder='passthrough' # Garde les colonnes non spécifiées
                    )

                    # Définir le modèle
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                    # Créer le pipeline complet
                    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                     ('regressor', model)])

                    # FIT THE MODEL PIPELINE ON THE TRAINING DATA
                    # C'EST LA LIGNE CRUCIALE À AJOUTER OU VÉRIFIER
                    model_pipeline.fit(X_train, y_train)

                    # Faire des prédictions
                    y_pred = model_pipeline.predict(X_test)

                    # Évaluation du modèle
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    st.write(f"**R² Score :** {r2:.4f}")
                    st.write(f"**Erreur Absolue Moyenne (MAE) :** {mae:.4f}")

                    st.markdown("""
                        * **R² Score :** Indique la proportion de la variance de la variable dépendante qui est prédictible à partir des variables indépendantes. Un R² de 1.0 indique un ajustement parfait.
                        * **MAE (Mean Absolute Error) :** Représente la moyenne des erreurs absolues entre les prédictions et les valeurs réelles. Une MAE plus faible indique un meilleur modèle.
                        """)

                    st.subheader("Importance des Caractéristiques")
                    # L'importance des features est obtenue à partir du modèle RFR qui est l'étape 'regressor' du pipeline
                    # et qui a été entraîné après le prétraitement.
                    if hasattr(model_pipeline.named_steps['regressor'], 'feature_importances_'):
                        # Récupérer les noms des colonnes après prétraitement
                        preprocessor_fitted = model_pipeline.named_steps['preprocessor']
                        
                        # Noms des features numériques
                        numeric_feature_names = numerical_features
                        
                        # Noms des features catégorielles après OneHotEncoding
                        # Assurez-vous que le OneHotEncoder est accessible et fit
                        try:
                            ohe_feature_names = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                        except Exception as e:
                            st.warning(f"Impossible de récupérer les noms de features OneHotEncoded : {e}")
                            ohe_feature_names = [] # Fallback if get_feature_names_out fails


                        # Concaténer tous les noms de features transformées
                        all_feature_names = numerical_features + list(ohe_feature_names)
                        
                        # Gérer les colonnes 'passthrough' si elles existent et ne sont pas numériques/catégoriques
                        # La logique 'remainder'='passthrough' est dans ColumnTransformer, mais ses noms de colonnes
                        # ne sont pas directement accessibles via get_feature_names_out() sans une complexité supplémentaire.
                        # Pour ce cas, nous allons nous fier aux features_importances_ qui correspondent à l'ordre de traitement.

                        importances = model_pipeline.named_steps['regressor'].feature_importances_

                        # Créer un DataFrame pour l'importance des caractéristiques
                        # Vérifier que le nombre d'importances correspond au nombre de features
                        if len(importances) == len(all_feature_names):
                            feature_importance_df = pd.DataFrame({
                                'Feature': all_feature_names,
                                'Importance': importances
                            }).sort_values(by='Importance', ascending=False)
                            st.dataframe(feature_importance_df)

                            fig_importance = px.bar(feature_importance_df.head(10),
                                                    x='Importance', y='Feature', orientation='h',
                                                    title="Top 10 Importance des Caractéristiques",
                                                    labels={'Importance': 'Score d\'Importance', 'Feature': 'Caractéristique'},
                                                    color='Importance', color_continuous_scale=px.colors.sequential.Greens)
                            st.plotly_chart(fig_importance, use_container_width=True)
                        else:
                            st.warning(f"Le nombre d'importances ({len(importances)}) ne correspond pas au nombre de caractéristiques ({len(all_feature_names)}) après prétraitement. L'affichage de l'importance des caractéristiques peut être incorrect.")
                            st.info("Cela peut arriver si des colonnes ont été supprimées par le préprocesseur ou si des types de colonnes inattendus sont présents.")
                            # Fallback: display importance with generic names if mismatch occurs
                            feature_importance_df = pd.DataFrame({
                                'Feature': [f'Feature_{i}' for i in range(len(importances))],
                                'Importance': importances
                            }).sort_values(by='Importance', ascending=False)
                            st.dataframe(feature_importance_df.head(10))
                    else:
                        st.info("Le modèle ne supporte pas l'accès direct à l'importance des caractéristiques.")

                    st.subheader("Visualisation des Prédictions")
                    df_results = pd.DataFrame({'Valeurs Réelles': y_test, 'Prédictions': y_pred})
                    fig_pred = px.scatter(df_results.sample(min(500, len(df_results)), random_state=42), # Échantillon pour la visibilité
                                        x='Valeurs Réelles', y='Prédictions',
                                        title=f"Prédictions vs. Valeurs Réelles pour {target_ml_column}",
                                        labels={'Valeurs Réelles': f'Valeurs Réelles de {target_ml_column}', 'Prédictions': f'Prédictions de {target_ml_column}'},
                                        trendline='ols', # Ligne de régression linéaire
                                        opacity=0.6)
                    fig_pred.update_traces(marker_size=5)
                    st.plotly_chart(fig_pred, use_container_width=True)

                    st.write("""
                        Un bon modèle aura des points groupés près de la ligne diagonale (où les prédictions égalent les valeurs réelles).
                        """)
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement ou de l'évaluation du modèle : {e}")

# --- Téléchargement du script ---
st.sidebar.markdown("---")
st.sidebar.subheader("Télécharger le Script")
script_code = io.StringIO()
with open(__file__, "r", encoding="utf-8") as f:
    script_code.write(f.read())
st.sidebar.download_button(
    label="Télécharger TTEST_dashboard_sap.py",
    data=script_code.getvalue(),
    file_name="TTEST_dashboard_sap.py",
    mime="text/x-python"
)
