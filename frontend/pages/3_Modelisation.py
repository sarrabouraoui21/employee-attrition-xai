
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import api_get, api_post


GREEN = "#16665C"
ORANGE = "#C46C54"

MODELS = {
    "logistic_regression": "Régression logistique",
    "decision_tree": "Arbre de décision",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}

METRIC_LABELS = {
    "average_precision": "PR-AUC",
    "f1": "F1-score",
    "recall": "Recall",
    "roc_auc": "ROC-AUC"
}


def build_results_table(results):
    rows = []

    for result in results:
        metrics = result["metrics"]

        rows.append({
            "Modèle": result["display_name"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "Accuracy": metrics["accuracy"],
            "Durée (s)": result["duration_seconds"]
        })

    return pd.DataFrame(rows)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="eyebrow">MODULE 03 / MACHINE LEARNING</div>',
    unsafe_allow_html=True
)

st.title("Entraînement des modèles")

st.markdown(
    '<div class="page-description">'
    "Configurez les modèles, optimisez leurs hyperparamètres "
    "et comparez leurs performances sur le jeu de test."
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# PREPROCESSING CONFIGURATION
# =====================================================

dataset_id = (
    st.session_state.get("active_dataset_id")
    or "kaggle"
)

prep_config = st.session_state.get(
    "preprocessing_config"
)

prep_dataset = st.session_state.get(
    "prep_dataset_id"
)

if prep_config is None or prep_dataset != dataset_id:
    st.warning(
        "Vous devez d'abord appliquer le preprocessing "
        "sur le dataset actif."
    )

    if st.button(
        "Ouvrir la préparation des données",
        type="primary"
    ):
        st.switch_page("pages/2_Preparation.py")

    st.stop()


features = prep_config.get("features") or []

with st.container(border=True):
    st.markdown("#### Configuration des données")

    st.caption(
        f"Dataset : {'Kaggle' if dataset_id == 'kaggle' else 'CSV importé'}"
        f"  ·  {len(features)} variables sélectionnées"
        f"  ·  Test : {prep_config['test_size']:.0%}"
    )

    st.caption(
        "Encodage One-Hot · Imputation "
        f"{prep_config['numeric_imputation']} · "
        f"StandardScaler : "
        f"{'Oui' if prep_config['scale_numeric'] else 'Non'}"
    )

    with st.expander("Voir les variables sélectionnées"):
        st.write(", ".join(features))


# =====================================================
# MODEL CONFIGURATION
# =====================================================

st.write("")

with st.container(border=True):
    st.markdown("### Configuration des modèles")

    with st.form("training_form"):

        selected = st.multiselect(
            "Modèles à entraîner",
            options=list(MODELS.keys()),
            default=list(MODELS.keys()),
            format_func=lambda name: MODELS[name]
        )

        st.divider()

        tune = st.toggle(
            "Optimiser les hyperparamètres",
            value=True,
            help=(
                "Recherche sur plusieurs configurations "
                "avec validation croisée stratifiée."
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            scoring = st.selectbox(
                "Métrique d'optimisation",
                options=list(METRIC_LABELS.keys()),
                format_func=lambda x: METRIC_LABELS[x],
                index=0
            )

            cv_folds = st.selectbox(
                "Nombre de folds",
                [2, 3, 5],
                index=1
            )

        with col2:
            search_iterations = st.slider(
                "Configurations testées par modèle",
                min_value=1,
                max_value=10,
                value=3
            )

        with st.expander("Hyperparamètres des modèles"):
            st.caption(
                "Ces valeurs servent de paramètres de base "
                "ou de référence pour la recherche."
            )

            st.markdown("**Régression logistique**")

            logistic_c = st.number_input(
                "C — Régularisation",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.1
            )

            st.markdown("**Arbre de décision**")

            a, b = st.columns(2)

            tree_depth = a.number_input(
                "Profondeur arbre",
                min_value=2,
                max_value=30,
                value=6
            )

            tree_leaf = b.number_input(
                "Min. observations par feuille",
                min_value=1,
                max_value=30,
                value=2
            )

            st.markdown("**KNN**")

            knn_neighbors = st.number_input(
                "Nombre de voisins",
                min_value=1,
                max_value=100,
                value=9
            )

            st.markdown("**Random Forest**")

            a, b = st.columns(2)

            rf_trees = a.number_input(
                "Nombre d'arbres RF",
                min_value=10,
                max_value=500,
                value=120,
                step=10
            )

            rf_depth = b.number_input(
                "Profondeur RF",
                min_value=2,
                max_value=30,
                value=10
            )

            st.markdown("**XGBoost**")

            a, b, c = st.columns(3)

            xgb_trees = a.number_input(
                "Nombre d'arbres XGB",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )

            xgb_depth = b.number_input(
                "Profondeur XGB",
                min_value=2,
                max_value=15,
                value=4
            )

            xgb_lr = c.number_input(
                "Learning rate",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01
            )

        submitted = st.form_submit_button(
            "Lancer l'entraînement",
            type="primary",
            use_container_width=True
        )


# =====================================================
# START TRAINING
# =====================================================

if submitted:
    if not selected:
        st.error("Sélectionnez au moins un modèle.")
        st.stop()

    payload = {
        "preprocessing": prep_config,
        "models": selected,
        "tune_hyperparameters": tune,
        "cv_folds": cv_folds,
        "search_iterations": search_iterations,
        "scoring": scoring,
        "hyperparameters": {
            "logistic_c": logistic_c,
            "tree_max_depth": tree_depth,
            "tree_min_samples_leaf": tree_leaf,
            "knn_neighbors": knn_neighbors,
            "rf_n_estimators": rf_trees,
            "rf_max_depth": rf_depth,
            "xgb_n_estimators": xgb_trees,
            "xgb_max_depth": xgb_depth,
            "xgb_learning_rate": xgb_lr
        }
    }

    try:
        response = api_post("/train/start", payload)

        st.session_state["training_job_id"] = (
            response["job_id"]
        )
        st.session_state["training_dataset_id"] = dataset_id

    except Exception as error:
        st.error(f"Impossible de lancer : {error}")
        st.stop()


# =====================================================
# LIVE TRAINING STATUS
# =====================================================

job_id = st.session_state.get("training_job_id")
job_dataset = st.session_state.get(
    "training_dataset_id"
)

if not job_id or job_dataset != dataset_id:
    st.info(
        "Sélectionnez les modèles puis lancez "
        "l'entraînement pour voir les résultats."
    )
    st.stop()


st.write("")
st.markdown("### Suivi de l'entraînement")

progress_area = st.empty()
status_area = st.empty()
results_area = st.empty()

started_polling = time.monotonic()
status = None

while True:
    try:
        status = api_get(f"/train/status/{job_id}")

    except Exception as error:
        st.error(f"Suivi indisponible : {error}")
        st.stop()

    progress_area.progress(
        status["progress"],
        text=(
            f"{status['progress']} % — "
            f"{status['stage']}"
        )
    )

    completed = len(status["results"])
    total = status["total_models"]

    status_area.caption(
        f"{completed} modèle(s) terminé(s) sur {total} "
        f"· Job : {job_id[:12]}"
    )

    if status["results"]:
        partial_df = build_results_table(
            status["results"]
        )

        results_area.dataframe(
            partial_df,
            use_container_width=True,
            hide_index=True
        )

    if status["state"] in {"completed", "failed"}:
        break

    # Évite une boucle sans fin côté navigateur.
    if time.monotonic() - started_polling > 600:
        st.info(
            "Le calcul continue côté backend. "
            "Actualisez cette page pour reprendre le suivi."
        )
        st.stop()

    time.sleep(1.5)


# =====================================================
# COMPLETED / FAILED
# =====================================================

if status["state"] == "failed":
    st.error(
        status.get("error")
        or "L'entraînement a échoué."
    )
    st.caption(
        "Consultez le terminal FastAPI pour "
        "le traceback détaillé."
    )
    st.stop()

st.success("Entraînement terminé.")

results = status["results"]

if not results:
    st.warning("Aucun résultat disponible.")
    st.stop()

df = build_results_table(results)

st.divider()
st.markdown("### Comparaison des modèles")

st.dataframe(
    df.sort_values("PR-AUC", ascending=False),
    use_container_width=True,
    hide_index=True
)

chart_df = df.melt(
    id_vars="Modèle",
    value_vars=[
        "Precision", "Recall", "F1",
        "ROC-AUC", "PR-AUC"
    ],
    var_name="Métrique",
    value_name="Score"
)

fig = px.bar(
    chart_df,
    x="Modèle",
    y="Score",
    color="Métrique",
    barmode="group",
    color_discrete_sequence=[
        GREEN,
        "#91ACA3",
        "#D0A86E",
        ORANGE,
        "#627C9B"
    ]
)

fig.update_layout(
    template="plotly_white",
    height=430,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    yaxis=dict(range=[0, 1], title=None),
    xaxis=dict(title=None),
    margin=dict(l=10, r=10, t=20, b=60),
    legend_title_text=""
)

st.plotly_chart(
    fig,
    use_container_width=True
)

best = max(
    results,
    key=lambda result: result["metrics"]["pr_auc"]
)

with st.container(border=True):
    st.markdown("#### Meilleur score PR-AUC observé")

    a, b, c = st.columns(3)

    a.metric("Modèle", best["display_name"])
    b.metric("PR-AUC", f"{best['metrics']['pr_auc']:.3f}")
    c.metric("Recall", f"{best['metrics']['recall']:.3f}")

    st.caption(
        "Classement exploratoire sur le test, "
        "sans validation indépendante supplémentaire."
    )


st.markdown("### Hyperparamètres retenus")

for result in results:
    with st.expander(result["display_name"]):
        if result["best_params"]:
            st.json(result["best_params"])
        else:
            st.caption(
                "Paramètres manuels, sans recherche CV."
            )

        if result["cv_score"] is not None:
            st.caption(
                "Score en validation croisée : "
                f"{result['cv_score']:.4f}"
            )

        st.caption(
            "Temps d'entraînement : "
            f"{result['duration_seconds']} secondes"
        )

st.caption(
    "Les pipelines entraînés sont enregistrés "
    "dans backend/artifacts au format .joblib."
)
