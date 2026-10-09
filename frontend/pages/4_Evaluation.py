
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import api_get, api_post


COLORS = [
    "#16665C",
    "#C46C54",
    "#627C9B",
    "#B89C55",
    "#876E9A"
]

MODEL_NAMES = {
    "logistic_regression": "Régression logistique",
    "decision_tree": "Arbre de décision",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}


# =====================================================
# OUTILS
# =====================================================

def style_figure(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Segoe UI, Arial",
            size=12,
            color="#56645E"
        ),
        margin=dict(l=50, r=25, t=30, b=55),
        legend=dict(
            orientation="h",
            y=-0.25,
            x=0
        ),
        hovermode="closest"
    )

    fig.update_xaxes(
        gridcolor="#EDF0EE",
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#EDF0EE",
        zeroline=False
    )

    return fig


@st.cache_data(ttl=600, show_spinner=False)
def get_report(job_id, config_json):
    return api_post(
        "/evaluate/report",
        {
            "job_id": job_id,
            "preprocessing": json.loads(config_json)
        }
    )


def build_comparison_table(results):
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
            "Accuracy": metrics["accuracy"]
        })

    return pd.DataFrame(rows)


# =====================================================
# NOMMAGE DES ENTRAÎNEMENTS
# =====================================================

def build_training_labels(jobs):
    """
    Attribue un nom lisible à chaque entraînement.

    Exemple :
    Entraînement n°2 — 09/10/2026 à 10:35 — 5 modèles
    """

    chronological = sorted(
        jobs,
        key=lambda job: (
            job["modified_at"],
            job["job_id"]
        )
    )

    labels = {}

    for number, job in enumerate(
        chronological, start=1
    ):
        job_id = job["job_id"]

        date = datetime.fromtimestamp(
            job["modified_at"]
        ).strftime("%d/%m/%Y à %H:%M")

        count = len(job["models"])

        model_text = (
            f"{count} modèle"
            if count == 1
            else f"{count} modèles"
        )

        labels[job_id] = (
            f"Entraînement n°{number} — "
            f"{date} — {model_text}"
        )

    return labels


# =====================================================
# COURBES ROC / PR
# =====================================================

def make_curves(results, curve_type, baseline=None):
    fig = go.Figure()

    for index, result in enumerate(results):
        points = result[curve_type]

        fig.add_trace(
            go.Scatter(
                x=[p["x"] for p in points],
                y=[p["y"] for p in points],
                mode="lines",
                name=result["display_name"],
                line=dict(
                    color=COLORS[index % len(COLORS)],
                    width=2.5
                )
            )
        )

    if curve_type == "roc_curve":
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Aléatoire",
                mode="lines",
                line=dict(
                    color="#B8C1BD",
                    dash="dash",
                    width=1.5
                )
            )
        )

        x_label = "Taux de faux positifs (FPR)"
        y_label = "Taux de vrais positifs (Recall)"

    else:
        if baseline is not None:
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="#B8C1BD",
                annotation_text="Prévalence"
            )

        x_label = "Recall"
        y_label = "Precision"

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02]
    )

    return style_figure(fig, 440)


# =====================================================
# MATRICE DE CONFUSION
# =====================================================

def make_confusion_matrix(metrics):
    cm = metrics["confusion_matrix"]

    matrix = [
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[
                "Prédit : classe 0",
                "Prédit : classe 1"
            ],
            y=[
                "Réel : classe 0",
                "Réel : classe 1"
            ],
            colorscale=[
                [0.0, "#EEF3F0"],
                [1.0, "#16665C"]
            ],
            showscale=False,
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=19),
            hovertemplate=(
                "Effectif : %{z}<extra></extra>"
            )
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=30)
    )

    return style_figure(fig, 340)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="eyebrow">MODULE 04 / ÉVALUATION</div>',
    unsafe_allow_html=True
)

st.title("Évaluation des modèles")

st.markdown(
    '<div class="page-description">'
    "Comparez les performances et analysez "
    "l'impact du seuil de décision sur les erreurs "
    "de classification."
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# CONFIGURATION DU DATASET
# =====================================================

prep_config = st.session_state.get(
    "preprocessing_config"
)

active_dataset = (
    st.session_state.get("active_dataset_id")
    or "kaggle"
)

if prep_config is None:
    st.warning(
        "Appliquez d'abord le preprocessing "
        "pour retrouver la configuration du test."
    )
    st.stop()

configured_dataset = (
    prep_config.get("dataset_id") or "kaggle"
)

if configured_dataset != active_dataset:
    st.error(
        "Le dataset actif ne correspond pas "
        "à la configuration du preprocessing."
    )
    st.stop()


# =====================================================
# CHARGEMENT DES ENTRAÎNEMENTS
# =====================================================

try:
    available_jobs = api_get(
        "/evaluate/jobs"
    )["jobs"]

except Exception as error:
    st.error(f"API inaccessible : {error}")
    st.stop()

if not available_jobs:
    st.warning(
        "Aucun modèle sauvegardé dans backend/artifacts."
    )
    st.stop()


# =====================================================
# NOMMAGE LISIBLE DES EXÉCUTIONS
# =====================================================

job_labels = build_training_labels(
    available_jobs
)

jobs_by_id = {
    item["job_id"]: item
    for item in available_jobs
}

job_ids = [
    item["job_id"]
    for item in sorted(
        available_jobs,
        key=lambda job: job["modified_at"],
        reverse=True
    )
]

current_job = st.session_state.get(
    "training_job_id"
)

default_index = (
    job_ids.index(current_job)
    if current_job in job_ids
    else 0
)

with st.container(border=True):
    st.markdown("### Entraînement à évaluer")

    selected_job = st.selectbox(
        "Sélectionnez un entraînement",
        options=job_ids,
        index=default_index,
        format_func=lambda job_id: job_labels[job_id]
    )

    st.caption(
        "Les modèles sont chargés depuis les fichiers "
        ".joblib. Aucun nouvel entraînement."
    )


# =====================================================
# RAPPORT D'ÉVALUATION
# =====================================================

config_json = json.dumps(
    prep_config,
    sort_keys=True
)

try:
    with st.spinner(
        "Chargement des métriques et courbes..."
    ):
        report = get_report(
            selected_job,
            config_json
        )

except Exception as error:
    st.error(
        f"Impossible d'évaluer cet entraînement : {error}"
    )

    st.info(
        "Vérifiez que la configuration du preprocessing "
        "correspond à celle utilisée lors "
        "de l'entraînement."
    )
    st.stop()


results = report["results"]

if not results:
    st.warning("Aucun résultat disponible.")
    st.stop()

df = build_comparison_table(results)

best = max(
    results,
    key=lambda x: x["metrics"]["pr_auc"]
)


# =====================================================
# INDICATEURS
# =====================================================

st.write("")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Observations test",
    report["test_size"]
)

c2.metric(
    "Classe positive",
    report["positive_count"]
)

c3.metric(
    "Modèles évalués",
    len(results)
)

c4.metric(
    "Meilleur PR-AUC",
    f"{best['metrics']['pr_auc']:.3f}"
)

st.caption(
    f"Variable cible : {report['target']} "
    f"· Classe positive : {report['positive_class']} "
    "· Seuil initial : 0,50"
)

st.write("")

tabs = st.tabs([
    "01 · Comparaison",
    "02 · Courbes ROC / PR",
    "03 · Seuil & confusion",
    "04 · Analyse métier"
])


# =====================================================
# ONGLET 1 — COMPARAISON
# =====================================================

with tabs[0]:
    st.write("")
    st.subheader("Comparaison des performances")

    st.dataframe(
        df.sort_values(
            "PR-AUC",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    metric_name = st.selectbox(
        "Métrique à comparer",
        [
            "PR-AUC",
            "Recall",
            "Precision",
            "F1",
            "ROC-AUC"
        ]
    )

    comparison = df.sort_values(
        metric_name,
        ascending=False
    )

    fig = go.Figure(
        go.Bar(
            x=comparison[metric_name],
            y=comparison["Modèle"],
            orientation="h",
            marker_color="#16665C",
            text=comparison[metric_name].round(3),
            textposition="outside"
        )
    )

    fig.update_layout(
        xaxis_range=[0, 1.08],
        yaxis=dict(autorange="reversed"),
        xaxis_title=metric_name,
        showlegend=False
    )

    st.plotly_chart(
        style_figure(fig, 350),
        use_container_width=True
    )

    st.caption(
        "L'Accuracy reste indicative pour un "
        "dataset déséquilibré. Recall, F1 et "
        "PR-AUC sont privilégiés."
    )


# =====================================================
# ONGLET 2 — COURBES
# =====================================================

with tabs[1]:
    st.write("")
    st.subheader("Courbes ROC superposées")

    st.plotly_chart(
        make_curves(
            results,
            "roc_curve"
        ),
        use_container_width=True
    )

    st.caption(
        "Compromis entre vrais positifs "
        "et faux positifs selon le seuil."
    )

    st.divider()

    st.subheader("Courbes Precision-Recall")

    st.plotly_chart(
        make_curves(
            results,
            "pr_curve",
            baseline=report["positive_rate"]
        ),
        use_container_width=True
    )

    st.caption(
        "Ces courbes sont particulièrement utiles "
        "pour les classifications déséquilibrées."
    )


# =====================================================
# ONGLET 3 — SEUIL ET CONFUSION
# =====================================================

with tabs[2]:
    st.write("")
    st.subheader("Analyse du seuil de décision")

    result_by_name = {
        result["model"]: result
        for result in results
    }

    selected_model = st.selectbox(
        "Modèle à analyser",
        options=list(result_by_name.keys()),
        format_func=lambda name: MODEL_NAMES[name]
    )

    threshold = st.slider(
        "Seuil de classification",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01
    )

    try:
        threshold_report = api_post(
            "/evaluate/threshold",
            {
                "job_id": selected_job,
                "preprocessing": prep_config,
                "model": selected_model,
                "threshold": threshold
            }
        )

    except Exception as error:
        st.error(f"Erreur seuil : {error}")
        st.stop()

    metrics = threshold_report["metrics"]
    cm = metrics["confusion_matrix"]

    st.write("")

    a, b, c, d = st.columns(4)

    a.metric(
        "Precision",
        f"{metrics['precision']:.3f}"
    )
    b.metric(
        "Recall",
        f"{metrics['recall']:.3f}"
    )
    c.metric(
        "F1-score",
        f"{metrics['f1']:.3f}"
    )
    d.metric(
        "Prédictions positives",
        metrics["predicted_positives"]
    )

    st.divider()
    st.subheader("Matrice de confusion")

    left, right = st.columns(
        [1.5, 1],
        gap="large"
    )

    with left:
        st.plotly_chart(
            make_confusion_matrix(metrics),
            use_container_width=True
        )

    with right:
        st.markdown("#### Détail des erreurs")

        st.metric("Vrais positifs (TP)", cm["tp"])
        st.metric("Vrais négatifs (TN)", cm["tn"])
        st.metric("Faux positifs (FP)", cm["fp"])
        st.metric("Faux négatifs (FN)", cm["fn"])

    st.caption(
        "FP : fausse alerte de départ. "
        "FN : départ réel non détecté."
    )

    if threshold < 0.50:
        st.info(
            "Un seuil plus bas augmente généralement "
            "le rappel, mais peut augmenter les FP."
        )
    elif threshold > 0.50:
        st.info(
            "Un seuil plus élevé diminue généralement "
            "les FP, mais peut augmenter les FN."
        )
    else:
        st.info(
            "Seuil standard de 0,50. Déplacez le "
            "curseur pour observer les variations."
        )

    st.warning(
        "Le seuil définitif doit être choisi sur "
        "un ensemble de validation distinct, "
        "et non optimisé sur le test."
    )


# =====================================================
# ONGLET 4 — ANALYSE MÉTIER
# =====================================================

with tabs[3]:
    st.write("")
    st.subheader("Interprétation métier")

    st.markdown(
        """
        Pour notre problème de prédiction du départ
        des employés :

        - **Faux négatif (FN)** : un départ réel
          que le modèle n'a pas détecté.
        - **Faux positif (FP)** : un employé signalé
          comme susceptible de partir alors qu'il reste.
        """
    )

    st.divider()

    st.markdown("#### Objectif métier")

    objective = st.radio(
        "Quelle erreur souhaitez-vous privilégier ?",
        [
            "Détecter davantage de départs",
            "Limiter les fausses alertes",
            "Équilibrer les deux objectifs"
        ]
    )

    if objective == "Détecter davantage de départs":
        st.markdown(
            "**Métrique à privilégier : Recall**"
        )
        st.write(
            "On cherche à réduire les faux négatifs."
        )

    elif objective == "Limiter les fausses alertes":
        st.markdown(
            "**Métrique à privilégier : Precision**"
        )
        st.write(
            "On cherche à réduire les alertes "
            "injustifiées parmi les prédictions positives."
        )

    else:
        st.markdown(
            "**Métrique à privilégier : F1-score**"
        )
        st.write(
            "On recherche un compromis entre "
            "Precision et Recall."
        )

    st.divider()

    st.markdown(
        "#### Modèle le mieux classé en PR-AUC"
    )

    st.markdown(
        f"**{best['display_name']}**"
    )

    x, y, z = st.columns(3)

    x.metric(
        "PR-AUC",
        f"{best['metrics']['pr_auc']:.3f}"
    )

    y.metric(
        "Recall",
        f"{best['metrics']['recall']:.3f}"
    )

    z.metric(
        "F1-score",
        f"{best['metrics']['f1']:.3f}"
    )

    st.caption(
        "Classement descriptif calculé sur "
        "le jeu de test."
    )

    st.warning(
        "Le dataset IBM HR est synthétique. "
        "Les prédictions ne doivent pas servir "
        "à automatiser des décisions RH réelles."
    )
