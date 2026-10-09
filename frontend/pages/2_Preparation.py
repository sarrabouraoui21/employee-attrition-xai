
import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import api_get, api_post


GREEN = "#16665C"
ORANGE = "#C46C54"
TEXT = "#56645E"


# =====================================================
# DESIGN DES GRAPHIQUES
# =====================================================

def plot_style(fig, height=330):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Segoe UI, Arial",
            size=12,
            color=TEXT
        ),
        margin=dict(l=10, r=10, t=30, b=35),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(gridcolor="#EDF0EE", title=None),
        legend_title_text=""
    )
    return fig


def draw_bar(data, title, color=GREEN):
    if not data:
        st.info("Aucune donnée à afficher.")
        return

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="label",
        y="count",
        title=title,
        color_discrete_sequence=[color]
    )

    fig.update_traces(marker_line_width=0)

    st.plotly_chart(
        plot_style(fig),
        use_container_width=True,
        config={"displayModeBar": False}
    )


# =====================================================
# EN-TÊTE
# =====================================================

st.markdown(
    '<div class="eyebrow">MODULE 02 / PREPROCESSING</div>',
    unsafe_allow_html=True
)

st.title("Préparation des données")

st.markdown(
    '<div class="page-description">'
    "Configurez votre pipeline et observez les "
    "transformations appliquées aux données."
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# DATASET ACTIF
# =====================================================

dataset_id = (
    st.session_state.get("active_dataset_id")
    or "kaggle"
)

# Éviter de réutiliser le rapport d'un autre dataset.
if st.session_state.get("prep_dataset_id") != dataset_id:
    st.session_state.pop("prep_report", None)
    st.session_state.pop("preprocessing_config", None)

try:
    options = api_get(
        "/preprocessing/options",
        params={"dataset_id": dataset_id}
    )

except Exception as error:
    st.error(f"Impossible de charger les options : {error}")
    st.stop()


st.caption(
    f"Dataset : {dataset_id if dataset_id == 'kaggle' else 'CSV importé'}"
    f"  ·  Cible : {options['target']}"
    f"  ·  Classe positive : {options['positive_class']}"
)


# =====================================================
# CONFIGURATION
# =====================================================

with st.container(border=True):
    st.markdown("### Configuration du pipeline")

    with st.form("preprocessing_form"):
        features = st.multiselect(
            "Variables explicatives",
            options=options["available_features"],
            default=options["available_features"],
            help="La variable cible est automatiquement exclue."
        )

        st.divider()

        c1, c2 = st.columns(2, gap="large")

        with c1:
            test_size = st.slider(
                "Proportion du test",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05
            )

            imputation = st.selectbox(
                "Imputation numérique",
                ["median", "mean"],
                format_func=lambda x: {
                    "median": "Médiane",
                    "mean": "Moyenne"
                }[x]
            )

        with c2:
            random_state = st.number_input(
                "Random state",
                min_value=0,
                value=42,
                step=1
            )

            scaling = st.toggle(
                "Normalisation StandardScaler",
                value=True
            )

        st.divider()

        st.markdown("#### Variables pour l'analyse avant / après")

        c3, c4 = st.columns(2)

        with c3:
            numeric_options = options["numeric_features"]

            inspect_numeric = (
                st.selectbox(
                    "Variable numérique",
                    numeric_options,
                    index=0
                )
                if numeric_options else None
            )

        with c4:
            categorical_options = options["categorical_features"]

            inspect_categorical = (
                st.selectbox(
                    "Variable catégorielle",
                    categorical_options,
                    index=0
                )
                if categorical_options else None
            )

        submitted = st.form_submit_button(
            "Appliquer le preprocessing",
            type="primary",
            use_container_width=True
        )


# =====================================================
# APPEL FASTAPI
# =====================================================

if submitted:
    if not features:
        st.error("Sélectionnez au moins une variable.")
        st.stop()

    config = {
        "dataset_id": dataset_id,
        "features": features,
        "test_size": test_size,
        "random_state": int(random_state),
        "numeric_imputation": imputation,
        "scale_numeric": scaling,
        "inspect_numeric": inspect_numeric,
        "inspect_categorical": inspect_categorical
    }

    try:
        with st.spinner("Préparation des données..."):
            report = api_post(
                "/preprocessing/preview",
                config
            )

        st.session_state["prep_report"] = report
        st.session_state["preprocessing_config"] = config
        st.session_state["prep_dataset_id"] = dataset_id

    except Exception as error:
        st.error(f"Erreur preprocessing : {error}")
        st.stop()


report = st.session_state.get("prep_report")

if report is None:
    st.info(
        "Configurez le pipeline et cliquez sur "
        "« Appliquer le preprocessing »."
    )
    st.stop()


# =====================================================
# RÉSULTATS GÉNÉRAUX
# =====================================================

dims = report["dimensions"]

st.write("")
st.markdown("### Résultats de la préparation")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Train", dims["train_rows"])
c2.metric("Test", dims["test_rows"])
c3.metric("Features initiales", dims["input_features"])
c4.metric("Features finales", dims["output_features"])

st.write("")

tabs = st.tabs([
    "01 · Train/Test",
    "02 · Imputation",
    "03 · Encodage",
    "04 · Normalisation",
    "05 · Résultat final"
])


# =====================================================
# 01 — TRAIN / TEST
# =====================================================

with tabs[0]:
    st.write("")
    st.subheader("Distribution des classes après split")

    split = report["split"]

    rows = []

    for key, label in [
        ("train", "Entraînement"),
        ("test", "Test")
    ]:
        rows.extend([
            {
                "Ensemble": label,
                "Classe": "Négative",
                "Effectif": split[key]["negative"]
            },
            {
                "Ensemble": label,
                "Classe": "Positive",
                "Effectif": split[key]["positive"]
            }
        ])

    df_split = pd.DataFrame(rows)

    fig = px.bar(
        df_split,
        x="Ensemble",
        y="Effectif",
        color="Classe",
        barmode="group",
        text="Effectif",
        color_discrete_map={
            "Négative": GREEN,
            "Positive": ORANGE
        }
    )

    st.plotly_chart(
        plot_style(fig),
        use_container_width=True,
        config={"displayModeBar": False}
    )

    a, b = st.columns(2)

    a.metric(
        "Classe positive · Train",
        f"{split['train']['positive_rate']:.2f} %"
    )

    b.metric(
        "Classe positive · Test",
        f"{split['test']['positive_rate']:.2f} %"
    )

    st.caption(
        "Le split stratifié conserve approximativement "
        "la proportion des classes dans chaque ensemble."
    )


# =====================================================
# 02 — IMPUTATION
# =====================================================

with tabs[1]:
    st.write("")
    st.subheader("Traitement des valeurs manquantes")

    missing = report["missing"]

    a, b = st.columns(2)

    a.metric(
        "Avant imputation",
        missing["before_count"]
    )

    b.metric(
        "Après imputation",
        missing["after_count"]
    )

    missing_df = pd.DataFrame(
        missing["by_feature"]
    )

    missing_df = missing_df[
        missing_df["before"] > 0
    ]

    if missing_df.empty:
        st.success(
            "Aucune valeur manquante dans les "
            "variables sélectionnées."
        )
    else:
        fig = px.bar(
            missing_df,
            x="feature",
            y=["before", "after"],
            barmode="group",
            color_discrete_sequence=[ORANGE, GREEN],
            labels={
                "feature": "Variable",
                "value": "Valeurs manquantes",
                "variable": "Étape"
            }
        )

        st.plotly_chart(
            plot_style(fig),
            use_container_width=True
        )

    st.caption(
        "L'imputation a été ajustée uniquement "
        "sur le jeu d'entraînement."
    )


# =====================================================
# 03 — ENCODAGE
# =====================================================

with tabs[2]:
    st.write("")
    st.subheader("Transformation des catégories")

    encoding = report["encoding"]

    if encoding["summary"]:
        df_encoding = pd.DataFrame(
            encoding["summary"]
        )

        st.dataframe(
            df_encoding.rename(columns={
                "feature": "Variable",
                "categories": "Catégories",
                "generated_columns": "Colonnes créées"
            }),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            df_encoding,
            x="feature",
            y="generated_columns",
            color_discrete_sequence=[GREEN]
        )

        st.plotly_chart(
            plot_style(fig),
            use_container_width=True,
            config={"displayModeBar": False}
        )

        inspection = encoding["inspection"]

        if inspection:
            st.divider()
            st.markdown(
                f"#### Exemple : {inspection['feature']}"
            )

            left, right = st.columns(2, gap="large")

            with left:
                draw_bar(
                    inspection["before"],
                    "Avant encodage"
                )

            with right:
                draw_bar(
                    inspection["after"],
                    "Après One-Hot Encoding",
                    ORANGE
                )
    else:
        st.info(
            "Aucune variable catégorielle sélectionnée."
        )


# =====================================================
# 04 — NORMALISATION
# =====================================================

with tabs[3]:
    st.write("")
    st.subheader("Transformation des variables numériques")

    scaling_info = report["scaling"]

    if scaling_info["enabled"]:
        st.success("StandardScaler activé")
    else:
        st.info("Normalisation désactivée")

    stats = scaling_info["statistics"]

    if stats:
        df_stats = pd.DataFrame(stats)

        st.dataframe(
            df_stats.rename(columns={
                "feature": "Variable",
                "mean_before": "Moyenne avant",
                "std_before": "Écart-type avant",
                "mean_after": "Moyenne après",
                "std_after": "Écart-type après"
            }),
            use_container_width=True,
            hide_index=True
        )

    inspection = scaling_info["inspection"]

    if inspection:
        st.divider()
        st.markdown(
            f"#### Distribution : {inspection['feature']}"
        )

        left, right = st.columns(2, gap="large")

        with left:
            draw_bar(
                inspection["before"],
                "Avant transformation"
            )

        with right:
            draw_bar(
                inspection["after"],
                "Après transformation",
                ORANGE
            )

    st.caption(
        "Les graphiques représentent les données "
        "d'entraînement avant et après traitement."
    )


# =====================================================
# 05 — DATASET FINAL
# =====================================================

with tabs[4]:
    st.write("")
    st.subheader("Dataset après preprocessing")

    a, b = st.columns(2)

    a.metric(
        "Dimensions train",
        f"{dims['train_rows']} × {dims['output_features']}"
    )

    b.metric(
        "Dimensions test",
        f"{dims['test_rows']} × {dims['output_features']}"
    )

    st.markdown("#### Avant préparation")

    st.dataframe(
        pd.DataFrame(report["preview"]["before"]),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Après préparation")

    st.dataframe(
        pd.DataFrame(report["preview"]["after"]),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Aperçu de 8 observations du train. "
        "Les données de test sont transformées "
        "sans ajustement supplémentaire."
    )

    st.success(
        "Préparation terminée. Les paramètres "
        "sont conservés pour le futur module Modélisation."
    )
