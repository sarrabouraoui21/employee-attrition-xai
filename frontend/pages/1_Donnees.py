
import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import api_get, api_upload

GREEN = "#16665C"
ORANGE = "#C46C54"
TEXT = "#56645E"


def style_chart(fig, height=340):
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
        margin=dict(l=12, r=12, t=25, b=30),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(gridcolor="#EDF0EE", title=None),
        legend_title_text=""
    )
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def fetch(endpoint, dataset_id=None, limit=None):
    params = {}

    if dataset_id:
        params["dataset_id"] = dataset_id

    if limit is not None:
        params["limit"] = limit

    return api_get(endpoint, params=params)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="eyebrow">MODULE 01 / DONNÉES</div>',
    unsafe_allow_html=True
)

st.title("Exploration des données")

st.markdown(
    '<div class="page-description">'
    "Sélectionnez votre source et explorez les "
    "distributions, statistiques et corrélations."
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# DATA SOURCE
# ==========================================

with st.container(border=True):
    st.markdown("### Source des données")

    source = st.radio(
        "Sélectionnez une source",
        [
            "IBM HR Analytics · Kaggle",
            "Importer un CSV"
        ],
        index=(
            1 if st.session_state.get("active_dataset_id")
            else 0
        ),
        horizontal=True,
        key="dataset_source"
    )

    active_id = st.session_state.get("active_dataset_id")

    if source == "IBM HR Analytics · Kaggle":

        if active_id:
            st.session_state["active_dataset_id"] = None
            st.rerun()

        st.success(
            "Dataset IBM HR Analytics chargé "
            "automatiquement via KaggleHub."
        )

    else:
        uploaded = st.file_uploader(
            "Fichier CSV",
            type=["csv"]
        )

        if uploaded is not None:

            signature = hashlib.sha256(
                uploaded.getvalue()
            ).hexdigest()

            if st.session_state.get("csv_signature") != signature:
                st.session_state["csv_signature"] = signature
                st.session_state.pop("csv_inspection", None)

            if st.button("Analyser le fichier CSV"):
                try:
                    with st.spinner("Analyse du fichier..."):
                        inspection = api_upload(
                            "/data/inspect-csv",
                            uploaded
                        )

                    st.session_state["csv_inspection"] = inspection

                except Exception as error:
                    st.error(str(error))

            inspection = st.session_state.get("csv_inspection")

            if inspection:
                st.caption(
                    f"{inspection['rows']} observations · "
                    f"{len(inspection['columns'])} colonnes"
                )

                candidates = inspection["binary_candidates"]

                if not candidates:
                    st.error(
                        "Aucune variable cible binaire valide. "
                        "Le fichier ne peut pas être importé."
                    )

                else:
                    st.markdown("#### Configuration de la cible")

                    target = st.selectbox(
                        "Variable cible *",
                        options=list(candidates.keys()),
                        index=None,
                        placeholder="Choisir la variable à prédire",
                        key=f"target_{signature[:16]}"
                    )

                    if target is not None:
                        classes = candidates[target]

                        st.caption(
                            "Classes détectées : "
                            + " / ".join(classes)
                        )

                        st.caption(
                            "La classe positive sera déterminée "
                            "automatiquement par le backend."
                        )

                        if st.button(
                            "Importer et utiliser ce dataset",
                            type="primary",
                            use_container_width=True
                        ):
                            try:
                                result = api_upload(
                                    "/data/upload",
                                    uploaded,
                                    fields={"target": target}
                                )

                                st.session_state[
                                    "active_dataset_id"
                                ] = result["dataset_id"]

                                st.session_state[
                                    "import_message"
                                ] = (
                                    "CSV importé · Classe positive : "
                                    + result["positive_class"]
                                )

                                st.rerun()

                            except Exception as error:
                                st.error(str(error))

        if st.session_state.get("import_message"):
            st.success(st.session_state.pop("import_message"))


# ==========================================
# DATASET ACTIF
# ==========================================

dataset_id = st.session_state.get("active_dataset_id")

try:
    overview = fetch("/data/overview", dataset_id)
    preview = fetch("/data/preview", dataset_id, limit=100)

except Exception as error:
    st.error(f"Erreur API : {error}")
    st.stop()


meta = overview["dataset"]
distribution = overview["target_distribution"]

positive = meta["positive_class"]
negative = meta["negative_class"]

positives = distribution.get(positive, 0)
negatives = distribution.get(negative, 0)

total = overview["rows"]
rate = positives / total * 100 if total else 0
missing_count = sum(overview["missing_values"].values())


# ==========================================
# INDICATEURS
# ==========================================

st.write("")
st.markdown("### Dataset actif")

st.caption(
    f"Source : {meta['source']}  |  "
    f"Cible : {meta['target']}  |  "
    f"Classe positive : {positive}"
)

if meta.get("class_selection") == "convention":
    st.warning(
        f"Classe positive définie par convention : {positive}. "
        "Vérifiez qu'elle correspond à votre événement métier."
    )

c1, c2, c3, c4 = st.columns(4)

c1.metric("Observations", total)
c2.metric("Variables", overview["columns_count"])
c3.metric("Classe positive", f"{rate:.1f} %")
c4.metric("Valeurs manquantes", missing_count)

st.write("")

tabs = st.tabs([
    "Vue générale",
    "Analyse des variables",
    "Qualité & corrélations"
])


# ==========================================
# ONGLET 1 — VUE GÉNÉRALE
# ==========================================

with tabs[0]:
    st.write("")
    st.subheader("Distribution de la cible")

    df_target = pd.DataFrame({
        "Classe": [negative, positive],
        "Effectif": [negatives, positives]
    })

    fig = px.bar(
        df_target,
        x="Classe",
        y="Effectif",
        color="Classe",
        text="Effectif",
        color_discrete_map={
            negative: GREEN,
            positive: ORANGE
        }
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.divider()

    st.subheader("Aperçu du dataset")

    st.dataframe(
        pd.DataFrame(preview["rows"]),
        use_container_width=True,
        hide_index=True,
        height=350
    )

    with st.expander("Statistiques descriptives"):
        stats = overview["numeric_statistics"]

        if stats:
            st.dataframe(
                pd.DataFrame.from_dict(
                    stats,
                    orient="index"
                ),
                use_container_width=True
            )
        else:
            st.info("Aucune variable numérique.")


# ==========================================
# ONGLET 2 — VARIABLES
# ==========================================

with tabs[1]:
    st.write("")
    st.subheader("Analyse par variable")

    excluded = {meta["target"]}

    if meta["source"] == "kaggle":
        excluded.update({
            "EmployeeNumber",
            "EmployeeCount",
            "StandardHours",
            "Over18"
        })

    features = [
        col for col in overview["columns"]
        if col not in excluded
    ]

    if features:
        default = (
            "OverTime" if "OverTime" in features
            else features[0]
        )

        selected = st.selectbox(
            "Variable explicative",
            features,
            index=features.index(default)
        )

        try:
            analysis = fetch(
                f"/data/feature/{selected}",
                dataset_id
            )

            df_feature = pd.DataFrame(
                analysis["groups"]
            )

            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("#### Distribution")

                fig = px.bar(
                    df_feature,
                    x="label",
                    y="count",
                    color_discrete_sequence=[GREEN]
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

            with right:
                st.markdown("#### Taux positif (%)")

                fig = px.bar(
                    df_feature,
                    x="label",
                    y="positive_rate",
                    color_discrete_sequence=[ORANGE]
                )

                fig.update_yaxes(range=[0, 100])

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        except Exception as error:
            st.error(f"Erreur EDA : {error}")

    else:
        st.info("Aucune variable explicative disponible.")


# ==========================================
# ONGLET 3 — QUALITÉ
# ==========================================

with tabs[2]:
    st.write("")
    st.subheader("Valeurs manquantes")

    missing = overview["missing_values"]

    if missing:
        df_missing = pd.DataFrame({
            "Variable": list(missing.keys()),
            "Valeurs manquantes": list(missing.values())
        })

        fig = px.bar(
            df_missing,
            x="Variable",
            y="Valeurs manquantes",
            color_discrete_sequence=[ORANGE]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    else:
        st.success("Aucune valeur manquante détectée.")

    st.divider()

    st.subheader("Matrice de corrélation")

    try:
        corr = fetch(
            "/data/correlations",
            dataset_id
        )

        if corr["features"]:
            fig = px.imshow(
                corr["matrix"],
                x=corr["features"],
                y=corr["features"],
                color_continuous_scale=[
                    [0, "#B85C4A"],
                    [0.5, "#F7F8F6"],
                    [1, GREEN]
                ],
                zmin=-1,
                zmax=1,
                aspect="auto"
            )

            fig.update_layout(
                height=620,
                paper_bgcolor="#FFFFFF"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "Pas assez de variables numériques "
                "pour calculer les corrélations."
            )

    except Exception as error:
        st.error(f"Erreur corrélations : {error}")
