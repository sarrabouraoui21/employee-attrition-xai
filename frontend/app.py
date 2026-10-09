
import streamlit as st

from api_client import api_get
from styles import apply_styles


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Attrition Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_styles()

if "active_dataset_id" not in st.session_state:
    st.session_state["active_dataset_id"] = None


@st.cache_data(ttl=300, show_spinner=False)
def get_overview(dataset_id=None):
    params = (
        {"dataset_id": dataset_id}
        if dataset_id else None
    )

    return api_get(
        "/data/overview",
        params=params
    )


# =====================================================
# DASHBOARD
# =====================================================

def render_home():
    st.markdown(
        '<div class="eyebrow">'
        'WORKSPACE / DATA ANALYTICS'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Analyse de l'attrition")

    st.markdown(
        '<div class="page-description">'
        "Explorez vos données, préparez vos variables, "
        "entraînez vos modèles et comparez "
        "leurs performances."
        '</div>',
        unsafe_allow_html=True
    )

    try:
        dataset_id = st.session_state[
            "active_dataset_id"
        ]

        data = get_overview(dataset_id)

    except Exception as error:
        st.error(
            f"Chargement impossible : {error}"
        )
        st.stop()

    meta = data["dataset"]
    distribution = data["target_distribution"]

    positive = meta["positive_class"]
    negative = meta["negative_class"]

    positive_count = distribution.get(
        positive, 0
    )

    negative_count = distribution.get(
        negative, 0
    )

    total = data["rows"]

    rate = (
        positive_count / total * 100
        if total else 0
    )

    negative_rate = (
        negative_count / total * 100
        if total else 0
    )

    st.markdown("### Indicateurs clés")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observations", total)
    c2.metric("Classe positive", positive_count)
    c3.metric("Taux positif", f"{rate:.1f} %")
    c4.metric("Variables", data["columns_count"])

    st.write("")

    left, right = st.columns(
        [1.4, 1],
        gap="large"
    )

    with left:
        with st.container(border=True):
            st.markdown(
                "#### Répartition des classes"
            )

            st.caption(
                f"Variable cible : {meta['target']}"
            )

            st.markdown(
                '<div class="bar-track">'
                f'<div class="bar-fill" '
                f'style="width:{negative_rate:.2f}%"></div>'
                '</div>',
                unsafe_allow_html=True
            )

            a, b = st.columns(2)

            a.metric(
                f"Classe {negative}",
                negative_count
            )

            b.metric(
                f"Classe {positive}",
                positive_count
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Dataset actif")

            is_kaggle = (
                meta["source"] == "kaggle"
            )

            st.markdown(
                "**IBM HR Analytics · Kaggle**"
                if is_kaggle
                else "**CSV personnalisé**"
            )

            st.caption(
                f"Cible : {meta['target']}"
            )

            st.caption(
                f"Classe positive : {positive}"
            )

            if is_kaggle:
                st.link_button(
                    "Consulter le dataset",
                    "https://www.kaggle.com/datasets/"
                    "pavansubhasht/"
                    "ibm-hr-analytics-attrition-dataset"
                )
            else:
                if st.button(
                    "Revenir au dataset Kaggle"
                ):
                    st.session_state[
                        "active_dataset_id"
                    ] = None

                    st.rerun()

    # =================================================
    # MODULES
    # =================================================

    st.write("")
    st.markdown("### Pipeline de classification")

    col1, col2 = st.columns(2, gap="large")
    col3, col4 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("#### 01 · Exploration")
            st.caption(
                "Statistiques, distributions et corrélations"
            )

            if st.button(
                "Explorer",
                key="home_eda",
                use_container_width=True
            ):
                st.switch_page(data_page)

    with col2:
        with st.container(border=True):
            st.markdown("#### 02 · Préparation")
            st.caption(
                "Split, imputation, encodage et normalisation"
            )

            if st.button(
                "Préparer",
                key="home_prep",
                use_container_width=True
            ):
                st.switch_page(preparation_page)

    with col3:
        with st.container(border=True):
            st.markdown("#### 03 · Modélisation")
            st.caption(
                "Entraînement et optimisation des modèles"
            )

            if st.button(
                "Entraîner",
                key="home_training",
                use_container_width=True
            ):
                st.switch_page(modelisation_page)

    with col4:
        with st.container(border=True):
            st.markdown("#### 04 · Évaluation")
            st.caption(
                "Métriques, courbes et matrices de confusion"
            )

            if st.button(
                "Évaluer",
                key="home_eval",
                type="primary",
                use_container_width=True
            ):
                st.switch_page(evaluation_page)


# =====================================================
# SIDEBAR — IDENTITÉ VISUELLE
# =====================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">A</div>
            <div>
                <div class="brand-name">
                    Attrition Analytics
                </div>
                <div class="brand-subtitle">
                    HR DATA PLATFORM
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# DÉCLARATION DES PAGES
# =====================================================

home_page = st.Page(
    render_home,
    title="Vue d'ensemble",
    icon=":material/dashboard:",
    default=True
)

data_page = st.Page(
    "pages/1_Donnees.py",
    title="Exploration des données",
    icon=":material/analytics:"
)

preparation_page = st.Page(
    "pages/2_Preparation.py",
    title="Préparation des données",
    icon=":material/tune:"
)

modelisation_page = st.Page(
    "pages/3_Modelisation.py",
    title="Modélisation",
    icon=":material/model_training:"
)

evaluation_page = st.Page(
    "pages/4_Evaluation.py",
    title="Évaluation",
    icon=":material/assessment:"
)


# =====================================================
# NAVIGATION
# =====================================================

navigation = st.navigation(
    {
        "ANALYSE": [
            home_page,
            data_page,
            preparation_page,
            modelisation_page,
            evaluation_page
        ]
    },
    position="sidebar"
)


# =====================================================
# STATUT FASTAPI
# =====================================================

with st.sidebar:
    st.divider()

    try:
        connected = (
            api_get("/health").get("status") == "ok"
        )
    except Exception:
        connected = False

    if connected:
        st.markdown(
            '<div class="status-row">'
            '<span class="status-dot"></span>'
            'API connectée · FastAPI'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.error("API FastAPI indisponible")


navigation.run()
