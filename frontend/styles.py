
import streamlit as st


def apply_styles():
    st.markdown("""
    <style>
    /* Base */
    .stApp {
        background: #F7F8F6;
        color: #222B29;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2.8rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    h1, h2, h3 {
        color: #222B29 !important;
        font-weight: 600 !important;
        letter-spacing: -0.035em !important;
    }

    h1 { font-size: 2.15rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.05rem !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F0F3F1;
        border-right: 1px solid #DFE5E1;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebarNav"] a {
        border-radius: 7px;
        font-weight: 500;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #DEEAE6;
        color: #16665C;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 22px;
    }

    .brand-mark {
        background: #16665C;
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 700;
    }

    .brand-name {
        font-size: 17px;
        font-weight: 650;
        letter-spacing: -0.5px;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #84908B;
        margin-top: 2px;
    }

    .sidebar-label {
        margin-top: 22px;
        margin-bottom: 10px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #89948F;
    }

    .sidebar-next {
        font-size: 13px;
        color: #98A39E;
        padding: 9px 12px;
    }

    /* Page headings */
    .eyebrow {
        color: #16665C;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 11px;
    }

    .page-description {
        color: #71807A;
        font-size: 14px;
        line-height: 1.7;
        max-width: 720px;
        margin-top: -8px;
        margin-bottom: 26px;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E4EAE6;
        border-radius: 11px;
        padding: 19px 21px;
    }

    [data-testid="stMetricLabel"] {
        color: #7A8781;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #25312D;
        font-size: 27px;
        font-weight: 650;
        letter-spacing: -1px;
    }

    /* Panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #E4EAE6;
        border-radius: 11px;
        background: #FFFFFF;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 23px;
        border-bottom: 1px solid #DFE5E1;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 2px;
        color: #71807A;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        color: #16665C;
    }

    /* Controls */
    .stButton > button[kind="primary"] {
        background: #16665C;
        border: 1px solid #16665C;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button[kind="primary"]:hover {
        background: #104F47;
        border-color: #104F47;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E4EAE6;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Small elements */
    .soft-divider {
        height: 1px;
        background: #E2E8E4;
        margin: 24px 0;
    }

    .status-row {
        font-size: 12px;
        color: #596F67;
        padding: 10px 0;
    }

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #2F9879;
        border-radius: 50%;
        margin-right: 8px;
    }

    .muted {
        color: #7A8781;
        font-size: 13px;
    }

    .bar-track {
        height: 9px;
        border-radius: 20px;
        background: #ECD8D0;
        overflow: hidden;
        margin: 17px 0 12px;
    }

    .bar-fill {
        height: 100%;
        background: #16665C;
        border-radius: 20px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
