import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# =========================================================
# PROJECT PATH
# =========================================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.llm_evaluator import evaluate_candidates, bias_check


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TalentIQ",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #07111f;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1626;
        border-right: 1px solid #172033;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    .brand {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 2rem;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.4rem 0.75rem;
        background: #102755;
        border: 1px solid #1e3a8a;
        border-radius: 999px;
        color: #93c5fd;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }

    .hero-title {
        color: #ffffff;
        font-size: 3.2rem;
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.8rem;
    }

    .hero-description {
        color: #94a3b8;
        max-width: 780px;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 2rem;
    }

    .metric-box {
        background: #0d1829;
        border: 1px solid #1d2a3e;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 110px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .metric-number {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    .section-heading {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 2rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 1rem;
    }

    .candidate-card {
        background: #0d1829;
        border: 1px solid #1d2a3e;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 0.8rem;
    }

    .candidate-card.best {
        border: 1px solid #2563eb;
        background: #0e1c33;
    }

    .rank-badge {
        display: inline-block;
        background: #172554;
        color: #bfdbfe;
        border-radius: 8px;
        padding: 0.35rem 0.55rem;
        font-weight: 800;
        margin-right: 0.5rem;
    }

    .candidate-title {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .category {
        display: inline-block;
        margin-top: 0.7rem;
        padding: 0.3rem 0.65rem;
        background: #102755;
        color: #93c5fd;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .distance {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 0.8rem;
    }

    .best-label {
        color: #60a5fa;
        font-size: 0.72rem;
        font-weight: 800;
        margin-left: 0.5rem;
    }

    .stTextArea textarea {
        background-color: #0d1829 !important;
        border: 1px solid #24334a !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        min-height: 150px;
    }

    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 50px;
        border: none;
        border-radius: 12px;
        background: #2563eb;
        color: white;
        font-size: 0.95rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
        border: none;
    }

    [data-testid="stExpander"] {
        background-color: #0a1422;
        border: 1px solid #1d2a3e;
        border-radius: 12px;
    }

    .footer {
        color: #475569;
        text-align: center;
        font-size: 0.78rem;
        padding-top: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# LOAD FAISS
# =========================================================

@st.cache_resource
def load_faiss_index():
    index_path = project_root / "vector_db" / "resume_index.faiss"

    return faiss.read_index(str(index_path))


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_resume_data():

    csv_path = (
        project_root
        / "data"
        / "resumes"
        / "Resume.csv"
    )

    df = pd.read_csv(csv_path)

    rag_df = df[
        [
            "ID",
            "Resume_str",
            "Category"
        ]
    ].copy()

    rag_df = rag_df.rename(
        columns={
            "Resume_str": "resume_text"
        }
    )

    rag_df = rag_df.dropna(
        subset=["resume_text"]
    )

    rag_df = rag_df[
        rag_df["resume_text"]
        .astype(str)
        .str.strip()
        != ""
    ].reset_index(drop=True)

    rag_df["clean_resume"] = (
        rag_df["resume_text"]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    return rag_df


embedding_model = load_embedding_model()
index = load_faiss_index()
rag_df = load_resume_data()


# =========================================================
# SEARCH FUNCTION
# =========================================================

def search_resumes(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, resume_index in enumerate(
        indices[0]
    ):

        candidate = rag_df.iloc[
            resume_index
        ]

        results.append(
            {
                "rank": rank + 1,
                "id": candidate["ID"],
                "category": candidate["Category"],
                "distance": float(
                    distances[0][rank]
                ),
                "resume": candidate["clean_resume"]
            }
        )

    return results


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            TalentIQ
        </div>

        <div class="brand-subtitle">
            Semantic AI talent discovery platform
            powered by RAG.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Search Settings")

    top_k = st.slider(
        "Candidates to retrieve",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.subheader("System Status")

    st.success(
        "FAISS Vector Index"
    )

    st.success(
        "Gemini Evaluation"
    )

    st.success(
        "Bias Audit"
    )

    st.divider()

    st.subheader("Dataset")

    st.metric(
        "Indexed resumes",
        f"{len(rag_df):,}"
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="eyebrow">
        AI RECRUITMENT INTELLIGENCE
    </div>

    <div class="hero-title">
        Find talent by meaning,<br>
        not just keywords.
    </div>

    <div class="hero-description">
        TalentIQ combines semantic search,
        vector retrieval and large language
        model evaluation to discover candidates
        whose experience matches your hiring
        requirements.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="metric-box">

            <div class="metric-label">
                Indexed Resumes
            </div>

            <div class="metric-number">
                {len(rag_df):,}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-box">

            <div class="metric-label">
                Vector Dimension
            </div>

            <div class="metric-number">
                {index.d}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="metric-box">

            <div class="metric-label">
                Retrieval Engine
            </div>

            <div class="metric-number">
                FAISS
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SEARCH AREA
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        Talent Search
    </div>

    <div class="section-description">
        Describe the candidate profile,
        skills and experience you are looking for.
    </div>
    """,
    unsafe_allow_html=True
)


query = st.text_area(
    "Candidate requirements",
    placeholder=(
        "Example: Find me a data analyst "
        "with Python, SQL, Tableau, machine "
        "learning and business intelligence "
        "experience."
    ),
    label_visibility="collapsed"
)


search_clicked = st.button(
    "Search Candidates",
    use_container_width=True
)


# =========================================================
# RESULTS
# =========================================================

if search_clicked:

    if not query.strip():

        st.warning(
            "Please describe the candidate "
            "you are looking for."
        )

    else:

        with st.spinner(
            "Searching the talent index..."
        ):

            results = search_resumes(
                query,
                top_k
            )


        # =================================================
        # CANDIDATES
        # =================================================

        st.markdown(
            """
            <div class="section-heading">
                Top Candidates
            </div>

            <div class="section-description">
                Ranked using semantic similarity
                between your request and the
                candidate resumes.
            </div>
            """,
            unsafe_allow_html=True
        )


        for result in results:

            best_class = (
                "best"
                if result["rank"] == 1
                else ""
            )

            best_label = (
                '<span class="best-label">'
                'BEST MATCH'
                '</span>'
                if result["rank"] == 1
                else ""
            )

            st.markdown(
                f"""
                <div class="candidate-card {best_class}">

                    <span class="rank-badge">
                        #{result['rank']}
                    </span>

                    <span class="candidate-title">
                        Candidate {result['id']}
                    </span>

                    {best_label}

                    <br>

                    <span class="category">
                        {result['category']}
                    </span>

                    <div class="distance">
                        Vector distance:
                        {result['distance']:.4f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander(
                f"View Candidate {result['id']} Resume"
            ):

                st.write(
                    result["resume"][:5000]
                )


        # =================================================
        # GEMINI EVALUATION
        # =================================================

        st.markdown(
            """
            <div class="section-heading">
                AI Candidate Evaluation
            </div>

            <div class="section-description">
                Gemini evaluates the three
                strongest candidates against
                the hiring requirements.
            </div>
            """,
            unsafe_allow_html=True
        )


        with st.spinner(
            "Gemini is evaluating candidate fit..."
        ):

            evaluation = evaluate_candidates(
                query,
                results[:3]
            )


        st.markdown(evaluation)


        # =================================================
        # BIAS CHECK
        # =================================================

        st.markdown(
            """
            <div class="section-heading">
                Bias & Fairness Audit
            </div>

            <div class="section-description">
                The recommendation is reviewed
                for possible reliance on sensitive
                demographic characteristics.
            </div>
            """,
            unsafe_allow_html=True
        )


        with st.spinner(
            "Running fairness audit..."
        ):

            bias_report = bias_check(
                query,
                results[:3],
                evaluation
            )


        st.markdown(bias_report)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        TalentIQ · Semantic Search · FAISS ·
        Sentence Transformers · Gemini
    </div>
    """,
    unsafe_allow_html=True
)