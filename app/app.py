import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.llm_evaluator import evaluate_candidates, bias_check


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TalentIQ — AI Talent Search",
    page_icon="◼",
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
        background:
            radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at bottom left,
                rgba(99, 102, 241, 0.08),
                transparent 30%
            ),
            #07111f;

        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: #0b1626;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    h1, h2, h3 {
        font-family: Inter, Arial, sans-serif;
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

    .hero {
        padding: 2rem 0 1.5rem 0;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.14);
        border: 1px solid rgba(96, 165, 250, 0.25);
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.3rem;
        line-height: 1.05;
        font-weight: 800;
        margin-bottom: 0.8rem;
        color: #f8fafc;
    }

    .hero-subtitle {
        max-width: 850px;
        font-size: 1.08rem;
        color: #94a3b8;
        line-height: 1.7;
    }

    .stTextArea textarea {
        background: #0f1b2d !important;
        color: #f8fafc !important;
        border: 1px solid #24354d !important;
        border-radius: 14px !important;
        min-height: 150px;
        font-size: 1rem;
    }

    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 52px;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.98rem;
        color: white;

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #4f46e5
            );

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 10px 24px
            rgba(37, 99, 235, 0.28);
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.76rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 0.25rem;
    }

    .candidate-card {
        background:
            linear-gradient(
                180deg,
                rgba(15, 23, 42, 0.96),
                rgba(10, 18, 31, 0.96)
            );

        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 20px;
        padding: 1.25rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.16);
    }

    .candidate-card-top {
        border-color: rgba(59, 130, 246, 0.55);
        box-shadow:
            0 16px 50px
            rgba(37,99,235,0.13);
    }

    .candidate-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #172554;
        color: #bfdbfe;
        font-weight: 800;
        margin-right: 0.7rem;
    }

    .candidate-id {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .category-chip {
        display: inline-block;
        padding: 0.34rem 0.72rem;
        border-radius: 999px;
        background: rgba(30, 64, 175, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.18);
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.65rem;
    }

    .distance-label {
        color: #64748b;
        font-size: 0.76rem;
        text-transform: uppercase;
        font-weight: 700;
        margin-top: 1rem;
    }

    .distance-value {
        color: #cbd5e1;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 2rem 0 1rem 0;
    }

    .section-subtitle {
        color: #64748b;
        margin-top: -0.7rem;
        margin-bottom: 1.2rem;
    }

    .ai-panel {
        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.18),
                rgba(79, 70, 229, 0.10)
            );

        border:
            1px solid
            rgba(96, 165, 250, 0.22);

        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        margin-top: 1rem;
    }

    .bias-panel {
        background:
            linear-gradient(
                135deg,
                rgba(15, 118, 110, 0.15),
                rgba(6, 78, 59, 0.08)
            );

        border:
            1px solid
            rgba(45, 212, 191, 0.18);

        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        margin-top: 1rem;
    }

    .sidebar-brand {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }

    .sidebar-caption {
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    .sidebar-section {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin: 1rem 0 0.6rem 0;
    }

    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 14px;
    }

    .footer-note {
        margin-top: 3rem;
        color: #475569;
        text-align: center;
        font-size: 0.78rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA + MODELS
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_faiss_index():
    index_path = (
        project_root
        / "vector_db"
        / "resume_index.faiss"
    )

    return faiss.read_index(
        str(index_path)
    )


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
        .str
        .strip()
        != ""
    ].reset_index(
        drop=True
    )

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
# SEARCH LOGIC
# =========================================================

def search_resumes(
    query,
    top_k=5
):
    query_embedding = (
        embedding_model.encode(
            [query]
        )
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = (
        index.search(
            query_embedding,
            top_k
        )
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
                "resume": candidate[
                    "clean_resume"
                ]
            }
        )

    return results


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            TalentIQ
        </div>

        <div class="sidebar-caption">
            AI-powered semantic talent discovery
            for modern recruiting teams.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-section">
            Search Settings
        </div>
        """,
        unsafe_allow_html=True
    )

    top_k = st.slider(
        "Candidates to retrieve",
        min_value=3,
        max_value=10,
        value=5
    )

    st.markdown(
        """
        <div class="sidebar-section">
            System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "FAISS index connected"
    )

    st.info(
        "Gemini evaluation enabled"
    )

    st.markdown(
        """
        <div class="sidebar-section">
            Dataset
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        f"{len(rag_df):,} resumes indexed"
    )

    st.markdown("---")

    st.caption(
        "Semantic retrieval powered by "
        "Sentence Transformers + FAISS."
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="eyebrow">
            AI Recruitment Intelligence
        </div>

        <div class="hero-title">
            Find the right talent<br>
            beyond keyword matching.
        </div>

        <div class="hero-subtitle">
            TalentIQ uses semantic search and
            large language model evaluation to
            surface candidates based on actual
            skills, experience, and contextual fit.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# METRICS
# =========================================================

m1, m2, m3 = st.columns(3)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Indexed Resumes
            </div>

            <div class="metric-value">
                {len(rag_df):,}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Embedding Dimension
            </div>

            <div class="metric-value">
                {index.d}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        """
        <div class="metric-card">

            <div class="metric-label">
                AI Evaluation
            </div>

            <div class="metric-value">
                Gemini
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SEARCH
# =========================================================

st.markdown(
    """
    <div class="section-title">
        Talent Search
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Describe the ideal candidate
        using natural language.
    </div>
    """,
    unsafe_allow_html=True
)


query = st.text_area(
    "Candidate requirements",
    placeholder=(
        "Example: Find me a data analyst "
        "with strong Python, SQL, machine "
        "learning, Tableau, and experience "
        "communicating insights to business "
        "stakeholders."
    ),
    label_visibility="collapsed"
)


search_button = st.button(
    "Search Talent",
    use_container_width=True
)


# =========================================================
# RESULTS
# =========================================================

if search_button:

    if not query.strip():

        st.warning(
            "Enter a candidate description "
            "before starting the search."
        )

    else:

        with st.spinner(
            "Searching semantic talent index..."
        ):

            results = search_resumes(
                query=query,
                top_k=top_k
            )


        st.markdown(
            """
            <div class="section-title">
                Top Candidates
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-subtitle">
                Candidates ranked by semantic
                similarity to your search.
            </div>
            """,
            unsafe_allow_html=True
        )


        for result in results:

            if result["rank"] == 1:

                card_class = (
                    "candidate-card "
                    "candidate-card-top"
                )

                top_label = (
                    " • BEST SEMANTIC MATCH"
                )

            else:

                card_class = (
                    "candidate-card"
                )

                top_label = ""


            st.markdown(
                f"""
                <div class="{card_class}">

                    <div>

                        <span class="candidate-rank">
                            #{result['rank']}
                        </span>

                        <span class="candidate-id">
                            Candidate {result['id']}
                            {top_label}
                        </span>

                    </div>

                    <div class="category-chip">
                        {result['category']}
                    </div>

                    <div class="distance-label">
                        FAISS Distance
                    </div>

                    <div class="distance-value">
                        {result['distance']:.4f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            with st.expander(
                f"View Candidate "
                f"{result['id']} Resume"
            ):

                st.write(
                    result["resume"][:5000]
                )


        # =============================================
        # LLM CANDIDATE EVALUATION
        # =============================================

        st.markdown(
            """
            <div class="section-title">
                AI Candidate Intelligence
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-subtitle">
                Gemini evaluates the three
                strongest retrieved candidates
                against the original hiring
                requirements.
            </div>
            """,
            unsafe_allow_html=True
        )


        with st.spinner(
            "Evaluating candidate fit "
            "with Gemini..."
        ):

            evaluation = evaluate_candidates(
                query,
                results[:3]
            )


        st.markdown(
            """
            <div class="ai-panel">
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            evaluation
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # =============================================
        # BIAS CHECK
        # =============================================

        st.markdown(
            """
            <div class="section-title">
                Bias & Fairness Audit
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-subtitle">
                A second AI review checks whether
                the recommendation is based on
                job-relevant qualifications rather
                than sensitive demographic factors.
            </div>
            """,
            unsafe_allow_html=True
        )


        with st.spinner(
            "Running bias and fairness audit..."
        ):

            bias_report = bias_check(
                query,
                results[:3],
                evaluation
            )


        st.markdown(
            """
            <div class="bias-panel">
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            bias_report
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-note">

        TalentIQ • RAG-powered talent discovery
        using Sentence Transformers, FAISS and Gemini

    </div>
    """,
    unsafe_allow_html=True
)