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

from src.llm_evaluator import evaluate_candidates


st.set_page_config(
    page_title="RAG Talent Search Engine",
    page_icon="🔎",
    layout="wide"
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_faiss_index():
    index_path = project_root / "vector_db" / "resume_index.faiss"
    return faiss.read_index(str(index_path))


@st.cache_data
def load_resume_data():
    csv_path = project_root / "data" / "resumes" / "Resume.csv"

    df = pd.read_csv(csv_path)

    rag_df = df[["ID", "Resume_str", "Category"]].copy()

    rag_df = rag_df.rename(
        columns={"Resume_str": "resume_text"}
    )

    rag_df = rag_df.dropna(
        subset=["resume_text"]
    )

    rag_df = rag_df[
        rag_df["resume_text"].str.strip() != ""
    ].reset_index(drop=True)

    rag_df["clean_resume"] = (
        rag_df["resume_text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return rag_df


embedding_model = load_embedding_model()
index = load_faiss_index()
rag_df = load_resume_data()


def search_resumes(query, top_k=5):
    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, resume_index in enumerate(indices[0]):
        candidate = rag_df.iloc[resume_index]

        results.append({
            "rank": rank + 1,
            "id": candidate["ID"],
            "category": candidate["Category"],
            "distance": float(distances[0][rank]),
            "resume": candidate["clean_resume"]
        })

    return results


st.title("RAG-Powered Talent Search Engine")

st.write(
    "Search thousands of resumes using natural language and "
    "receive AI-assisted candidate recommendations."
)

query = st.text_area(
    "Describe the candidate you are looking for:",
    placeholder=(
        "Example: Find me a data analyst with Python, SQL, "
        "machine learning, and visualization experience."
    ),
    height=120
)

top_k = st.slider(
    "Number of candidates to retrieve",
    min_value=3,
    max_value=10,
    value=5
)


if st.button("Search Candidates"):
    if not query.strip():
        st.warning("Please enter a recruiter query.")

    else:
        with st.spinner("Searching resume database..."):
            results = search_resumes(
                query,
                top_k=top_k
            )

        st.subheader("Retrieved Candidates")

        for result in results:
            with st.expander(
                f"Rank {result['rank']} — "
                f"Candidate {result['id']} — "
                f"{result['category']}"
            ):
                st.write(
                    f"FAISS Distance: "
                    f"{result['distance']:.4f}"
                )

                st.write(result["resume"][:2500])

        st.subheader("AI Candidate Evaluation")

        with st.spinner("Gemini is evaluating the top candidates..."):
            evaluation = evaluate_candidates(
                query,
                results[:3]
            )

        st.markdown(evaluation)