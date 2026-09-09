# TalentIQ — RAG-Powered Talent Search Engine

TalentIQ is an industry-level AI recruitment system that uses **Retrieval-Augmented Generation (RAG)** to search, retrieve, evaluate, and rank candidates from a large resume dataset using natural-language recruiter queries.

Instead of relying only on keyword matching, TalentIQ combines **semantic embeddings, FAISS vector search, and Gemini-based evaluation** to identify candidates whose experience and skills are contextually relevant to a hiring request.

## Project Overview

Recruiters often need to review hundreds or thousands of resumes manually.

TalentIQ solves this problem by allowing a recruiter to write a natural-language request such as:

```text
Find me a Data Analyst with Python, SQL, Tableau, machine learning,
and experience communicating insights to business stakeholders.

The system then:

1. Converts the recruiter query into a semantic embedding
2. Searches a vector database of resume embeddings
3. Retrieves the most relevant candidates
4. Sends the strongest candidates to Gemini
5. Generates a detailed AI evaluation
6. Produces a ranked recommendation
7. Runs an additional bias and fairness audit

## Core Architecture

```text
Resume Dataset
      ↓
Text Preparation
      ↓
Sentence Transformer Embeddings
      ↓
FAISS Vector Database
      ↓
Recruiter Query
      ↓
Query Embedding
      ↓
Semantic Similarity Search
      ↓
Top Candidate Retrieval
      ↓
Gemini LLM Evaluation
      ↓
Candidate Ranking
      ↓
Bias & Fairness Audit
      ↓
Streamlit Recruiter Interface
```

## Dataset

This project uses a resume dataset containing **2,484 resumes** across multiple professional categories.

The dataset includes the following fields:

```text
ID
Resume_str
Resume_html
Category
```

The main field used for the RAG pipeline is:

```text
Resume_str
```

which contains the textual content of each candidate resume.

The dataset contains resumes from multiple professional domains, including:

* Information Technology
* Engineering
* Finance
* Accounting
* Human Resources
* Sales
* Healthcare
* Business Development
* and other professional categories

## Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Embeddings

* Sentence Transformers
* `all-MiniLM-L6-v2`

### Vector Search

* FAISS

### Large Language Model

* Google Gemini

### RAG / LLM Integration

* LangChain
* `langchain-google-genai`

### Frontend

* Streamlit

### Environment Management

* Python Virtual Environment
* python-dotenv

### Version Control

* Git
* GitHub

## Semantic Embeddings

Each cleaned resume is converted into a dense vector representation using:

```text
all-MiniLM-L6-v2
```

The resulting embedding dimension is:

```text
384
```

Each resume is therefore represented as a vector containing 384 numerical features describing its semantic meaning.

This allows the system to search based on meaning rather than exact keyword overlap.

## FAISS Vector Database

The generated resume embeddings are stored inside a **FAISS vector index**.

FAISS enables efficient similarity search across thousands of candidate vectors.

For each recruiter query:

```text
Natural Language Query
        ↓
Sentence Transformer
        ↓
Query Embedding
        ↓
FAISS Search
        ↓
Top-K Resume Matches
```

The recruiter can control how many candidates are initially retrieved.

## Semantic Talent Search

TalentIQ supports natural-language hiring requests.

Example:

```text
Find me a candidate with Python, SQL, data analysis,
machine learning, and visualization experience.
```

Instead of searching only for exact words, the system compares the semantic meaning of the request against the semantic representation of every resume.

The retrieved candidates are ranked according to vector similarity.

## LLM-Based Candidate Evaluation

After semantic retrieval, the top candidates are passed to **Gemini** for deeper evaluation.

For each candidate, the LLM evaluates:

* relevant technical skills
* professional experience
* education
* projects
* tools and technologies
* missing requirements
* overall fit for the recruiter request

Each candidate receives a suitability score from:

```text
0 - 100
```

Gemini then produces a final ranking of the strongest candidates.

Example output structure:

```text
Candidate 1
Suitability Score: 92/100

Strengths:
- Strong Python experience
- Advanced SQL knowledge
- Machine learning projects
- Data visualization experience

Missing Requirements:
- Limited Tableau evidence

Final Ranking:
1. Candidate A
2. Candidate B
3. Candidate C
```

## Bias & Fairness Audit

TalentIQ includes an additional AI-based fairness audit.

The audit reviews the generated candidate evaluation and checks whether the recommendation appears to rely on sensitive demographic characteristics rather than job-relevant qualifications.

The system checks for potential inappropriate influence from factors such as:

* age
* gender
* race or ethnicity
* religion
* nationality
* marital status
* family status

The audit is designed to encourage candidate evaluation based primarily on:

* skills
* experience
* education
* certifications
* projects
* technologies
* job-related qualifications

The system does not infer demographic traits that are not explicitly present.

The audit returns:

```text
Bias Risk
Potential Issues
Evidence
Recommended Correction
Fair Evaluation Principle
```

## Streamlit Recruiter Dashboard

TalentIQ includes a professional recruiter-facing dashboard built with Streamlit.

The interface includes:

* natural-language talent search
* configurable candidate retrieval count
* indexed resume statistics
* FAISS status
* Gemini evaluation status
* ranked candidate cards
* candidate category information
* semantic vector distance
* expandable resume previews
* AI candidate evaluation
* bias and fairness audit

## Project Structure

```text
rag-talent-search-engine/
│
├── app/
│   └── app.py
│
├── data/
│   └── resumes/
│       └── Resume.csv
│
├── images/
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── loader.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── llm_evaluator.py
│
├── vector_db/
│   └── resume_index.faiss
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/rewanahmedelbayoumi/rag-talent-search-engine.git
```

Enter the project:

```bash
cd rag-talent-search-engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Gemini API Configuration

Create a file named:

```text
.env
```

Add:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Do not commit the `.env` file to GitHub.

Make sure `.gitignore` contains:

```gitignore
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
vector_db/
```

## Run the Application

From the project root:

```powershell
streamlit run app\app.py
```

The Streamlit dashboard will open in your browser.

## Example Recruiter Query

```text
Find me a Data Analyst with strong Python, SQL,
machine learning, Tableau, and business intelligence experience.
```

TalentIQ will:

```text
Query
↓
Generate Embedding
↓
Search FAISS
↓
Retrieve Top Candidates
↓
Evaluate Top 3 with Gemini
↓
Rank Candidates
↓
Run Bias Audit
↓
Display Results
```

## Key Features

* Semantic resume retrieval
* Natural-language recruiter queries
* Vector-based candidate matching
* FAISS similarity search
* Transformer-based embeddings
* LLM candidate evaluation
* AI-generated suitability scoring
* Candidate ranking
* Missing-requirement detection
* Bias and fairness auditing
* Professional Streamlit interface
* Reproducible Python environment
* Git and GitHub version control

## Key Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Natural Language Processing
* Sentence embeddings
* Semantic search
* Vector databases
* FAISS
* Large Language Models
* Prompt engineering
* Candidate retrieval
* AI-assisted ranking
* Responsible AI concepts
* Streamlit application development
* End-to-end AI system architecture

## Important Note

TalentIQ is an experimental AI recruitment support system.

Its recommendations should be treated as decision-support information rather than autonomous hiring decisions.

Human review should remain part of any real recruitment process.



## Final Pipeline

```text
2,484 Resumes
      ↓
Text Preparation
      ↓
384-Dimensional Semantic Embeddings
      ↓
FAISS Vector Index
      ↓
Natural Language Recruiter Query
      ↓
Semantic Candidate Retrieval
      ↓
Top Candidate Selection
      ↓
Gemini Candidate Evaluation
      ↓
Suitability Scoring + Ranking
      ↓
Bias & Fairness Audit
      ↓
Professional Recruiter Dashboard
```
