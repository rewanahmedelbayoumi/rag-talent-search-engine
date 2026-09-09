import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


def evaluate_candidates(query, candidates):
    candidate_text = ""

    for candidate in candidates:
        candidate_text += f"""
Candidate Rank: {candidate['rank']}
Candidate ID: {candidate['id']}
Category: {candidate['category']}

Resume:
{candidate['resume']}

{"-" * 80}
"""

    prompt = f"""
You are an AI recruitment assistant.

The recruiter is looking for:

{query}

Below are the top candidates retrieved using semantic search.

Evaluate each candidate based only on the information provided in the resumes.

For each candidate:
1. Explain why the candidate matches the recruiter's request.
2. Identify relevant skills and experience.
3. Mention any important missing requirements.
4. Give a suitability score from 0 to 100.

Finally, rank the candidates from best to least suitable.

Candidates:

{candidate_text}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        return response.content[0]["text"]

    return response.content