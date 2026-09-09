import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


def _extract_text(response):
    if isinstance(response.content, list):
        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")

    return response.content


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

Important rules:

- Base the evaluation only on job-relevant information.
- Do not infer information that is not present in the resume.
- Do not use age, gender, race, ethnicity, religion, nationality,
  marital status, or family status as reasons for ranking candidates.
- Focus on skills, experience, education, projects, and demonstrated qualifications.

Candidates:

{candidate_text}
"""

    response = llm.invoke(prompt)

    return _extract_text(response)


def bias_check(query, candidates, evaluation):
    candidate_text = ""

    for candidate in candidates:
        candidate_text += f"""
Candidate ID: {candidate['id']}
Category: {candidate['category']}

Resume:
{candidate['resume']}

{"-" * 80}
"""

    prompt = f"""
You are auditing an AI recruitment recommendation for potential bias.

Recruiter Query:

{query}

Candidates:

{candidate_text}

Previous AI Evaluation:

{evaluation}

Review the recommendation and determine whether the ranking or explanation
appears to rely on demographic or personally sensitive characteristics
instead of job-relevant qualifications.

Check for inappropriate influence from factors such as:

- age
- gender
- race or ethnicity
- religion
- nationality
- marital status
- family status

Do not infer demographic traits that are not explicitly present.

Evaluate whether the recommendation is based primarily on:

- technical skills
- professional experience
- education
- certifications
- projects
- tools and technologies
- demonstrated responsibilities
- other job-related qualifications

Return the report in this format:

### Bias Risk
Low, Medium, or High

### Potential Issues
Describe any possible bias concerns.
If none are found, clearly state that no significant bias was detected.

### Evidence
Explain what parts of the recommendation support your conclusion.

### Recommended Correction
Explain how the evaluation should be corrected if bias is detected.
If no correction is needed, say so.

### Fair Evaluation Principle
Provide a short statement describing which job-relevant factors should
drive the hiring recommendation.
"""

    response = llm.invoke(prompt)

    return _extract_text(response)