import streamlit as st
import pdfplumber
import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("AI Resume Screener")

st.sidebar.info(
    """
    Upload your resume and compare it with a job description
    using AI-powered ATS analysis and RAG retrieval.
    """
)

st.sidebar.markdown("### Features")

st.sidebar.write("- ATS Match Score")
st.sidebar.write("- Skill Detection")
st.sidebar.write("- Missing Skills")
st.sidebar.write("- Resume Role Prediction")
st.sidebar.write("- Resume Suggestions")
st.sidebar.write("- RAG Resume Analysis")

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📄 AI Resume Screening System")

st.markdown(
    """
    Analyze resumes intelligently using NLP-powered ATS scoring,
    semantic similarity, RAG retrieval,
    and resume optimization suggestions.
    """
)

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type="pdf"
)

job_description = st.text_area(
    "Paste Job Description Here"
)

# ---------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------

def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text

# ---------------------------------------------------
# TEXT CHUNKING FOR RAG
# ---------------------------------------------------

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks

# ---------------------------------------------------
# SKILL DETECTION
# ---------------------------------------------------

def contains_skill(text, skill):

    escaped_skill = re.escape(skill)

    pattern = (
        r'(?<![a-zA-Z])'
        + escaped_skill +
        r'(?![a-zA-Z])'
    )

    return bool(
        re.search(
            pattern,
            text,
            re.IGNORECASE
        )
    )

# ---------------------------------------------------
# FIX 1: CACHE MODEL — prevents reload on every run
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------
# FIX 4: SKILL ALIASES — catches common variations
# ---------------------------------------------------

SKILL_ALIASES = {
    "scikit-learn": ["scikit learn", "sklearn"],
    "node.js":      ["nodejs", "node js"],
    "nlp":          ["natural language processing"],
    "machine learning": ["ml"],
    "deep learning":    ["dl"],
    "postgresql":   ["postgres"],
    "javascript":   ["js"],
    "typescript":   ["ts"],
}

def skill_in_text(text, skill):
    """Check skill and all its known aliases against text."""
    if contains_skill(text, skill):
        return True
    for alias in SKILL_ALIASES.get(skill, []):
        if contains_skill(text, alias):
            return True
    return False

# ---------------------------------------------------
# FIX 5: NAMED WEIGHT CONSTANTS — no magic numbers
# ---------------------------------------------------

SEMANTIC_WEIGHT = 0.40
SKILL_WEIGHT    = 0.60   # must sum to 1.0

# ---------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------

if uploaded_file is not None and job_description != "":

    # ---------------------------------------------------
    # EXTRACT RESUME TEXT
    # ---------------------------------------------------

    resume_text = extract_text_from_pdf(uploaded_file)

    # FIX 6: GUARD FOR IMAGE-BASED / EMPTY PDFs
    if not resume_text.strip():
        st.error(
            "No text could be extracted from this PDF. "
            "It may be image-based or scanned. "
            "Please upload a text-selectable PDF."
        )
        st.stop()

    resume_lower = resume_text.lower()

    jd_lower = job_description.lower()

    # ---------------------------------------------------
    # LOAD EMBEDDING MODEL (cached)
    # ---------------------------------------------------

    model = load_model()

    # ---------------------------------------------------
    # RAG CHUNKING
    # ---------------------------------------------------

    chunks = chunk_text(resume_text)

    # ---------------------------------------------------
    # CREATE EMBEDDINGS
    # ---------------------------------------------------

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings, dtype='float32')

    # ---------------------------------------------------
    # FIX 2: USE IndexFlatIP + normalise for cosine similarity
    # L2 index returns distances (lower = better), which is
    # the wrong direction for a similarity-based retrieval.
    # Inner product on normalised vectors = cosine similarity.
    # ---------------------------------------------------

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # ---------------------------------------------------
    # JOB DESCRIPTION EMBEDDING
    # ---------------------------------------------------

    query_embedding = model.encode([job_description])

    query_embedding = np.array(query_embedding, dtype='float32')

    # Normalise query too — required for IndexFlatIP
    faiss.normalize_L2(query_embedding)

    # ---------------------------------------------------
    # RETRIEVE TOP RELEVANT CHUNKS
    # ---------------------------------------------------

    distances, indices = index.search(
        query_embedding,
        k=3
    )

    retrieved_chunks = [chunks[idx] for idx in indices[0]]

    # ---------------------------------------------------
    # AI RECRUITER ANALYSIS
    # ---------------------------------------------------

    st.divider()

    st.subheader("🤖 AI Recruiter Analysis")

    for chunk in retrieved_chunks:

        st.info(chunk)

    # ---------------------------------------------------
    # SKILL DATABASE
    # ---------------------------------------------------

    skills = [

        # Programming
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "rust",
        "sql",

        # AI / ML
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "nlp",
        "rag",
        "langchain",
        "transformers",

        # Data Science
        "pandas",
        "numpy",
        "scikit-learn",
        "data analysis",

        # Web Development
        "react",
        "node.js",
        "django",
        "flask",
        "fastapi",
        "streamlit",

        # Cloud / DevOps
        "aws",
        "docker",
        "kubernetes",
        "terraform",
        "git",
        "github",

        # Databases
        "mongodb",
        "mysql",
        "postgresql",

        # Visualization
        "tableau",
        "power bi"
    ]

    # ---------------------------------------------------
    # DETECT FOUND + MISSING SKILLS
    # FIX 4 applied: uses skill_in_text() with aliases
    # ---------------------------------------------------

    found_skills    = []
    missing_skills  = []
    required_skills = []

    for skill in skills:

        skill_lower = skill.lower()

        has_in_resume = skill_in_text(resume_lower, skill_lower)
        has_in_jd     = skill_in_text(jd_lower, skill_lower)

        if has_in_resume and skill not in found_skills:
            found_skills.append(skill)

        if has_in_jd:
            if skill not in required_skills:
                required_skills.append(skill)
            if not has_in_resume and skill not in missing_skills:
                missing_skills.append(skill)

    # ---------------------------------------------------
    # SKILL MATCH RATIO
    # ---------------------------------------------------

    if required_skills:

        matched_skills = [
            skill for skill in required_skills
            if skill in found_skills
        ]

        skill_match_ratio = len(matched_skills) / len(required_skills)

    else:

        skill_match_ratio = 0

    # ---------------------------------------------------
    # SEMANTIC SIMILARITY SCORE
    # ---------------------------------------------------

    try:

        semantic_embeddings = model.encode(
            [resume_text, job_description]
        )

        text_similarity = cosine_similarity(
            [semantic_embeddings[0]],
            [semantic_embeddings[1]]
        )[0][0]

    except Exception:

        text_similarity = 0.0

    # ---------------------------------------------------
    # FIX 3 + FIX 5: FINAL ATS SCORE with named constants
    # ---------------------------------------------------

    final_score = (
        (text_similarity   * SEMANTIC_WEIGHT)
        + (skill_match_ratio * SKILL_WEIGHT)
    )

    final_score = round(final_score * 100, 2)

    final_score = max(0, min(100, final_score))

    # ---------------------------------------------------
    # ATS SCORE UI
    # ---------------------------------------------------

    st.divider()

    st.subheader("✅ ATS Match Score")

    st.progress(int(final_score))

    if final_score >= 75:

        st.success(
            f"{final_score}% Match — Excellent Resume Match 🚀"
        )

    elif final_score >= 50:

        st.warning(
            f"{final_score}% Match — Good Match, but can be improved."
        )

    else:

        st.error(
            f"{final_score}% Match — Low Match, add more relevant skills."
        )

    # ---------------------------------------------------
    # SKILLS UI
    # ---------------------------------------------------

    st.divider()

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # DETECTED SKILLS
    # ---------------------------------------------------

    with col1:

        st.subheader("🛠 Detected Skills")

        if found_skills:

            for skill in found_skills:
                st.write(f"✔️ {skill}")

        else:

            st.write("No matching skills detected.")

    # ---------------------------------------------------
    # MISSING SKILLS
    # ---------------------------------------------------

    with col2:

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:
                st.write(f"⚠️ {skill}")

        else:

            st.write("No missing skills detected.")

    # ---------------------------------------------------
    # ROLE PREDICTION
    # ---------------------------------------------------

    st.divider()

    st.subheader("🧠 Predicted Role")

    if (
        "tensorflow" in found_skills
        or "pytorch" in found_skills
        or "deep learning" in found_skills
    ):
        st.success("Machine Learning Engineer")

    elif (
        "machine learning" in found_skills
        or "pandas" in found_skills
    ):
        st.success("Data Scientist")

    elif (
        "tableau" in found_skills
        or "power bi" in found_skills
    ):
        st.success("Data Analyst")

    elif (
        "react" in found_skills
        or "node.js" in found_skills
    ):
        st.success("Full Stack Developer")

    else:
        st.success("Software Engineer")

    # ---------------------------------------------------
    # FIX 5: RESUME SUGGESTIONS — driven by missing_skills
    # instead of hardcoded skill checks that ignore the JD
    # ---------------------------------------------------

    st.divider()

    st.subheader("📌 Resume Improvement Suggestions")

    if final_score < 50:
        st.warning(
            "Add more technical skills relevant to the job description."
        )

    if missing_skills:
        for skill in missing_skills:
            st.write(
                f"- Add experience or projects demonstrating **{skill}**."
            )
    else:
        st.success(
            "Your resume covers all the key skills mentioned in the JD!"
        )

    # ---------------------------------------------------
    # FOOTER
    # ---------------------------------------------------

    st.divider()

    st.caption(
        """
        Built using Python, Streamlit,
        FAISS, Sentence Transformers,
        NLP, and RAG Architecture
        """
    )