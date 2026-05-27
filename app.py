import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide"
)

def contains_skill(text, skill):
    escaped_skill = re.escape(skill)
    # Use negative lookbehind and lookahead to assert that the skill is not adjacent to other letters.
    # This prevents false substring matches (e.g., 'r' in 'programmer', 'go' in 'ongoing')
    # while perfectly allowing matches next to numbers or symbols (e.g., 'html' in 'html5', 'python' in 'python3').
    pattern = r'(?<![a-zA-Z])' + escaped_skill + r'(?![a-zA-Z])'
    return bool(re.search(pattern, text, re.IGNORECASE))


st.sidebar.title("AI Resume Screener")

st.sidebar.info(
    """
    Upload your resume and compare it with a job description using AI-powered ATS analysis.
    """
)

st.sidebar.markdown("### Features")
st.sidebar.write("- ATS Match Score")
st.sidebar.write("- Skill Detection")
st.sidebar.write("- Missing Skills")
st.sidebar.write("- Resume Role Prediction")
st.sidebar.write("- Resume Suggestions")

st.title("📄 AI Resume Screening System")

st.markdown(
    """
    Analyze resumes intelligently using NLP-powered ATS scoring, 
    skill detection, and resume optimization suggestions.
    """
)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type="pdf"
)

job_description = st.text_area(
    "Paste Job Description Here"
)

def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text

if uploaded_file is not None and job_description != "":

    resume_text = extract_text_from_pdf(uploaded_file)

    # 1. Comprehensive list of standard technical, analytical, design, and product skills
    skills = [
        # Programming Languages
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin", "r", "html", "css", "sql", "bash",
        # ML / AI / Data Science
        "machine learning", "deep learning", "artificial intelligence", "natural language processing", "nlp", "computer vision", 
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "opencv", "nltk", "spacy",
        # Data Analysis & BI
        "data analysis", "tableau", "power bi", "excel", "snowflake", "bigquery", "redshift", "looker",
        # Web Frameworks & Libraries
        "react", "angular", "vue", "node.js", "node", "express", "django", "flask", "fastapi", "spring boot", "spring", "ruby on rails", "rails", "laravel", "streamlit",
        # Databases
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "sqlite", "oracle",
        # Cloud / DevOps / Version Control
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab", "jenkins", "terraform", "ansible", "ci/cd",
        # Design & Product Management
        "figma", "photoshop", "illustrator", "sketch", "ui/ux", "product management", "agile", "scrum", "jira"
    ]

    # 2. Extract found and missing skills using robust regex word boundaries
    found_skills = []
    missing_skills = []
    required_skills = []

    for skill in skills:
        has_in_resume = contains_skill(resume_text, skill)
        has_in_jd = contains_skill(job_description, skill)

        if has_in_resume:
            found_skills.append(skill)

        if has_in_jd:
            required_skills.append(skill)
            if not has_in_resume:
                missing_skills.append(skill)

    # 3. Calculate Text Similarity Score (using CountVectorizer to avoid 2-doc IDF suppression)
    try:
        count_vect = CountVectorizer(stop_words='english')
        matrix = count_vect.fit_transform([resume_text, job_description])
        text_similarity = cosine_similarity(matrix)[0][1]
    except Exception:
        text_similarity = 0.0

    # 4. Calculate Skill Match Ratio
    if required_skills:
        skill_match_ratio = len([s for s in required_skills if s in found_skills]) / len(required_skills)
    else:
        # Default to resume's skill footprint if the JD mentions none of the recognized skills
        skill_match_ratio = len(found_skills) / len(skills) if skills else 1.0

    # 5. Compute Hybrid ATS Match Score (40% Text Similarity + 60% Skill Match Ratio)
    raw_score = (text_similarity * 0.40) + (skill_match_ratio * 0.60)
    score = round(raw_score * 100, 2)

    # Ensure score is within valid Streamlit progress bar bounds [0.0, 100.0]
    score = max(0.0, min(100.0, score))

    st.divider()
    st.subheader("✅ ATS Match Score")

    st.progress(int(score))

    if score >= 75:
        st.success(f"{score}% Match — Excellent Resume Match 🚀")

    elif score >= 50:
        st.warning(f"{score}% Match — Good Match, but can be improved.")

    else:
        st.error(f"{score}% Match — Low Match, add more relevant skills.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🛠 Detected Skills")

        if found_skills:

            for skill in found_skills:
                st.write(f"✔️ {skill}")

        else:
            st.write("No matching skills detected.")

    with col2:

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:
                st.write(f"⚠️ {skill}")

        else:
            st.write("No missing skills detected.")

    st.divider()

    st.subheader("🧠 Predicted Role")

    if "tensorflow" in found_skills or "deep learning" in found_skills:
        st.success("Machine Learning Engineer")

    elif "machine learning" in found_skills:
        st.success("Data Scientist")

    elif "tableau" in found_skills or "power bi" in found_skills:
        st.success("Data Analyst")

    else:
        st.success("Software Engineer")

    st.divider()

    st.subheader("📌 Resume Improvement Suggestions")

    if score < 50:
        st.warning("Add more technical skills relevant to the job description.")

    if "sql" not in found_skills:
        st.write("- Add SQL projects or certifications.")

    if "machine learning" not in found_skills:
        st.write("- Include machine learning experience.")

    if "python" not in found_skills:
        st.write("- Highlight Python projects.")
    st.divider()

    st.caption(
        "Built using Python, Streamlit, NLP, and Scikit-learn"
    )