import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener")

st.title("📄 AI Resume Screening System")

st.markdown("### Upload Resume and Compare with Job Description")

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

    text_data = [resume_text, job_description]

    tfidf = TfidfVectorizer(stop_words='english')

    matrix = tfidf.fit_transform(text_data)

    similarity_score = cosine_similarity(matrix)[0][1]

    st.subheader("✅ ATS Match Score")

    score = round(similarity_score * 100, 2)

    st.progress(int(score))

    st.success(f"{score} % Match")
    if score >= 75:
        st.info("Excellent Resume Match 🚀")

    elif score >= 50:
        st.warning("Good Match — Resume can be improved.")

    else:
        st.error("Low Match — Add more relevant skills.")

    skills = [
        "python",
        "machine learning",
        "sql",
        "pandas",
        "numpy",
        "tensorflow",
        "deep learning",
        "data analysis",
        "power bi",
        "tableau",
        "flask",
        "streamlit"
    ]

    found_skills = []
    missing_skills = []

    for skill in skills:

        if skill.lower() in resume_text.lower():
            found_skills.append(skill)

        elif skill.lower() in job_description.lower():
            missing_skills.append(skill)

    st.subheader("🛠 Detected Skills")

    if found_skills:
        for skill in found_skills:
            st.write(f"✔️ {skill}")

    else:
        st.write("No matching skills detected.")
    st.subheader("❌ Missing Skills")

    if missing_skills:

        for skill in missing_skills:
            st.write(f"⚠️ {skill}")

    else:
        st.write("No missing skills detected.")
    
    
    st.subheader("🧠 Predicted Role")

    if "tensorflow" in found_skills or "deep learning" in found_skills:
        st.success("Machine Learning Engineer")

    elif "machine learning" in found_skills:
        st.success("Data Scientist")

    elif "tableau" in found_skills or "power bi" in found_skills:
        st.success("Data Analyst")

    else:
        st.success("Software Engineer")
    
    st.subheader("📌 Resume Improvement Suggestions")

if score < 50:
    st.warning("Add more technical skills relevant to the job description.")

if "sql" not in found_skills:
    st.write("- Add SQL projects or certifications.")

if "machine learning" not in found_skills:
    st.write("- Include machine learning experience.")

if "python" not in found_skills:
    st.write("- Highlight Python projects.")