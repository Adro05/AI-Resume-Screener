import re
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption(
    "Compare a resume with a job description using chunk-level semantic matching "
    "and transparent skill coverage. Scores are screening aids, not hiring decisions."
)


# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
SEMANTIC_WEIGHT = 0.40
SKILL_WEIGHT = 0.60
CHUNK_SIZE = 500          # characters; keeps chunks comfortably below model token limit
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------
# SKILL TAXONOMY + ALIASES
# Keep canonical skill names unique. Aliases map alternate spellings to a canonical skill.
# ---------------------------------------------------
SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql",
    "machine learning", "deep learning", "tensorflow", "pytorch", "nlp", "rag",
    "langchain", "transformers", "pandas", "numpy", "scikit-learn", "data analysis",
    "react", "node.js", "django", "flask", "fastapi", "streamlit", "aws", "docker",
    "kubernetes", "terraform", "git", "github", "mongodb", "mysql", "postgresql",
    "tableau", "power bi"
]

SKILL_ALIASES: Dict[str, List[str]] = {
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "c++": ["cpp", "c plus plus"],
    "c#": ["c sharp", "c-sharp"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "nlp": ["natural language processing"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "scikit-learn": ["scikit learn", "sklearn"],
    "node.js": ["nodejs", "node js"],
    "postgresql": ["postgres"],
    "aws": ["amazon web services"],
    "kubernetes": ["k8s"],
    "power bi": ["powerbi"],
    "github": ["git hub"],
}


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Match a phrase as a whole term, allowing flexible whitespace."""
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


@st.cache_data(show_spinner=False)
def _compiled_patterns():
    patterns = {}
    for skill in SKILLS:
        variants = [skill] + SKILL_ALIASES.get(skill, [])
        patterns[skill] = [_phrase_pattern(v) for v in variants]
    return patterns


def detect_skills(text: str) -> List[str]:
    patterns = _compiled_patterns()
    found = []
    for skill in SKILLS:
        if any(p.search(text or "") for p in patterns[skill]):
            found.append(skill)
    return found


# ---------------------------------------------------
# MODEL + TEXT UTILITIES
# ---------------------------------------------------
@st.cache_resource(show_spinner="Loading semantic model...")
def load_model():
    model = SentenceTransformer(MODEL_NAME)
    return model


def extract_text_from_pdf(pdf_file) -> str:
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def encode_normalized(model, texts: List[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------
# CHUNK-LEVEL SEMANTIC MATCHING
# For every JD chunk, find its best matching resume chunk.
# This avoids embedding/truncating an entire long document as one sequence.
# ---------------------------------------------------
def semantic_match(model, resume_chunks: List[str], jd_chunks: List[str]) -> Tuple[float, List[dict]]:
    if not resume_chunks or not jd_chunks:
        return 0.0, []

    resume_vectors = encode_normalized(model, resume_chunks)
    jd_vectors = encode_normalized(model, jd_chunks)

    dimension = resume_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # normalized inner product = cosine similarity
    index.add(resume_vectors)

    # For each JD chunk, retrieve the single best resume chunk.
    similarities, indices = index.search(jd_vectors, 1)
    best_scores = np.clip(similarities[:, 0], 0.0, 1.0)

    matches = []
    for jd_i, (score, resume_i) in enumerate(zip(best_scores, indices[:, 0])):
        if resume_i >= 0:
            matches.append({
                "score": float(score),
                "jd_chunk": jd_chunks[jd_i],
                "resume_chunk": resume_chunks[int(resume_i)],
                "resume_index": int(resume_i),
            })

    # Average best-match similarity across JD chunks; not a calibrated probability.
    semantic_score = float(np.mean(best_scores)) if len(best_scores) else 0.0

    # Show strongest distinct resume evidence chunks in the UI.
    matches.sort(key=lambda item: item["score"], reverse=True)
    top_matches = []
    seen_resume_chunks = set()
    for item in matches:
        if item["resume_index"] not in seen_resume_chunks:
            top_matches.append(item)
            seen_resume_chunks.add(item["resume_index"])
        if len(top_matches) == TOP_K:
            break

    return semantic_score, top_matches


# ---------------------------------------------------
# UI INPUTS
# ---------------------------------------------------
with st.sidebar:
    st.header("How scoring works")
    st.write("• 40% chunk-level semantic relevance")
    st.write("• 60% detected required-skill coverage")
    st.caption(
        "If the job description contains no skills from the supported taxonomy, "
        "the app uses semantic relevance alone rather than assigning a false zero skill score."
    )

uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the complete job description", height=240)

if st.button("Analyze resume", type="primary", disabled=not (uploaded_file and job_description.strip())):
    try:
        with st.spinner("Extracting and analyzing resume..."):
            resume_text = clean_text(extract_text_from_pdf(uploaded_file))
            jd_text = clean_text(job_description)

            if len(resume_text) < 40:
                st.error(
                    "I couldn't extract enough text from this PDF. It may be scanned/image-based. "
                    "Please upload a text-selectable PDF."
                )
                st.stop()
            if len(jd_text) < 80:
                st.warning("The job description is very short. Add the responsibilities and requirements for a more useful comparison.")

            resume_chunks = chunk_text(resume_text)
            jd_chunks = chunk_text(jd_text)
            if not resume_chunks or not jd_chunks:
                st.error("Not enough text was available to perform semantic matching.")
                st.stop()

            model = load_model()
            semantic_score, top_matches = semantic_match(model, resume_chunks, jd_chunks)

            resume_skills = detect_skills(resume_text)
            jd_skills = detect_skills(jd_text)
            matched_skills = [s for s in jd_skills if s in resume_skills]
            missing_skills = [s for s in jd_skills if s not in resume_skills]

            skill_score: Optional[float]
            if jd_skills:
                skill_score = len(matched_skills) / len(jd_skills)
                final_score = (
                    SEMANTIC_WEIGHT * semantic_score
                    + SKILL_WEIGHT * skill_score
                ) * 100
                scoring_note = "Hybrid score: 40% semantic relevance + 60% detected skill coverage."
            else:
                skill_score = None
                final_score = semantic_score * 100
                scoring_note = (
                    "No supported taxonomy skills were detected in the job description. "
                    "The score therefore uses semantic relevance alone; no zero skill penalty was applied."
                )

            final_score = round(float(np.clip(final_score, 0, 100)), 1)
            semantic_percent = round(semantic_score * 100, 1)
            skill_percent = round(skill_score * 100, 1) if skill_score is not None else None

        # ---------------------------------------------------
        # RESULTS
        # ---------------------------------------------------
        st.divider()
        st.subheader("ATS Match Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall match", f"{final_score}%")
        c2.metric("Semantic relevance", f"{semantic_percent}%")
        c3.metric("Skill coverage", f"{skill_percent}%" if skill_percent is not None else "N/A")
        st.progress(int(round(final_score)))
        st.caption(scoring_note)
        st.caption(
            "Interpretation: this is a heuristic resume-to-JD similarity score, not the probability "
            "of getting shortlisted and not a universally standardized ATS score."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Skills detected in resume")
            if resume_skills:
                st.write(" · ".join(resume_skills))
            else:
                st.info("No skills from the current 38-skill taxonomy were detected. This does not mean the resume has no skills.")
        with right:
            st.subheader("Required skills found in job description")
            if jd_skills:
                st.write(" · ".join(jd_skills))
            else:
                st.info("No supported taxonomy skills were detected in this job description.")

        st.subheader("Matched skills")
        if matched_skills:
            st.success(" · ".join(matched_skills))
        else:
            st.write("No exact canonical skill matches were detected from the supported taxonomy.")

        st.subheader("Potential missing skills")
        if missing_skills:
            st.warning(" · ".join(missing_skills))
            st.caption("Check these against the candidate's actual experience before recommending changes.")
        else:
            st.success("No missing skills from the supported taxonomy were detected.")

        st.divider()
        st.subheader("Most relevant resume evidence")
        if top_matches:
            for i, item in enumerate(top_matches, start=1):
                with st.expander(f"Match {i} · semantic similarity {item['score'] * 100:.1f}%", expanded=(i == 1)):
                    st.markdown("**Relevant job-description passage**")
                    st.write(item["jd_chunk"])
                    st.markdown("**Matching resume passage**")
                    st.write(item["resume_chunk"])
        else:
            st.info("No relevant resume passages could be retrieved.")

        st.divider()
        st.subheader("Suggested next steps")
        if missing_skills:
            st.write("If you genuinely have experience with these requirements, make that experience easier to find in your resume:")
            for skill in missing_skills:
                st.write(f"• **{skill}** — add a truthful project, coursework or work-experience example if applicable.")
        else:
            st.write("Review whether your strongest projects and achievements clearly demonstrate the requirements in the job description.")

        st.subheader("Role indication")
        if any(s in resume_skills for s in ["tensorflow", "pytorch", "deep learning", "machine learning"]):
            st.write("Possible direction: Machine Learning / AI roles")
        elif any(s in resume_skills for s in ["pandas", "numpy", "tableau", "power bi", "data analysis"]):
            st.write("Possible direction: Data Science / Data Analytics roles")
        elif any(s in resume_skills for s in ["react", "node.js", "django", "flask", "fastapi"]):
            st.write("Possible direction: Software / Web Development roles")
        else:
            st.write("Possible direction: General Software Engineering; review the resume manually for a more specific role.")

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.info("If this persists, check the PDF, installed dependencies and model download/network access.")

st.divider()
st.caption(
    "Built with Python, Streamlit, pdfplumber, Sentence-Transformers, FAISS, "
    "LangChain text splitting, NumPy and scikit-learn."
)
