import re
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screening System")
st.caption(
    "Evidence-aware resume-to-job matching. Scores are transparent screening aids, "
    "not hiring probabilities or standardized ATS scores."
)

# --------------------------- Configuration ---------------------------
SEMANTIC_WEIGHT = 0.40
SKILL_WEIGHT = 0.60
REQUIRED_WEIGHT = 0.80       # when both required and preferred skills exist
PREFERRED_WEIGHT = 0.20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MODEL_NAME = "all-MiniLM-L6-v2"

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql",
    "machine learning", "deep learning", "tensorflow", "pytorch", "nlp", "rag",
    "langchain", "transformers", "llm", "faiss", "rest api", "vector database",
    "pandas", "numpy", "scikit-learn", "data analysis",
    "react", "node.js", "django", "flask", "fastapi", "streamlit", "aws", "docker",
    "kubernetes", "terraform", "git", "github", "mongodb", "mysql", "postgresql",
    "tableau", "power bi"
]

SKILL_ALIASES = {
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "c++": ["cpp", "c plus plus"],
    "c#": ["c sharp", "c-sharp"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "nlp": ["natural language processing"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "llm": ["llms", "large language model", "large language models"],
    "rest api": ["rest apis", "restful api", "restful apis"],
    "vector database": ["vector databases", "vector db", "vector store", "vector stores"],
    "scikit-learn": ["scikit learn", "sklearn"],
    "node.js": ["nodejs", "node js"],
    "postgresql": ["postgres"],
    "aws": ["amazon web services"],
    "kubernetes": ["k8s"],
    "power bi": ["powerbi"],
    "github": ["git hub"],
}

# These are evidence clues, not automatic proof of a skill. They are shown to the user.
RELATED_EVIDENCE = {
    "machine learning": [
        r"\bisolation\s+forest\b", r"\brandom\s+forest\b", r"\bxgboost\b",
        r"\blogistic\s+regression\b", r"\bdecision\s+tree(s)?\b",
        r"\bk-?nearest\s+neighbors?\b", r"\bknn\b", r"\bk-means\b",
        r"\bgradient\s+boost(ing)?\b", r"\bclassifier\b", r"\bregression\s+model\b"
    ],
    "deep learning": [
        r"\bconvolutional\s+neural\s+network(s)?\b", r"\bcnn(s)?\b",
        r"\brecurrent\s+neural\s+network(s)?\b", r"\brnn(s)?\b",
        r"\blstm\b", r"\btransformer\s+model(s)?\b", r"\bneural\s+network(s)?\b",
        r"\btrained\s+(a\s+)?(deep\s+)?neural\s+network\b"
    ]
}

SKILLS_HEADING = re.compile(r"^\s*(technical\s+skills|skills|core\s+competencies|competencies|technologies)\s*:?\s*$", re.I)
PROJECT_OR_EXPERIENCE_HEADING = re.compile(
    r"^\s*(projects?|experience|work\s+experience|internships?|professional\s+experience|education|coursework)\s*:?\s*$", re.I
)
PREFERRED_HEADING = re.compile(r"\b(preferred|nice\s+to\s+have|good\s+to\s+have|bonus|desirable)\b", re.I)
REQUIRED_HEADING = re.compile(r"\b(required|must\s+have|minimum\s+qualifications|required\s+skills)\b", re.I)


def phrase_pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.I)


@st.cache_data(show_spinner=False)
def compiled_patterns():
    return {
        skill: [phrase_pattern(x) for x in [skill] + SKILL_ALIASES.get(skill, [])]
        for skill in SKILLS
    }


def skill_found(text: str, skill: str) -> bool:
    return any(p.search(text or "") for p in compiled_patterns()[skill])


def detect_skills(text: str) -> List[str]:
    return [s for s in SKILLS if skill_found(text, s)]


def extract_jd_skill_priorities(jd_text: str) -> Tuple[List[str], List[str]]:
    """Classify skill mentions by nearby JD section heading. Required wins on duplicates.
    Skills outside a preferred section default to required so requirements in responsibilities
    are not silently ignored.
    """
    required, preferred = [], []
    section = "required"
    for line in jd_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading_text = re.sub(r"^#+\s*", "", stripped).strip().rstrip(":").strip()
        is_section_heading = (
            stripped.startswith("#")
            or bool(re.match(r"^[A-Za-z][A-Za-z &/()_-]{1,45}:$", stripped))
            or heading_text.lower() in {
                "about the role", "responsibilities", "what you will do", "what you'll do",
                "requirements", "qualifications", "minimum qualifications", "what we're looking for",
                "required skills", "preferred skills", "nice to have", "benefits"
            }
        )
        if PREFERRED_HEADING.search(heading_text) and is_section_heading:
            section = "preferred"
        elif (REQUIRED_HEADING.search(heading_text) or is_section_heading):
            section = "required"
        for skill in detect_skills(stripped):
            target = required if section == "required" else preferred
            if skill not in target:
                target.append(skill)
    # If a skill occurs in both groups, treat it as required.
    preferred = [s for s in preferred if s not in required]
    return required, preferred


def resume_skill_evidence(resume_text: str) -> Dict[str, dict]:
    """Return transparent evidence strength for each skill.
    A mention in a technical-skills inventory gets partial credit; a mention in project,
    experience or coursework text gets full credit. Related algorithm/framework clues are
    reported separately and never mislabeled as an exact skill mention.
    """
    lines = resume_text.splitlines()
    in_skills_section = False
    skill_lines: Dict[str, List[Tuple[str, bool]]] = {s: [] for s in SKILLS}
    all_non_skill_text = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if SKILLS_HEADING.match(stripped):
            in_skills_section = True
            # Heading may contain actual skill text after a colon.
            remainder = re.sub(r"^\s*(technical\s+skills|skills|core\s+competencies|competencies|technologies)\s*:?\s*", "", stripped, flags=re.I)
            if not remainder:
                continue
            stripped = remainder
        elif PROJECT_OR_EXPERIENCE_HEADING.match(stripped):
            in_skills_section = False
        elif re.match(r"^\s*[A-Z][A-Za-z &/()-]{2,35}:\s*$", stripped):
            # Other section headings such as Certifications can end the skills inventory.
            in_skills_section = False

        if not in_skills_section:
            all_non_skill_text.append(stripped)
        for skill in detect_skills(stripped):
            skill_lines[skill].append((stripped, in_skills_section))

    non_skill_text = "\n".join(all_non_skill_text)
    result: Dict[str, dict] = {}
    for skill, mentions in skill_lines.items():
        if not mentions:
            continue
        contextual_mentions = [line for line, is_skill_list in mentions if not is_skill_list]
        # A repository link printed beside a project title is not proof of GitHub workflow skill.
        if skill == "github":
            # Count GitHub as demonstrated only when the resume describes actual workflow,
            # not merely a profile URL or a repository link beside a project title.
            contextual_mentions = [
                line for line in contextual_mentions
                if re.search(
                    r"\b(version control|git workflow|pull request|commit|branch|"
                    r"collaborat(?:e|ed|ing).{0,40}github|github.{0,40}(?:collaborat|repository|pull request))\b",
                    line, re.I
                )
            ]
        if contextual_mentions:
            result[skill] = {
                "credit": 1.0,
                "status": "Project/experience/coursework evidence",
                "evidence": contextual_mentions[:2],
            }
        else:
            result[skill] = {
                "credit": 0.65,
                "status": "Listed in technical skills only",
                "evidence": [line for line, _ in mentions[:2]],
            }

    # Related evidence rules are deliberately conservative and visible.
    for skill, rules in RELATED_EVIDENCE.items():
        if skill in result:
            continue
        found = []
        for line in all_non_skill_text:
            if any(re.search(rule, line, re.I) for rule in rules):
                found.append(line)
        if found:
            result[skill] = {
                "credit": 0.85 if skill == "machine learning" else 0.80,
                "status": "Related project evidence (inferred; verify manually)",
                "evidence": found[:2],
            }

    # Framework names alone are only weak supporting evidence for deep learning.
    if "deep learning" not in result:
        framework_mentions = []
        for line in lines:
            if skill_found(line, "tensorflow") or skill_found(line, "pytorch"):
                framework_mentions.append(line.strip())
        if framework_mentions:
            result["deep learning"] = {
                "credit": 0.35,
                "status": "Related framework mention only; no DL project evidence found",
                "evidence": framework_mentions[:2],
            }
    return result


@st.cache_resource(show_spinner="Loading semantic model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


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
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def encode_normalized(model, texts: List[str]) -> np.ndarray:
    vectors = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=32
    )
    return np.asarray(vectors, dtype=np.float32)


def semantic_match(model, resume_chunks: List[str], jd_chunks: List[str]) -> Tuple[float, List[dict]]:
    if not resume_chunks or not jd_chunks:
        return 0.0, []
    resume_vectors = encode_normalized(model, resume_chunks)
    jd_vectors = encode_normalized(model, jd_chunks)
    index = faiss.IndexFlatIP(resume_vectors.shape[1])
    index.add(resume_vectors)
    similarities, indices = index.search(jd_vectors, 1)
    best_scores = np.clip(similarities[:, 0], 0.0, 1.0)
    matches = []
    for score, resume_i, jd_chunk in zip(best_scores, indices[:, 0], jd_chunks):
        if resume_i >= 0:
            matches.append({
                "score": float(score), "jd_chunk": jd_chunk,
                "resume_chunk": resume_chunks[int(resume_i)], "resume_index": int(resume_i)
            })
    matches.sort(key=lambda x: x["score"], reverse=True)
    top, seen = [], set()
    for item in matches:
        if item["resume_index"] not in seen:
            top.append(item)
            seen.add(item["resume_index"])
        if len(top) >= TOP_K:
            break
    return float(np.mean(best_scores)), top


def weighted_skill_coverage(required: List[str], preferred: List[str], evidence: Dict[str, dict]) -> Tuple[Optional[float], dict]:
    def group_score(items):
        if not items:
            return None
        return float(np.mean([evidence.get(s, {}).get("credit", 0.0) for s in items]))

    req_score = group_score(required)
    pref_score = group_score(preferred)
    if req_score is not None and pref_score is not None:
        combined = REQUIRED_WEIGHT * req_score + PREFERRED_WEIGHT * pref_score
    elif req_score is not None:
        combined = req_score
    elif pref_score is not None:
        combined = pref_score
    else:
        combined = None
    return combined, {"required": req_score, "preferred": pref_score}


# --------------------------- UI ---------------------------
with st.sidebar:
    st.header("How scoring works")
    st.write("• 40% chunk-level semantic relevance")
    st.write("• 60% evidence-aware skill coverage")
    st.write("• Within skills: 80% required + 20% preferred, when both are present")
    st.caption("A project/experience mention receives full evidence credit. A skills-list-only mention receives partial credit. Related evidence is labeled as inferred and should be checked manually.")

uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the complete job description", height=260)

if st.button("Analyze resume", type="primary", disabled=not (uploaded_file and job_description.strip())):
    try:
        with st.spinner("Extracting resume and evaluating evidence..."):
            resume_text = clean_text(extract_text_from_pdf(uploaded_file))
            jd_text = clean_text(job_description)
            if len(resume_text) < 40:
                st.error("I couldn't extract enough text from this PDF. It may be scanned/image-based. Please upload a text-selectable PDF.")
                st.stop()
            if len(jd_text) < 80:
                st.warning("The job description is very short. Add responsibilities and requirements for a more useful comparison.")

            resume_chunks, jd_chunks = chunk_text(resume_text), chunk_text(jd_text)
            if not resume_chunks or not jd_chunks:
                st.error("Not enough text was available to perform semantic matching.")
                st.stop()
            model = load_model()
            semantic_score, top_matches = semantic_match(model, resume_chunks, jd_chunks)

            required_skills, preferred_skills = extract_jd_skill_priorities(jd_text)
            resume_evidence = resume_skill_evidence(resume_text)
            all_jd_skills = required_skills + preferred_skills
            skill_score, group_scores = weighted_skill_coverage(required_skills, preferred_skills, resume_evidence)

            if skill_score is None:
                final_score = semantic_score * 100
                scoring_note = "No supported skills were detected. Overall score uses semantic relevance alone."
            else:
                final_score = (SEMANTIC_WEIGHT * semantic_score + SKILL_WEIGHT * skill_score) * 100
                scoring_note = "Overall = 40% semantic relevance + 60% evidence-aware skill coverage. Skill coverage = 80% required + 20% preferred when both groups exist."
            final_score = round(float(np.clip(final_score, 0, 100)), 1)
            semantic_percent = round(semantic_score * 100, 1)
            skill_percent = round(skill_score * 100, 1) if skill_score is not None else None

        st.divider()
        st.subheader("ATS Match Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall match", f"{final_score}%")
        c2.metric("Semantic relevance", f"{semantic_percent}%")
        c3.metric("Evidence-aware skill coverage", f"{skill_percent}%" if skill_percent is not None else "N/A")
        st.progress(int(round(final_score)))
        st.caption(scoring_note)
        st.caption("This is a heuristic comparison score, not a hiring probability or a universally standardized ATS score.")

        st.divider()
        st.subheader("Skills detected in resume")
        if resume_evidence:
            st.write(" · ".join(sorted(resume_evidence.keys())))
        else:
            st.info("No supported skills from the current taxonomy were detected.")

        st.subheader("Job-description skill priorities")
        left, right = st.columns(2)
        with left:
            st.markdown("**Required skills**")
            st.write(" · ".join(required_skills) if required_skills else "None detected")
            if group_scores["required"] is not None:
                st.caption(f"Required evidence coverage: {group_scores['required'] * 100:.1f}%")
        with right:
            st.markdown("**Preferred skills**")
            st.write(" · ".join(preferred_skills) if preferred_skills else "None detected")
            if group_scores["preferred"] is not None:
                st.caption(f"Preferred evidence coverage: {group_scores['preferred'] * 100:.1f}%")

        st.subheader("Skill-by-skill evidence review")
        for priority, items in [("Required", required_skills), ("Preferred", preferred_skills)]:
            if not items:
                continue
            st.markdown(f"**{priority}**")
            for skill in items:
                info = resume_evidence.get(skill)
                if info:
                    st.write(f"✅ **{skill}** — {info['status']} (credit: {info['credit'] * 100:.0f}%)")
                    with st.expander(f"View resume evidence for {skill}"):
                        for line in info["evidence"]:
                            st.write(f"- {line}")
                else:
                    st.write(f"⚠️ **{skill}** — no direct or supported related evidence detected")

        st.caption("Related evidence is not the same as an explicit skill claim. Verify it manually before treating it as proficiency.")

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
        unsupported = [s for s in all_jd_skills if s not in resume_evidence]
        inferred = [s for s in all_jd_skills if s in resume_evidence and "inferred" in resume_evidence[s]["status"].lower()]
        if unsupported:
            st.write("If you genuinely have this experience, add a truthful project, coursework or work-experience example:")
            st.write(" · ".join(unsupported))
        if inferred:
            st.write("These skills have related evidence but need clearer explicit proof:")
            st.write(" · ".join(inferred))
        if not unsupported and not inferred:
            st.success("Every detected JD skill has direct resume evidence. Review the evidence excerpts for accuracy.")

        st.subheader("Role indication")
        if any(s in resume_evidence for s in ["tensorflow", "pytorch", "deep learning", "machine learning", "nlp"]):
            st.write("Possible direction: Machine Learning / AI roles")
        elif any(s in resume_evidence for s in ["pandas", "numpy", "tableau", "power bi", "data analysis"]):
            st.write("Possible direction: Data Science / Data Analytics roles")
        elif any(s in resume_evidence for s in ["react", "node.js", "django", "flask", "fastapi"]):
            st.write("Possible direction: Software / Web Development roles")
        else:
            st.write("Possible direction: General Software Engineering; review the resume manually for a more specific role.")

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.info("If this persists, check the PDF, installed dependencies and model download/network access.")

st.divider()
st.caption("Built with Python, Streamlit, pdfplumber, Sentence-Transformers, FAISS, LangChain text splitting and NumPy.")
