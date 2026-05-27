
# 📄 AI Resume Screening System

<div align="center">

### AI-Powered ATS Resume Analyzer using NLP & Resume Intelligence

Analyze resumes intelligently using NLP-powered ATS scoring, skill extraction, role prediction, and resume optimization recommendations.

---

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-CountVectorizer-success?style=for-the-badge)

</div>

---

# 🔗 Project Links

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Adr005/AI-Resume-Screener)

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red?style=for-the-badge&logo=streamlit)](PASTE_YOUR_STREAMLIT_LINK_HERE)

---

# 🚀 Features

## 📊 ATS Match Score Analysis
- Calculates a highly accurate and balanced ATS compatibility score.
- Uses a **Hybrid Scoring Model** combining overall text structure cosine similarity (via `CountVectorizer`) with keyword skill-matching ratios.

---

## 🛠 Skill Extraction Engine
- Dynamically extracts skills using robust letter-boundary regex patterns.
- Supports **85+ technical, analytical, DevOps, design, and PM tools** (preventing false substring positives and fully supporting versioned tools like `HTML5`, `CSS3`, `Python3`).

---

## ❌ Missing Skills Detection
- Identifies important job-relevant skills absent from the resume.
- Helps candidates optimize resumes for ATS systems.

---

## 🧠 Resume Role Prediction
Predicts suitable technical roles such as:
- Data Scientist
- Machine Learning Engineer
- Data Analyst
- Software Engineer

based on detected skills.

---

## 📌 Resume Improvement Suggestions
Generates intelligent recommendations to improve resume quality and ATS compatibility.

---

# 🖥️ Dashboard Preview

The dashboard provides:

- ATS Score Visualization
- Resume Skill Analytics
- Missing Skills Dashboard
- Resume Intelligence Recommendations
- Role Prediction Engine
- Interactive Resume Analysis UI

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend Logic |
| Streamlit | Interactive Web Application |
| Scikit-learn | NLP & Similarity Analysis |
| CountVectorizer | Text Vectorization (prevents tiny-corpus IDF penalty) |
| Regex (Lookaround boundaries) | Precision Skill Extraction |
| Cosine Similarity | Overall Text Matching |
| pdfplumber | Resume Parsing |
| Pandas | Data Processing |

---

# 📂 Project Structure

```bash
AI-Resume-Screener/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

# 🧭 System Architecture

```text
                ┌─────────────────────┐
                │ Resume PDF Upload   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ PDF Text Extraction │
                │    (pdfplumber)     │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Job Description     │
                │ Input Processing    │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Count Vectorizer   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Hybrid ATS Score   │
                │   Engine (Cosine)   │
                └─────────┬───────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
 ┌─────────────┐ ┌────────────────┐ ┌─────────────────┐
 │ Skill       │ │ Missing Skills │ │ Role Prediction │
 │ Detection   │ │ Detection      │ │ Engine          │
 └──────┬──────┘ └────────┬───────┘ └────────┬────────┘
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ▼
                ┌─────────────────────┐
                │ Resume Suggestions  │
                │ & Dashboard Output  │
                └─────────────────────┘
```

---

# ⚙️ Workflow

## Step 1 — Resume Upload
User uploads resume PDF and enters target job description.

---

## Step 2 — Resume Parsing
Text is extracted from PDF using `pdfplumber`.

---

## Step 3 — NLP Processing
Count-based vectorization converts textual data into numerical vector representations, avoiding the IDF suppression typical of tiny 2-document corpora.

---

## Step 4 — ATS Analysis
A hybrid algorithm combines textual cosine similarity (40% weight) with keyword skill match ratio (60% weight) to calculate the final compatibility score.

---

## Step 5 — Resume Intelligence
The system:
- extracts technical skills using robust letter-boundary regex,
- identifies missing skills,
- predicts technical role,
- generates improvement suggestions.

---

# ✨ Core Functionalities

| Functionality | Description |
|---|---|
| ATS Scoring | Hybrid score combining Count cosine similarity (40%) and skill match ratio (60%) |
| Skill Extraction | Word-boundary scan covering 85+ standard tools |
| Missing Skills Analysis | Detects absent required skills |
| Role Prediction | Predicts technical domain |
| Resume Suggestions | Resume improvement guidance |
| Dashboard UI | Interactive analytics interface |

---

# 🎨 UI Features

- Wide responsive dashboard layout
- Sidebar navigation panel
- Two-column analytics interface
- Interactive progress indicators
- Intelligent recommendation sections
- Professional Streamlit styling

---

# ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 🌐 Live Demo
https://ai-resume-screener-himnznnruqezneyi6vzmaz.streamlit.app/
Launch the deployed Streamlit application using the badge above.

---
# 📸 Screenshots

## Dashboard Preview

![Dashboard Screenshot](images/image.png)



---

# 🔮 Future Improvements

- Transformer-based semantic matching
- Resume ranking system
- Multi-resume comparison
- Recruiter analytics dashboard
- LLM-powered resume optimization
- AI-generated interview preparation
- Semantic skill matching
- Resume scoring analytics

---

# 👨‍💻 Author

### Aadhya Rohatgi

B.Tech Data Science Student  
AI • ML • NLP • Data Science