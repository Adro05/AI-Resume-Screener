
# 📄 AI Resume Screening System

<div align="center">

### AI-Powered ATS Resume Analyzer using NLP, Semantic Search & RAG Architecture

Analyze resumes intelligently using semantic embeddings, FAISS vector retrieval, AI-powered ATS scoring, recruiter-style resume analysis, and resume optimization workflows.

---

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge\&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=for-the-badge)
![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-NLP-success?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-AI_Retrieval-purple?style=for-the-badge)

</div>

---

# 🚀 Features

## ✅ AI-Powered ATS Match Score

* Calculates ATS compatibility between resume and job description.
* Uses:

  * Semantic similarity analysis
  * Skill-match evaluation
  * Hybrid weighted ATS scoring
* Score Formula:

  * 40% Semantic Similarity
  * 60% Skill Match Ratio

---

## 🤖 RAG-Based Recruiter Analysis

* Implements Retrieval-Augmented Generation (RAG) architecture.
* Splits resumes into semantic chunks using LangChain text splitters.
* Stores embeddings inside a FAISS vector database.
* Retrieves recruiter-relevant resume sections using semantic search.

---

## 🧠 Semantic Resume Intelligence

* Uses Sentence-Transformers (`all-MiniLM-L6-v2`) for semantic embeddings.
* Performs contextual resume-job matching beyond keyword comparison.
* Improves ATS scoring accuracy for modern AI hiring workflows.

---

## 🛠 Intelligent Skill Detection

* Detects technical skills using:

  * Regex-based word-boundary matching
  * Skill alias normalization
  * NLP-driven parsing
* Supports:

  * Python
  * Java
  * SQL
  * Machine Learning
  * TensorFlow
  * PyTorch
  * React
  * Docker
  * AWS
  * and more.

---

## ❌ Missing Skill Detection

* Identifies skills present in the Job Description but missing from the resume.
* Generates recruiter-style improvement suggestions.

---

## 🧩 AI Role Prediction Engine

Predicts suitable technical roles such as:

* Machine Learning Engineer
* Data Scientist
* Data Analyst
* Full Stack Developer
* Software Engineer

based on detected skills.

---

## 📌 Resume Optimization Suggestions

* Generates actionable ATS improvement recommendations.
* Suggests missing skills and project improvements.
* Helps optimize resumes for AI-driven hiring systems.

---

# 🧠 RAG Workflow Architecture

```text
                           ┌──────────────────────┐
                           │  Upload Resume PDF   │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ PDF Text Extraction  │
                           │    (pdfplumber)      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Resume Preprocessing │
                           │   & Text Cleaning    │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Resume Text Chunking │
                           │ (LangChain Splitter) │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │ Semantic Embedding   │
                           │ SentenceTransformers │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  FAISS Vector Store  │
                           │   Vector Indexing    │
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          ┌──────────────────────┐         ┌──────────────────────┐
          │ Job Description      │         │ Skill Detection      │
          │ Semantic Embedding   │         │ Regex + NLP Engine   │
          └──────────┬───────────┘         └──────────┬───────────┘
                     │                                │
                     ▼                                ▼
          ┌──────────────────────┐         ┌──────────────────────┐
          │ Semantic Similarity  │         │ Found Skills         │
          │ Retrieval Search     │         │ Missing Skills       │
          └──────────┬───────────┘         │ Required Skills      │
                     │                     └──────────┬───────────┘
                     ▼                                │
          ┌──────────────────────┐                   │
          │ Top Relevant Resume  │                   │
          │ Chunks Retrieved     │                   │
          └──────────┬───────────┘                   │
                     └──────────────┬────────────────┘
                                    ▼
                     ┌────────────────────────────┐
                     │ Hybrid ATS Scoring Engine  │
                     │ 40% Semantic Similarity    │
                     │ 60% Skill Match Ratio      │
                     └──────────┬─────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────┐
                │ AI Resume Analysis Dashboard    │
                │                                 │
                │ • ATS Match Score               │
                │ • RAG Recruiter Analysis        │
                │ • Skill Detection               │
                │ • Missing Skills Analysis       │
                │ • Role Prediction               │
                │ • Resume Suggestions            │
                └─────────────────────────────────┘
```

---

# 🖥️ Dashboard Preview

The application dashboard provides:

* ATS Match Score Visualization
* Semantic Resume Analysis
* RAG Resume Retrieval
* Skill Detection Dashboard
* Missing Skills Analysis
* Role Prediction Engine
* Resume Optimization Suggestions
* Interactive Streamlit Interface

---

# ⚙️ Tech Stack

| Technology            | Purpose                     |
| --------------------- | --------------------------- |
| Python                | Core Backend Logic          |
| Streamlit             | Interactive Web Application |
| Sentence-Transformers | Semantic Embeddings         |
| FAISS                 | Vector Similarity Search    |
| LangChain             | Resume Chunking             |
| Scikit-learn          | Similarity Analysis         |
| NLP                   | Resume Intelligence         |
| pdfplumber            | PDF Text Extraction         |
| NumPy                 | Vector Operations           |
| Regex                 | Skill Detection             |

---

# 📂 Project Structure

```bash
AI-Resume-Screener/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── images/
│   └── image.png
└── venv/
```

---

# 🧪 Core AI Pipeline

## 1️⃣ Resume Parsing

Extracts text from uploaded resumes using `pdfplumber`.

## 2️⃣ Resume Chunking

Splits resume text into semantic chunks using LangChain.

## 3️⃣ Embedding Generation

Generates semantic vector embeddings using Sentence-Transformers.

## 4️⃣ FAISS Vector Indexing

Stores embeddings in a FAISS vector database for semantic retrieval.

## 5️⃣ Semantic Retrieval

Retrieves recruiter-relevant resume chunks based on job description context.

## 6️⃣ Hybrid ATS Scoring

Combines:

* Semantic similarity analysis
* Skill overlap evaluation

## 7️⃣ Resume Intelligence

Performs:

* Skill detection
* Missing skill analysis
* Role prediction
* Resume optimization

---

# 📸 Screenshots

## Dashboard Preview

```md
![Dashboard Screenshot](images/image.png)
```

---

# ▶️ Run Locally

```bash
git clone https://github.com/Adro05/AI-Resume-Screener.git

cd AI-Resume-Screener

pip install -r requirements.txt

streamlit run app.py
```

---

# 🌐 Live Demo

### Streamlit App

[https://ai-resume-screener-himnznnruqezneyi6vzmaz.streamlit.app/](https://ai-resume-screener-himnznnruqezneyi6vzmaz.streamlit.app/)

---

# 🔗 Repository

### GitHub Repository

[https://github.com/Adro05/AI-Resume-Screener](https://github.com/Adro05/AI-Resume-Screener)

---

# 📦 Requirements

Dependencies used in the project include:

* streamlit
* pdfplumber
* sentence-transformers
* scikit-learn
* numpy
* langchain-text-splitters 

---

# 🧠 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* FAISS Indexing
* NLP Pipelines
* ATS Optimization
* Semantic Similarity
* Resume Intelligence
* Information Retrieval
* Skill Extraction

---

# 👨‍💻 Author

### Aadhya Rohatgi

Built using:

* Python
* Streamlit
* NLP
* FAISS
* Sentence-Transformers
* RAG Architecture

Project implementation based on the AI resume analysis pipeline defined in `app.py`. 
