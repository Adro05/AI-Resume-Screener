# AI Resume Screening System

An AI-powered Resume Screening System that compares resumes with job descriptions using NLP-based similarity analysis and ATS-style scoring.

## Features

- Upload PDF resumes
- ATS match score calculation
- NLP-based resume analysis
- Skill extraction
- Missing skill detection
- Real-time Streamlit interface
- TF-IDF vectorization and cosine similarity

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NLP
- TF-IDF Vectorization
- pdfplumber

---

## How It Works

1. User uploads a resume PDF
2. User pastes a job description
3. System extracts text from the resume
4. TF-IDF vectorization converts text into numerical form
5. Cosine similarity calculates ATS match score
6. Skills and missing keywords are displayed

---

## Features Implemented

- ATS Score Prediction
- Skill Detection
- Missing Skills Identification
- Resume Parsing
- Interactive Dashboard

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

Live Demo
https://ai-resume-screener-himnznnruqezneyi6vzmaz.streamlit.app/

Screenshot


Future Improvements:
Transformer-based semantic matching
Resume ranking system
AI-generated resume suggestions
Multiple resume comparison
Recruiter dashboard