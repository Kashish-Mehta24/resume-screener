# 🤖 Resume Screening App (BERT-Powered)

This is a Resume Screening System that compares a candidate's resume (in PDF format) with a job description using **BERT embeddings** and **cosine similarity** to compute a match score.

> Built with `sentence-transformers`, `scikit-learn`, `PyMuPDF`, and `Streamlit`.

---

## 🚀 Features

- Upload your resume (PDF)
- Paste a job description
- Get a similarity score between 0 and 1
- See whether your resume is a strong, moderate, or weak match
- Lightweight and runs in the browser with Streamlit

---

## 🧠 How It Works

1. Extracts text from uploaded PDF resume
2. Uses a pre-trained BERT model (`all-MiniLM-L6-v2`) to create embeddings
3. Calculates cosine similarity between resume and JD
4. Displays score and interpretation

---

## 📦 Tech Stack

- **Language**: Python
- **Frontend**: Streamlit
- **NLP Model**: SentenceTransformers (BERT)
- **Similarity**: Cosine Similarity via scikit-learn
- **PDF Parsing**: PyMuPDF (`fitz`)

---

## 💻 How to Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/Kashish-Mehta24/resume-screener.git
cd resume-screener

# 2. Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate  # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
