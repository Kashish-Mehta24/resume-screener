import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer , PorterStemmer
from nltk.corpus import wordnet
import string

nltk.download("wordnet")
lemmatizer= WordNetLemmatizer()
stemmer = PorterStemmer()

# Download NLTK stopwords
nltk.download("stopwords")

# Keep stopwords but remove only common filler words (exclude tech words)
default_stopwords = set(stopwords.words("english"))
tech_terms = {"python", "java", "sql", "c++", "machine", "learning", "data", "analysis", "developer", "model"}
stop_words = default_stopwords - tech_terms

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Extract text from PDF
def extract_text(uploaded_file):
    text = ""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        st.error("❌ Failed to extract text from the uploaded PDF.")
        return ""
    return text

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    words = text.split()
    # Lemmatize & remove stopwords
    cleaned_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words
    ]
    # Join back to string
    return " ".join(cleaned_words)
    

# Chunk text to avoid model input length limit
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Main semantic similarity function
def semantic_similarity(resume_text, jd_text):
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)
    resume_chunks = chunk_text(resume_text)
    scores = []

    for chunk in resume_chunks:
        resume_embedding = model.encode([chunk], normalize_embeddings=True)
        score = np.dot(resume_embedding, jd_embedding.T)  # cosine similarity (since normalized)
        scores.append(score[0])

    # Take average of top 25% scores for better robustness
    scores.sort(reverse=True)
    top_n = max(1, len(scores) // 4)
    return float(sum(scores[:top_n]) / top_n)

# Synonym expansion with lemmatization + stemming
def expand_with_synonyms(word):
    word = lemmatizer.lemmatize(word)
    word = stemmer.stem(word)
    synonyms = {word}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            lemma_word = lemma.name().lower().replace("_", " ")
            lemma_word = lemmatizer.lemmatize(lemma_word)
            lemma_word = stemmer.stem(lemma_word)
            synonyms.add(lemma_word)
    return synonyms

# Token cleaning
def clean_tokens_for_matching(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    cleaned_tokens = []
    for w in tokens:
        w_lem = lemmatizer.lemmatize(w)
        w_stem = stemmer.stem(w_lem)
        if (
            w_lem not in stop_words
            and len(w_lem) > 2
            and w_lem.isalpha()
        ):
            cleaned_tokens.append(w_stem)
    return set(cleaned_tokens)

# Keyword match score
def keyword_match_score(resume_text, jd_text):
    resume_words = clean_tokens_for_matching(resume_text)
    jd_words = clean_tokens_for_matching(jd_text)
    if len(jd_words) == 0:
        return 0, []
    matched_words = set()
    missing_keywords = []
    for jd_word in jd_words:
        synonyms = expand_with_synonyms(jd_word)
        if synonyms & resume_words:
            matched_words.add(jd_word)
        else:
            missing_keywords.append(jd_word)
    match_score = len(matched_words) / len(jd_words)
    return match_score, sorted(missing_keywords)

# Streamlit UI
st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("Resume Screening System")

uploaded_files = st.file_uploader("📄 Upload your resume (PDF) ...", type="pdf",accept_multiple_files=True)
jd_text = st.text_area("📝 Enter the job description here")

if st.button("🔍 Score Resume"):
    if uploaded_files is not None and jd_text.strip() != "":
        result=[]    # List to store each resume's scores
        jd_text_clean = preprocess_text(jd_text)
        
        for files in uploaded_files:
            resume_text = preprocess_text(extract_text(files))
       
            if resume_text.strip() == "":
                 st.error("⚠️ Could not extract text from the resume.")
                 continue  # skip this file

            
            # Calculate scores
            semantic_score = semantic_similarity(resume_text, jd_text_clean)
            keyword_score, missing_keywords = keyword_match_score(resume_text, jd_text_clean)

            # Final score = semantic similarity
            final_score = (semantic_score * 0.85) + (keyword_score * 0.15)
            
            result.append({
                "Resume": files.name,
                "Final Score": final_score,
                "Semantic Score": semantic_score,
                "Keyword Score": keyword_score,
                "Missing Keywords": ", ".join(missing_keywords[:10])
            })
            
        # Sort the results list by Final Score in descending order
        results = sorted(result, key=lambda x: x["Final Score"], reverse=True)
        
        
        
        # Show results as a ranked table
        import pandas as pd

        # Create a rank number
        for idx, res in enumerate(results, start=1):
            res["Rank"] = idx
            res["Final Score"] = round(res["Final Score"], 2)
            res["Semantic Score"] = round(res["Semantic Score"], 2)
            res["Keyword Score"] = round(res["Keyword Score"], 2)

# Convert to DataFrame for table display
        df = pd.DataFrame(results, columns=["Rank", "Resume", "Final Score", "Semantic Score", "Keyword Score", "Missing Keywords"])

        st.subheader("📊 Ranked Resumes")
        st.dataframe(df, use_container_width=True)




        for res in results:
            st.markdown(f"### 📄 {res['Resume']} - Final Score: **{res['Final Score']}**")
            if res["Final Score"] >= 0.65:
                st.success("✅ Strong Match — Looks like a great fit!")
            elif res["Final Score"] >= 0.50:
                st.info("🟡 Moderate Match — Some overlap, consider improving your resume.")
            else:
                st.warning("🔴 Weak Match — Resume and job role don't align well.")
    else:
        st.warning("⚠️ Please upload at least one resume and enter a job description.")
