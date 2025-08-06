import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz

model = SentenceTransformer("all-MiniLM-L6-v2")

#Extract Text from Uploaded Resume PDF

def extract_text(uploaded_file):
    text=""
    try:
        doc= fitz.open(stream= uploaded_file.read(), filetype="pdf")
        for page in doc:
            text+= page.get_text()
        doc.close()
    
    except Exception as e:
        st.error("❌ Failed to extract text from the uploaded PDF.")
        return ""
    
    return text    

def compare_similiarity(resume_text, jd_text):
    
    # creating embedding for them both
    resume_embedding =model.encode([resume_text])
    jd_embedding =model.encode([jd_text])
    
    # comaring the cosine similiarity
    
    similiarity_score= cosine_similarity(resume_embedding,jd_embedding)[0][0]
    return similiarity_score

st.set_page_config(page_title= "Resume Screener", layout= "centered")
st.title("Resume Screening System")

uploaded_file= st.file_uploader("upload your file here (PDF)" , type="pdf")
jd_text= st.text_area(" enter the job description here")


if st.button("🔍 Score Resume"):
    if uploaded_file is not None and jd_text.strip() != "":
        resume_text = extract_text(uploaded_file)


        if resume_text.strip() == "":
            st.error("⚠️ Could not extract text from the resume.")
        else:
            score = compare_similiarity(resume_text, jd_text)
            st.success(f"🧠 Similarity Score: **{score:.2f}**")

            # Give a human-readable verdict
            if score >= 0.75:
                st.markdown("✅ **Strong Match** – Looks like a great fit!")
            elif score >= 0.5:
                st.markdown("🟡 **Moderate Match** – Some overlap, consider improving your resume.")
            else:
                st.markdown("❌ **Weak Match** – Resume and job role don't align well.")
    else:
        st.warning("Please upload a resume and paste a job description.")

    
    
                