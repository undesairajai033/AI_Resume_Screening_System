import streamlit as st

st.title(
    "AI Resume Screening System"
)

uploaded_resume = st.file_uploader(
    "Upload Resume"
)

uploaded_jd = st.file_uploader(
    "Upload Job Description"
)

if uploaded_resume and uploaded_jd:

    st.success(
        "Files Uploaded Successfully"
    )