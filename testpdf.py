# test_pdf.py

from src.ingestion.pdf_parser import extract_text

resume = extract_text("data/resumes/Sairaj_Unde_Resume.pdf")

print(resume)



from src.preprocessing.text_cleaning import clean_text

text = "Python, SQL, AWS!!!"

print(clean_text(text))


import pandas as pd

print("Notebook Working")



from src.feature_engineering.embeddings import get_embedding

embedding = get_embedding(
    "Python Machine Learning"
)

print(embedding.shape)

from src.feature_engineering.embeddings import get_embedding
from src.models.ranking_model import calculate_similarity

resume = get_embedding(
    "Python Machine Learning SQL"
)

job = get_embedding(
    "Python SQL AWS"
)

score = calculate_similarity(
    resume,
    job
)

print(score)