from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_embedding, jd_embedding):

    score = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )

    return score[0][0]