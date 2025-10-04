import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import linear_kernel, cosine_distances, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from KnowLedgeBased import Preprocess, TF_IDF, KnowLedge_Based

# Gọi TF_IDF để lấy các biến cần thiết
TfIdf_matrix, movie_id_name, linksm_mm = TF_IDF()

def recommender_CB(title, top):
    index = movie_id_name[title]
    cosine_similar = linear_kernel(TfIdf_matrix[index], TfIdf_matrix).flatten()
    score_similar = pd.Series(cosine_similar, index=movie_id_name.index)
    score_similar = score_similar.drop(title)
    return score_similar.nlargest(top).index

recommend_movie = recommender_CB('The Dark Knight', 10)
print("Recommend_movie:")
Res = pd.DataFrame({
    'Index': range(1, len(recommend_movie) + 1),
    'Title': linksm_mm.loc[recommend_movie, 'title']  # Lấy tiêu đề từ linksm_mm
})
print(Res.to_string(index=False))