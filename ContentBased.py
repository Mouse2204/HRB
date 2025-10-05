import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import linear_kernel, cosine_distances, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import warnings; warnings.simplefilter('ignore')
# Gọi TF_IDF để lấy các biến cần thiết
def preprocessing(x):
    if pd.isnull(x):
        return ''
    return ast.literal_eval(x)

def null(x):
    if pd.isnull(x):
        return ''
    return x
def Preprocess():
    mm = pd.read_csv('D:/Recommender System/movies_metadata.csv', low_memory=False)
    mm['belongs_to_collection'] = mm['belongs_to_collection'].apply(preprocessing)
    mm['genres'] = mm['genres'].apply(preprocessing)
    mm['spoken_languages']=mm['spoken_languages'].apply(preprocessing)
    ##--------------------------------
    mm['id'] = pd.to_numeric(mm['id'], errors='coerce') 
    mm = mm.dropna(subset=['id']) 
    mm['id'] = mm['id'].astype(int)
    mm['budget'] = mm['budget'].astype('int')
    ##--------------------------------
    mm['release_date'] = pd.to_datetime(mm['release_date'], errors='coerce')
    mm = mm.sort_values(by=['release_date', 'original_title'], ascending=[0,0])
    mm = mm.drop(columns=['homepage','video'])
    ##--------------------------------
    linksm = pd.read_csv('D:/Recommender System/links_small.csv')
    linksm = linksm[linksm['tmdbId'].notnull()]['tmdbId'].astype('int')
    linksm_mm = mm['id'].isin(linksm)
    linksm_mm = mm[linksm_mm]
    linksm_mm['tagline'] = linksm_mm['tagline'].apply(null)
    linksm_mm['overview'] = linksm_mm['overview'].apply(null)
    linksm_mm['script'] = linksm_mm['tagline'] +' '+ linksm_mm['overview']
    linksm_mm['script'] = linksm_mm['script'].apply(null)
    return linksm_mm, linksm, mm
def TF_IDF():
    #TF-IDF
    linksm_mm, _, _ = Preprocess()
    TfIdf_Cal = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0.0,stop_words='english')
    TfIdf_matrix = TfIdf_Cal.fit_transform(linksm_mm['script'])
    #Processing
    linksm_mm = linksm_mm.reset_index()
    movie_id_name = pd.Series(linksm_mm.index, index=linksm_mm['title'])
    return TfIdf_matrix, movie_id_name, linksm_mm

def recommender_CB(title, top):
    TfIdf_matrix, movie_id_name, _ = TF_IDF()
    index = movie_id_name[title]
    cosine_similar = linear_kernel(TfIdf_matrix[index], TfIdf_matrix).flatten()
    score_similar = pd.Series(cosine_similar, index=movie_id_name.index)
    score_similar = score_similar.drop(title)
    top_movies = score_similar.nlargest(top)
    return top_movies.index.tolist()
