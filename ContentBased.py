import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from Cache import cache_manager
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
@cache_manager.cache
def Preprocess():
    mm = pd.read_csv('./movies_metadata.csv', low_memory=False)
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
    linksm = pd.read_csv('./links.csv')
    linksm = linksm[linksm['tmdbId'].notnull()]['tmdbId'].astype('int')
    linksm_mm = mm['id'].isin(linksm)
    linksm_mm = mm[linksm_mm]
    linksm_mm['tagline'] = linksm_mm['tagline'].apply(null)
    linksm_mm['overview'] = linksm_mm['overview'].apply(null)
    linksm_mm['script'] = linksm_mm['tagline'] +' '+ linksm_mm['overview']
    linksm_mm['script'] = linksm_mm['script'].apply(null)
    return linksm_mm, linksm, mm
@cache_manager.cache
def TF_IDF():
    #TF-IDF
    linksm_mm, _, _ = Preprocess()
    linksm_mm = linksm_mm.drop_duplicates(subset=['title'], keep='first')
    linksm_mm['title'] = linksm_mm['title'].fillna('Unknown Movie')
    linksm_mm['title'] = linksm_mm['title'].astype(str)
    TfIdf_Cal = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0.0,stop_words='english')
    TfIdf_matrix = TfIdf_Cal.fit_transform(linksm_mm['script'])
    #Processing
    linksm_mm = linksm_mm.reset_index(drop=True)
    movie_id_name = pd.Series(linksm_mm.index, index=linksm_mm['title'])
    return TfIdf_matrix, movie_id_name, linksm_mm
def near_similar_movie(movie):
    _, movie_id_name, _ = TF_IDF()
    all_title = movie_id_name.index.tolist()
    movie_lower = movie.lower()
    contains_keyw = [title for title in all_title if movie_lower in title.lower()]

    keyw_movie = [title for title in all_title if title.lower().startswith(movie_lower)]
    similar_titles = list(set(contains_keyw + keyw_movie))
    return similar_titles[:10]

def recommender_CB(title, top):
    try:
        TfIdf_matrix, movie_id_name, _ = TF_IDF()
        if title not in movie_id_name:
            print(f"Not Found '{title}'")
            similar_movies = near_similar_movie(title)
            if similar_movies:
                print(" Or you can want: ")
                for i, similar_title in enumerate(similar_movies, 1):
                    print(f"   {i}. {similar_title}")
            return []
        
        index = movie_id_name[title]
        cosine_similar = linear_kernel(TfIdf_matrix[index], TfIdf_matrix).flatten()
        
        if len(cosine_similar) != len(movie_id_name.index):
            cosine_similar = cosine_similar[:len(movie_id_name.index)]
        
        score_similar = pd.Series(cosine_similar, index=movie_id_name.index)
        score_similar = score_similar.drop(title, errors='ignore')
        top_movies = score_similar.nlargest(top)
        return top_movies.index.tolist()
    
    except Exception as e:
        print(f"Error: {e}")
        return []
