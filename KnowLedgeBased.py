import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import linear_kernel, cosine_distances, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings; warnings.simplefilter('ignore')
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
##--------------------------------
def KnowLedge_Based():
    cre = pd.read_csv('D:/Recommender System/credits.csv')  
    key=pd.read_csv('D:/Recommender System/keywords.csv')
    _, linksm, mm = Preprocess()
    key['id']=key['id'].astype('int')
    cre['id']=cre['id'].astype('int')
    mm = mm.merge(cre, on='id')
    mm = mm.merge(key, on='id')

    meta_mm = mm['id'].isin(linksm)
    meta_mm = mm[meta_mm]
    meta_mm['crew']=meta_mm['crew'].apply(preprocessing)
    meta_mm['keywords']=meta_mm['keywords'].apply(preprocessing)

    #KnowLedge-Based
    def extract_list_name(temp):
        try:
            if isinstance(temp, str):
                data = ast.literal_eval(temp)
            else:
                data = temp
            return [d['name'] for d in data if 'name' in d]
        except:
            return []
    def extract_list_character(temp):
        try:
            if isinstance(temp, str):
                data = ast.literal_eval(temp)
            else:
                data = temp
            return [d['character'] for d in data if 'character' in d]
        except:
            return []
        
    #Genres-Key-Cast-Crew: Name
    meta_mm['genres_list']=meta_mm['genres'].apply(extract_list_name)
    meta_mm['keyw']=meta_mm['keywords'].apply(extract_list_name)
    meta_mm['crew_name']=meta_mm['crew'].apply(extract_list_name)
    meta_mm['character']=meta_mm['cast'].apply(extract_list_character)
    return meta_mm
def option_choosen(type, Fval, top=10):
    meta_mm = KnowLedge_Based()
    valid_columns = ['genres_list', 'keyw', 'crew_name', 'character']
    
    if type not in valid_columns:
        print(f"Loại '{type}' không hợp lệ. Vui lòng chọn trong: {valid_columns}")
        return pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'genres_list', 'keyw', 'crew_name', 'character'])
    
    meta_mm[type] = meta_mm[type].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else x))
    df = meta_mm[meta_mm[type].explode().eq(Fval).groupby(level=0).any()]
    
    search_value_lower = Fval.lower()
    df = meta_mm[meta_mm[type].apply(lambda x: any(search_value_lower in item.lower() for item in x))]
    
    if df.empty:
        print(f"Không tìm thấy phim nào với '{Fval}' trong '{type}'.")
        return pd.DataFrame(columns=['title', 'vote_count', 'vote_average', 'genres_list', 'keyw', 'crew_name', 'character'])
    
    vote_avg = df[df['vote_average'].notnull()]['vote_average'].astype('float')
    C = vote_avg.mean()
    M = 3000

    specific_data = df[(df['vote_count'] >= M) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'vote_count', 'vote_average', 'genres_list', 'keyw', 'crew_name', 'character']].copy()
    specific_data['vote_count'] = specific_data['vote_count'].astype('int')
    specific_data['vote_average'] = specific_data['vote_average'].astype('float')

    specific_data['wr'] = (specific_data['vote_count'] * specific_data['vote_average'] + M * C) / (specific_data['vote_count'] + M)
    specific_data = specific_data.sort_values('wr', ascending=False).head(top)
    specific_data.to_csv('../specific_data.csv', index=False)
    print(f"Dữ liệu đã được lưu vào '{'../specific_data.csv'}'.")
    return specific_data

