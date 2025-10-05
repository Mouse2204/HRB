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

def null(x):
    if pd.isnull(x):
        return ''
    return x

def CF_Preprocessing():
    rate = pd.read_csv('D:/Recommender System/ratings_small.csv')
    rate['rating'] = rate['rating'].apply(null)
    rate['timestamp'] = rate['timestamp'].apply(null)
    rate['movieId'] = rate['movieId'].astype('int')
    rate['userId'] = rate['userId'].astype('int')
    rate['rating'] = rate['rating'].astype('float')
    return rate

def load_movie_data_with_links():
    movies = pd.read_csv('D:/Recommender System/movies_metadata.csv')
    movies = movies[['id', 'title']]
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype('int')
    
    links = pd.read_csv('D:/Recommender System/links.csv')
    links = links[['movieId', 'tmdbId']]
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype('int')
    
    movie_mapping = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='left')
    movie_mapping = movie_mapping[['movieId', 'title']]
    movie_mapping['title'] = movie_mapping['title'].fillna('Unknown Movie')
    
    return movie_mapping

def CF_Cosine():
    rate = CF_Preprocessing()
    train_data, test_data = train_test_split(rate, test_size=0.2, random_state=42)
    ultility_matrix = train_data.pivot(index='userId', columns='movieId', values='rating')
    ultility_matrix = ultility_matrix.fillna(0)
    cosine_sim = cosine_similarity(ultility_matrix.T)
    cosine_simdf = pd.DataFrame(cosine_sim, index=ultility_matrix.columns, columns=ultility_matrix.columns)
    return train_data, ultility_matrix, cosine_sim, cosine_simdf, test_data

def trainML():
    train_data, _, _, _, test_data = CF_Cosine()
    X_train = train_data[['userId', 'movieId']]
    Y_train = train_data['rating']

    X_test = test_data[['userId', 'movieId']]
    Y_test = test_data['rating']

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, rmse

def collaborative_filtering(userId, model, top):
    rate = CF_Preprocessing()
    movies = load_movie_data_with_links()
    
    _, ultility_matrix, _, _, _ = CF_Cosine()
    complete_matrix = ultility_matrix.columns
    
    unrated_user = complete_matrix[~complete_matrix.isin(rate[rate['userId']==userId]['movieId'])]
    
    predict = pd.DataFrame({'userId': userId, 'movieId': unrated_user})
    predict['RaPredict'] = model.predict(predict)
    
    recommend_CF = predict.sort_values('RaPredict', ascending=False).head(top)
    
    recommend_CF = pd.merge(recommend_CF, movies, on='movieId', how='left')
    
    return recommend_CF[['movieId', 'title', 'RaPredict']]
