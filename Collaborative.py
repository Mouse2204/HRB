import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import linear_kernel, cosine_distances, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def CF_Preprocessing():
    rate = pd.read_csv('D:/Recommender System/ratings_small.csv')
    rate['rating'] = rate['rating'].apply(null)
    rate['timestamp'] = rate['timestamp'].apply(null)
    rate['movieId'] = rate['movieId'].astype('int')
    rate['userId'] = rate['userId'].astype('int')
    rate['rating'] = rate['rating'].astype('float')
    return rate
#Cosine
def CF_Cosine():
    rate = CF_Preprocessing()
    train_data, test_data = train_test_split(rate, test_size=0.2, random_state=42)
    ultility_matrix = train_data.pivot(index='userId', columns='movieId', values='rating')
    ultility_matrix = ultility_matrix.fillna(0)
    cosine_sim = cosine_similarity(ultility_matrix.T)
    cosine_simdf = pd.DataFrame(cosine_sim, index=ultility_matrix.columns, columns=ultility_matrix.columns)
    return train_data, ultility_matrix, cosine_sim, cosine_simdf

def trainML():
    train_data, test_data = CF_Cosine()
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
    ultility_matrix = CF_Cosine()
    complete_matrix = ultility_matrix.columns
    unrated_user = complete_matrix[~complete_matrix.isin(rate[rate['userId']==userId]['movieId'])]
    predict = pd.DataFrame({'userId': userId, 'movieId': unrated_user})
    predict['RaPredict'] = model.predict(predict)

    recommend_CF=predict.sort_values('RaPredict', ascending=False).head(top)
    return recommend_CF[['movieId', 'RaPredict']]

model = trainML()
print(collaborative_filtering(1, model, 10))