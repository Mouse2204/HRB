import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter('ignore')

# =============================================================================
# IMPROVED COLLABORATIVE FILTERING (BASIC VERSION)
# =============================================================================

class ImprovedCollaborativeFiltering:
    def __init__(self):
        self.ratings = None
        self.movie_mapping = None
        self.baseline_params = {}
        
    def CF_Preprocessing(self):
        """Tiền xử lý dữ liệu ratings"""
        rate = pd.read_csv('D:/Recommender System/ratings.csv')
        rate['rating'] = rate['rating'].apply(lambda x: 0 if pd.isnull(x) else x)
        rate['movieId'] = rate['movieId'].astype('int')
        rate['userId'] = rate['userId'].astype('int')
        rate['rating'] = rate['rating'].astype('float')
        return rate

    def load_movie_data_with_links(self):
        """Tải và mapping dữ liệu movie"""
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
        
        movie_mapping = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
        movie_mapping = movie_mapping[['movieId', 'title']]
        movie_mapping['title'] = movie_mapping['title'].fillna('Unknown Movie')
        
        return movie_mapping

    def calculate_baseline_parameters(self, ratings):
        """Tính baseline parameters theo công thức: b_xi = μ + b_x + b_i"""
        # Overall mean rating
        mu = ratings['rating'].mean()
        
        # User biases
        user_means = ratings.groupby('userId')['rating'].mean()
        user_biases = user_means - mu
        
        # Movie biases
        movie_means = ratings.groupby('movieId')['rating'].mean()
        movie_biases = movie_means - mu
        
        return {
            'mu': mu,
            'user_biases': user_biases.to_dict(),
            'movie_biases': movie_biases.to_dict()
        }

    def get_baseline_prediction(self, userId, movieId):
        """Dự đoán baseline cho user-movie pair"""
        mu = self.baseline_params['mu']
        user_bias = self.baseline_params['user_biases'].get(userId, 0)
        movie_bias = self.baseline_params['movie_biases'].get(movieId, 0)
        return mu + user_bias + movie_bias

    def train_hybrid_model(self):
        """Train hybrid model kết hợp baseline + XGBoost"""
        ratings = self.CF_Preprocessing()
        
        # Tính baseline parameters
        self.baseline_params = self.calculate_baseline_parameters(ratings)
        
        # Chia train/test
        train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
        
        # Thêm baseline features
        train_data['baseline'] = train_data.apply(
            lambda row: self.get_baseline_prediction(row['userId'], row['movieId']), axis=1
        )
        
        test_data['baseline'] = test_data.apply(
            lambda row: self.get_baseline_prediction(row['userId'], row['movieId']), axis=1
        )
        
        # Features: userId, movieId, baseline prediction
        X_train = train_data[['userId', 'movieId', 'baseline']]
        Y_train = train_data['rating']
        
        X_test = test_data[['userId', 'movieId', 'baseline']]
        Y_test = test_data['rating']
        
        # Train XGBoost model
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, Y_train)
        
        # Evaluate
        Y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        baseline_rmse = np.sqrt(mean_squared_error(Y_test, test_data['baseline']))
        
        print(f"✅ Hybrid Model Training Complete")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   Baseline RMSE: {baseline_rmse:.4f}")
        
        return model, rmse

    def collaborative_filtering_v2(self, userId, top_n=10, method='hybrid'):
        """
        Collaborative Filtering cải tiến với multiple methods
        """
        ratings = self.CF_Preprocessing()
        movie_mapping = self.load_movie_data_with_links()
        
        if method == 'hybrid':
            # Sử dụng hybrid model
            model, _ = self.train_hybrid_model()
            
            # Lấy movies user chưa rated
            rated_movies = ratings[ratings['userId'] == userId]['movieId']
            all_movies = ratings['movieId'].unique()
            unrated_movies = np.setdiff1d(all_movies, rated_movies)
            
            # Dự đoán ratings
            predictions = []
            for movie_id in unrated_movies:
                baseline = self.get_baseline_prediction(userId, movie_id)
                features = np.array([[userId, movie_id, baseline]])
                predicted_rating = model.predict(features)[0]
                predictions.append((movie_id, predicted_rating))
            
        else:  # baseline only
            rated_movies = ratings[ratings['userId'] == userId]['movieId']
            all_movies = ratings['movieId'].unique()
            unrated_movies = np.setdiff1d(all_movies, rated_movies)
            
            predictions = []
            for movie_id in unrated_movies:
                predicted_rating = self.get_baseline_prediction(userId, movie_id)
                predictions.append((movie_id, predicted_rating))
        
        # Sắp xếp và lấy top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]
        
        # Tạo result DataFrame
        result_df = pd.DataFrame(top_predictions, columns=['movieId', 'predicted_rating'])
        result_df = pd.merge(result_df, movie_mapping, on='movieId', how='left')
        
        # Thêm baseline để so sánh
        result_df['baseline'] = result_df.apply(
            lambda row: self.get_baseline_prediction(userId, row['movieId']), axis=1
        )
        
        print(f"🎯 {method.upper()} Recommendations for User {userId}:")
        print(f"   Method: {method}")
        print(f"   Top {top_n} recommendations generated")
        
        return result_df[['movieId', 'title', 'predicted_rating', 'baseline']]

    def debug_system(self):
        """Debug toàn bộ hệ thống"""
        ratings = self.CF_Preprocessing()
        movie_mapping = self.load_movie_data_with_links()
        
        print("🔍 COLLABORATIVE FILTERING SYSTEM DEBUG")
        print("=" * 50)
        
        # Basic stats
        print(f"📊 Data Statistics:")
        print(f"   - Total ratings: {len(ratings)}")
        print(f"   - Unique users: {ratings['userId'].nunique()}")
        print(f"   - Unique movies: {ratings['movieId'].nunique()}")
        print(f"   - Mapped movies: {len(movie_mapping)}")
        
        # Rating distribution
        print(f"   - Average rating: {ratings['rating'].mean():.2f}")
        print(f"   - Rating std: {ratings['rating'].std():.2f}")
        
        # Calculate baseline parameters
        self.baseline_params = self.calculate_baseline_parameters(ratings)
        print(f"📈 Baseline Parameters:")
        print(f"   - Overall mean (μ): {self.baseline_params['mu']:.3f}")
        print(f"   - User biases calculated: {len(self.baseline_params['user_biases'])}")
        print(f"   - Movie biases calculated: {len(self.baseline_params['movie_biases'])}")

# =============================================================================
# ADVANCED COLLABORATIVE FILTERING (ENHANCED VERSION)
# =============================================================================

class AdvancedCollaborativeFiltering:
    def __init__(self):
        self.ratings = None
        self.movie_mapping = None
        self.baseline_params = {}
        self.scaler = StandardScaler()
        
    def CF_Preprocessing(self):
        """Tiền xử lý dữ liệu ratings cải tiến"""
        rate = pd.read_csv('D:/Recommender System/ratings.csv')
        rate = rate.dropna(subset=['rating'])
        rate['movieId'] = rate['movieId'].astype('int')
        rate['userId'] = rate['userId'].astype('int')
        rate['rating'] = rate['rating'].astype('float')
        
        # Lọc outliers
        Q1 = rate['rating'].quantile(0.25)
        Q3 = rate['rating'].quantile(0.75)
        IQR = Q3 - Q1
        rate = rate[~((rate['rating'] < (Q1 - 1.5 * IQR)) | (rate['rating'] > (Q3 + 1.5 * IQR)))]
        
        return rate

    def load_movie_data_with_links(self):
        """Tải movie mapping"""
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
        
        movie_mapping = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
        movie_mapping = movie_mapping[['movieId', 'title']]
        movie_mapping['title'] = movie_mapping['title'].fillna('Unknown Movie')
        
        return movie_mapping

    def calculate_improved_baseline(self, ratings):
        """Baseline cải tiến với regularization"""
        mu = ratings['rating'].mean()
        
        # User biases với smoothing
        user_stats = ratings.groupby('userId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        )
        global_user_avg = user_stats['avg_rating'].mean()
        
        # Bayesian average cho user biases
        user_stats['user_bias'] = (
            (user_stats['rating_count'] * user_stats['avg_rating'] + 10 * global_user_avg) / 
            (user_stats['rating_count'] + 10)
        ) - mu
        
        # Movie biases với smoothing
        movie_stats = ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        )
        global_movie_avg = movie_stats['avg_rating'].mean()
        
        movie_stats['movie_bias'] = (
            (movie_stats['rating_count'] * movie_stats['avg_rating'] + 5 * global_movie_avg) / 
            (movie_stats['rating_count'] + 5)
        ) - mu
        
        return {
            'mu': mu,
            'user_biases': user_stats['user_bias'].to_dict(),
            'movie_biases': movie_stats['movie_bias'].to_dict(),
            'user_counts': user_stats['rating_count'].to_dict(),
            'movie_counts': movie_stats['rating_count'].to_dict()
        }

    def get_improved_baseline(self, userId, movieId):
        """Baseline prediction cải tiến"""
        mu = self.baseline_params['mu']
        user_bias = self.baseline_params['user_biases'].get(userId, 0)
        movie_bias = self.baseline_params['movie_biases'].get(movieId, 0)
        
        # Giới hạn bias để tránh predictions cực đoan
        user_bias = np.clip(user_bias, -2, 2)
        movie_bias = np.clip(movie_bias, -2, 2)
        
        baseline = mu + user_bias + movie_bias
        return np.clip(baseline, 0.5, 5.0)  # Giới hạn trong range rating

    def create_advanced_features(self, df):
        """Tạo advanced features cho model"""
        # Baseline features
        df['baseline'] = df.apply(
            lambda row: self.get_improved_baseline(row['userId'], row['movieId']), axis=1
        )
        
        # User features
        user_stats = self.ratings.groupby('userId').agg(
            user_avg_rating=('rating', 'mean'),
            user_rating_count=('rating', 'count'),
            user_rating_std=('rating', 'std')
        ).fillna(0)
        
        # Movie features  
        movie_stats = self.ratings.groupby('movieId').agg(
            movie_avg_rating=('rating', 'mean'),
            movie_rating_count=('rating', 'count'),
            movie_rating_std=('rating', 'std')
        ).fillna(0)
        
        # Merge features
        df = df.merge(user_stats, on='userId', how='left')
        df = df.merge(movie_stats, on='movieId', how='left')
        
        # Fill NaN values
        df['user_avg_rating'] = df['user_avg_rating'].fillna(self.baseline_params['mu'])
        df['movie_avg_rating'] = df['movie_avg_rating'].fillna(self.baseline_params['mu'])
        df['user_rating_count'] = df['user_rating_count'].fillna(0)
        df['movie_rating_count'] = df['movie_rating_count'].fillna(0)
        df['user_rating_std'] = df['user_rating_std'].fillna(0)
        df['movie_rating_std'] = df['movie_rating_std'].fillna(0)
        
        return df

    def train_advanced_model(self):
        """Train model cải tiến với advanced features - FIXED VERSION"""
        self.ratings = self.CF_Preprocessing()
        
        # Tính baseline parameters cải tiến
        self.baseline_params = self.calculate_improved_baseline(self.ratings)
        
        print("📊 Training Data Summary:")
        print(f"   - Total ratings: {len(self.ratings)}")
        print(f"   - Unique users: {self.ratings['userId'].nunique()}")
        print(f"   - Unique movies: {self.ratings['movieId'].nunique()}")
        print(f"   - Global mean rating: {self.baseline_params['mu']:.3f}")
        
        # Chia train/test
        train_data, test_data = train_test_split(self.ratings, test_size=0.2, random_state=42)
        
        # Tạo features
        train_data = self.create_advanced_features(train_data)
        test_data = self.create_advanced_features(test_data)
        
        # Feature columns - ĐỊNH NGHĨA RÕ RÀNG
        self.feature_cols = [
            'userId', 'movieId', 'baseline', 
            'user_avg_rating', 'user_rating_count', 'user_rating_std',
            'movie_avg_rating', 'movie_rating_count', 'movie_rating_std'
        ]
        
        # Features cho model (loại bỏ ID columns)
        self.model_features = [col for col in self.feature_cols if col not in ['userId', 'movieId']]
        
        X_train = train_data[self.model_features]
        Y_train = train_data['rating']
        X_test = test_data[self.model_features] 
        Y_test = test_data['rating']
        
        # Scale features và lưu scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
        model.fit(X_train_scaled, Y_train)
        
        # Evaluate
        Y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        baseline_rmse = np.sqrt(mean_squared_error(Y_test, test_data['baseline']))
        
        print(f"✅ Advanced Model Training Complete")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   Baseline RMSE: {baseline_rmse:.4f}")
        print(f"   Improvement: {baseline_rmse - rmse:.4f}")
        
        return model, rmse

    def generate_diverse_recommendations(self, userId, top_n=10):
        """Tạo recommendations đa dạng hơn - FINAL IMPROVED VERSION"""
        self.ratings = self.CF_Preprocessing()
        movie_mapping = self.load_movie_data_with_links()
        
        # Train model
        model, _ = self.train_advanced_model()
        
        # Lấy movies user chưa rated
        rated_movies = self.ratings[self.ratings['userId'] == userId]['movieId'].unique()
        all_movies = self.ratings['movieId'].unique()
        unrated_movies = np.setdiff1d(all_movies, rated_movies)
        
        print(f"🔍 Generating recommendations for user {userId}...")
        print(f"   Rated movies: {len(rated_movies)}")
        print(f"   Unrated movies: {len(unrated_movies)}")
        
        # Tạo features cho unrated movies
        unrated_df = pd.DataFrame({
            'userId': userId,
            'movieId': unrated_movies
        })
        
        unrated_df = self.create_advanced_features(unrated_df)
        
        # Lấy features cho model
        X_pred = unrated_df[self.model_features]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Dự đoán
        predictions = model.predict(X_pred_scaled)
        unrated_df['predicted_rating'] = predictions
        unrated_df['baseline'] = unrated_df.apply(
            lambda row: self.get_improved_baseline(userId, row['movieId']), axis=1
        )
        
        # Thêm movie information và rating counts
        result_df = pd.merge(unrated_df, movie_mapping, on='movieId', how='inner')
        
        # LỌC VÀ SẮP XẾP THÔNG MINH HƠN
        print("🎯 Applying smart filtering...")
        
        # Strategy 1: Lấy top predictions với độ tin cậy
        min_ratings_threshold = 5  # Chỉ lấy movies có ít nhất 5 ratings
        reliable_movies = result_df[result_df['movie_rating_count'] >= min_ratings_threshold]
        
        if len(reliable_movies) >= top_n:
            # Đủ movies tin cậy → lấy top từ reliable movies
            result_df = reliable_movies.sort_values('predicted_rating', ascending=False).head(top_n)
        else:
            # Không đủ movies tin cậy → kết hợp reliable + top predicted
            reliable_top = reliable_movies.sort_values('predicted_rating', ascending=False)
            remaining_slots = top_n - len(reliable_top)
            
            if remaining_slots > 0:
                # Lấy thêm từ tất cả movies (kể cả ít ratings)
                all_top = result_df.sort_values('predicted_rating', ascending=False).head(top_n * 2)
                # Ưu tiên movies có nhiều ratings hơn
                additional_movies = all_top[~all_top['movieId'].isin(reliable_top['movieId'])]
                additional_movies = additional_movies.sort_values(
                    ['predicted_rating', 'movie_rating_count'], 
                    ascending=[False, False]
                ).head(remaining_slots)
                
                result_df = pd.concat([reliable_top, additional_movies]).head(top_n)
            else:
                result_df = reliable_top.head(top_n)
        
        # Đảm bảo đa dạng: nhóm theo predicted_rating ranges
        if len(result_df) > top_n:
            # Tạo rating groups để đảm bảo đa dạng
            result_df['rating_group'] = pd.cut(result_df['predicted_rating'], bins=5)
            result_df = result_df.sort_values(['rating_group', 'predicted_rating'], ascending=[True, False])
            result_df = result_df.drop_duplicates(subset=['rating_group'], keep='first').head(top_n)
            result_df = result_df.drop(columns=['rating_group'])
        
        # Final sort by predicted rating
        result_df = result_df.sort_values('predicted_rating', ascending=False).head(top_n)
        
        print(f"🎯 FINAL Recommendations for User {userId}:")
        print(f"   Top {len(result_df)} recommendations generated")
        print(f"   Prediction range: {result_df['predicted_rating'].min():.3f} - {result_df['predicted_rating'].max():.3f}")
        print(f"   Rating counts: {result_df['movie_rating_count'].min()} - {result_df['movie_rating_count'].max()}")
        
        return result_df[['movieId', 'title', 'predicted_rating', 'baseline', 'movie_rating_count']]

# =============================================================================
# STANDALONE FUNCTIONS FOR COMPATIBILITY
# =============================================================================

def CF_Preprocessing():
    """Standalone function for compatibility"""
    cf_system = ImprovedCollaborativeFiltering()
    return cf_system.CF_Preprocessing()

def load_movie_data_with_links():
    """Standalone function for compatibility"""
    cf_system = ImprovedCollaborativeFiltering()
    return cf_system.load_movie_data_with_links()

def trainML():
    """Standalone function for compatibility"""
    cf_system = ImprovedCollaborativeFiltering()
    return cf_system.train_hybrid_model()

def collaborative_filtering(userId, model=None, top=10):
    """Standalone function for compatibility"""
    cf_system = ImprovedCollaborativeFiltering()
    if model is None:
        model, _ = cf_system.train_hybrid_model()
    return cf_system.collaborative_filtering_v2(userId, top, method='hybrid')

def debug_data_sources():
    """Standalone function for compatibility"""
    cf_system = ImprovedCollaborativeFiltering()
    return cf_system.debug_system()

# Advanced Collaborative Filtering instance
advanced_cf = AdvancedCollaborativeFiltering()

def displayCF_improved():
    """Improved version of collaborative filtering display"""
    print("\n👥 COLLABORATIVE RECOMMENDATION (IMPROVED)")
    user_input = input("Enter user ID (Default: 2): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    userId = int(user_input) if user_input.isdigit() else 2
    top = int(top_input) if top_input.isdigit() else 10
    
    print(f"\nGenerating IMPROVED recommendations for user {userId}...")
    try:
        results = advanced_cf.generate_diverse_recommendations(userId, top)
        if not results.empty:
            print(f"\nRecommended {len(results)} movies:")
            print(results.to_string(index=False))
            return True
        else:
            print("No recommendations found!")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False