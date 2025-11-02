import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle
import joblib
from Cache import cache_manager
warnings.simplefilter('ignore')

# =============================================================================
# ADVANCED COLLABORATIVE FILTERING WITH PRE-TRAINED MODEL
# =============================================================================

class CollaborativeFiltering:
    def __init__(self):
        self.ratings = None
        self.movie_mapping = None
        self.baseline_params = {}
        self.scaler = StandardScaler()
        self.trained_model = None
        self.is_initialized = False
        self.model_features = None
        self.model_path = "./cf_model.pkl"
        self.scaler_path = "./cf_scaler.pkl"
        self.features_path = "./cf_features.pkl"
    
    def CF_Preprocessing(self):
        """Tiền xử lý dữ liệu ratings - KHÔNG CACHE"""
        print(" Loading ratings data...")
        rate = pd.read_csv('./ratings.csv')
        rate = rate.dropna(subset=['rating'])
        rate['movieId'] = rate['movieId'].astype('int')
        rate['userId'] = rate['userId'].astype('int')
        rate['rating'] = rate['rating'].astype('float')
        
        
        return rate

    def load_movie_data_with_links(self):
        """Tải movie mapping - KHÔNG CACHE"""
        print(" Loading movie mapping...")
        movies = pd.read_csv('./movies_metadata.csv')
        movies = movies[['id', 'title']]
        movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
        movies = movies.dropna(subset=['id'])
        movies['id'] = movies['id'].astype('int')
        
        links = pd.read_csv('./links.csv')
        links = links[['movieId', 'tmdbId']]
        links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
        links = links.dropna(subset=['tmdbId'])
        links['tmdbId'] = links['tmdbId'].astype('int')
        
        movie_mapping = pd.merge(links, movies, left_on='tmdbId', right_on='id', how='inner')
        movie_mapping = movie_mapping[['movieId', 'title']]
        movie_mapping['title'] = movie_mapping['title'].fillna('Unknown Movie')
        
        return movie_mapping

    def calculate_baseline_parameters(self, ratings):
        """Tính baseline parameters - KHÔNG CACHE"""
        print(" Calculating baseline parameters...")
        
        # Sử dụng sample nhỏ hơn để tính baseline nhanh hơn

        sample_ratings = ratings
            
        mu = sample_ratings['rating'].mean()
        
        # User biases với smoothing
        user_stats = sample_ratings.groupby('userId').agg(
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
        movie_stats = sample_ratings.groupby('movieId').agg(
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

    def get_baseline_prediction(self, userId, movieId):
        """Dự đoán baseline cho user-movie pair"""
        try:
            mu = self.baseline_params['mu']
            user_bias = self.baseline_params['user_biases'].get(userId, 0)
            movie_bias = self.baseline_params['movie_biases'].get(movieId, 0)
            
            user_bias = np.clip(user_bias, -2, 2)
            movie_bias = np.clip(movie_bias, -2, 2)
            
            baseline = mu + user_bias + movie_bias
            return np.clip(baseline, 0.5, 5.0)
        except:
            return self.baseline_params.get('mu', 3.5)

    def save_model(self, model, scaler, features):
        """Lưu model và các components"""
        try:
            print("Saving model and components...")
            joblib.dump(model, self.model_path)
            joblib.dump(scaler, self.scaler_path)
            joblib.dump(features, self.features_path)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load model và các components"""
        try:
            if os.path.exists(self.model_path):
                print("Loading pre-trained model...")
                self.trained_model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.model_features = joblib.load(self.features_path)
                print("Model loaded successfully!")
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train_and_save_model(self):
        """Train và lưu model một lần duy nhất"""
        print("Training and saving model...")
        
        if self.ratings is None:
            self.ratings = self.CF_Preprocessing()
        
        print("Training Data Summary:")
        print(f"   - Total ratings: {len(self.ratings)}")
        print(f"   - Unique users: {self.ratings['userId'].nunique()}")
        print(f"   - Unique movies: {self.ratings['movieId'].nunique()}")
        
        # Sample data để train nhanh hơn

        train_sample = self.ratings
        
        # Chia train/test
        train_data, test_data = train_test_split(train_sample, test_size=0.2, random_state=42)
        
        # Tạo features
        print("   - Creating features...")
        train_data = self.create_features(train_data)
        test_data = self.create_features(test_data)
        
        # Feature columns
        feature_cols = [
            'userId', 'movieId', 'baseline', 
            'user_avg_rating', 'user_rating_count', 'user_rating_std',
            'movie_avg_rating', 'movie_rating_count', 'movie_rating_std'
        ]
        
        # Features cho model
        self.model_features = [col for col in feature_cols if col not in ['userId', 'movieId']]
        
        X_train = train_data[self.model_features]
        Y_train = train_data['rating']
        X_test = test_data[self.model_features] 
        Y_test = test_data['rating']
        
        # Scale features
        print("   - Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost
        print("   - Training XGBoost model...")
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train_scaled, Y_train)
        
        # Evaluate
        Y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        
        print(f"Model Training Complete - RMSE: {rmse:.4f}")
        
        # Lưu model
        self.save_model(model, self.scaler, self.model_features)
        self.trained_model = model
        
        return model, rmse

    def initialize_system(self):
        """Khởi tạo hệ thống với pre-trained model"""
        if not self.is_initialized:
            print("Initializing Collaborative Filtering System...")
            
            # Load dữ liệu cơ bản
            self.ratings = self.CF_Preprocessing()
            self.movie_mapping = self.load_movie_data_with_links()
            self.baseline_params = self.calculate_baseline_parameters(self.ratings)
            
            # Thử load model đã train
            if not self.load_model():
                print("No pre-trained model found, training new one...")
                self.train_and_save_model()
            
            self.is_initialized = True
            print("System initialized and ready!")

    def create_features(self, df):
        """Tạo features cho model"""
        try:
            # Baseline features
            user_ids = df['userId'].values
            movie_ids = df['movieId'].values
            
            baselines = []
            for user_id, movie_id in zip(user_ids, movie_ids):
                baselines.append(self.get_baseline_prediction(user_id, movie_id))
            
            df['baseline'] = baselines
            
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
            mu = self.baseline_params['mu']
            df['user_avg_rating'] = df['user_avg_rating'].fillna(mu)
            df['movie_avg_rating'] = df['movie_avg_rating'].fillna(mu)
            df['user_rating_count'] = df['user_rating_count'].fillna(0)
            df['movie_rating_count'] = df['movie_rating_count'].fillna(0)
            df['user_rating_std'] = df['user_rating_std'].fillna(0)
            df['movie_rating_std'] = df['movie_rating_std'].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error in create_features: {e}")
            df['baseline'] = self.baseline_params['mu']
            return df

    def get_recommendations(self, userId, top_n=10, fast_mode=True):
        """Generate recommendations"""
        if not self.is_initialized:
            self.initialize_system()
        
        if fast_mode:
            print(f"Generating FAST recommendations for user {userId}...")
        else:
            print(f"Generating recommendations for user {userId}...")
        
        try:
            # Lấy movies user chưa rated
            rated_movies = self.ratings[self.ratings['userId'] == userId]['movieId'].unique()
            all_movies = self.ratings['movieId'].unique()
            unrated_movies = np.setdiff1d(all_movies, rated_movies)
            
            print(f"   Rated: {len(rated_movies)}, Unrated: {len(unrated_movies)}")
            
            
            # Tạo features cho unrated movies
            unrated_df = pd.DataFrame({
                'userId': userId,
                'movieId': unrated_movies
            })
            
            unrated_df = self.create_features(unrated_df)
            
            # Lấy features cho model
            X_pred = unrated_df[self.model_features]
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Dự đoán
            predictions = self.trained_model.predict(X_pred_scaled)
            unrated_df['predicted_rating'] = predictions
            
            # Thêm movie information
            result_df = pd.merge(unrated_df, self.movie_mapping, on='movieId', how='inner')
            
            # Lọc movies có ít nhất 3 ratings
            result_df = result_df[result_df['movie_rating_count'] >= 3]
            result_df = result_df.sort_values('predicted_rating', ascending=False).head(top_n)
            
            mode_text = "FAST" if fast_mode else "ADVANCED"
            print(f" {mode_text} Recommendations for User {userId}:")
            print(f"   Generated {len(result_df)} recommendations")
            if len(result_df) > 0:
                print(f"   Prediction range: {result_df['predicted_rating'].min():.3f} - {result_df['predicted_rating'].max():.3f}")
            
            return result_df[['movieId', 'title', 'predicted_rating', 'movie_rating_count']]
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            return pd.DataFrame()

# =============================================================================
# GLOBAL INSTANCE & MAIN INTERFACE
# =============================================================================

cf_system = CollaborativeFiltering()

def displayCF_New():
    """Main interface for collaborative filtering"""
    print("\n COLLABORATIVE RECOMMENDATION")
    print("Options: 1. Fast | 2. Standard")
    option = input("Choose mode (Default: 1): ").strip() or "1"
    
    user_input = input("Enter user ID (Default: 2): ").strip()
    top_input = input("Number of recommendations (Default: 10): ").strip()
    userId = int(user_input) if user_input.isdigit() else 2
    top = int(top_input) if top_input.isdigit() else 10
    
    fast_mode = (option == "1")
    
    try:
        results = cf_system.get_recommendations(userId, top, fast_mode=fast_mode)
        if not results.empty:
            print(f"\nRecommended {len(results)} movies:")
            print(results.to_string(index=False))
            
            # Hiển thị model info
            model_exists = os.path.exists(cf_system.model_path)
            print(f"\nModel Status: {'TRAINED' if model_exists else '❌ NOT SAVED'}")
            return True
        else:
            print("No recommendations found!")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def CF_Preprocessing():
    return cf_system.CF_Preprocessing()

def load_movie_data_with_links():
    return cf_system.load_movie_data_with_links()

def trainML():
    cf_system.initialize_system()
    return cf_system.trained_model, 0.0

def collaborative_filtering(userId, model=None, top=10):
    return cf_system.get_recommendations(userId, top, fast_mode=True)

def debug_data_sources():
    return cf_system.debug_system()

# Function để train model một lần
def train_model_once():
    print("TRAINING MODEL ONCE AND SAVING...")
    cf_system.initialize_system()
    print("Model is now ready for use!")

if __name__ == "__main__":
    train_model_once()