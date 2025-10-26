# Hybrid.py - Complete fixed version
import pandas as pd
import numpy as np
import warnings
import re
from fuzzywuzzy import fuzz, process
from Collaborative import load_movie_data_with_links
warnings.filterwarnings('ignore')

class DirectHybridRecommender:
    def __init__(self):
        self.smd = None
        self.cosine_sim = None
        self.indices = None
        self.indices_map = None
        
    def debug_data_sources(self, userId, title):
        """Debug để hiểu data sources"""
        from ContentBased import recommender_CB
        from Collaborative import trainML
        
        print(f"\n🔍 DEBUG DATA SOURCES:")
        print(f"   Movie: '{title}', User: {userId}")
        
        # 1. Xem CB recommendations
        cb_results = recommender_CB(title, 20)
        print(f"   ✅ Top 10 CB results:")
        for i, movie in enumerate(cb_results[:10], 1):
            print(f"      {i:2d}. {movie}")
        
        return cb_results

    def find_best_cf_match(self, cb_movie, cf_mapping, cf_normalized, threshold=60):
        """Tìm CF match tốt nhất cho CB movie - FIXED"""
        cf_titles = list(cf_mapping.keys())
        
        # Thử enhanced fuzzy matching
        best_match, match_score = self.enhanced_fuzzy_match(cb_movie, cf_titles, threshold)
        
        if best_match:
            return best_match, cf_normalized[best_match], cf_mapping[best_match], match_score
        
        return None, 0.0, 0.0, 0

    def enhanced_fuzzy_match(self, cb_movie, cf_titles, threshold=60):
        """Enhanced fuzzy matching với threshold thấp"""
        if cb_movie in cf_titles:
            return cb_movie, 100
        
        cb_lower = cb_movie.lower().strip()
        cf_titles_lower = [title.lower().strip() for title in cf_titles]
        
        # Strategy 1: Token set ratio (tolerant nhất)
        best_match = None
        best_score = 0
        
        try:
            matches = process.extract(cb_lower, cf_titles_lower, scorer=fuzz.token_set_ratio, limit=5)
            for match, score, _ in matches:
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = cf_titles[cf_titles_lower.index(match)]
        except:
            pass
        
        # Strategy 2: Simple word overlap
        if not best_match:
            cb_words = set(cb_lower.split())
            for cf_title in cf_titles:
                cf_words = set(cf_title.lower().split())
                common_words = cb_words.intersection(cf_words)
                if len(common_words) >= 1:  # Chỉ cần 1 từ chung
                    overlap_ratio = len(common_words) / min(len(cb_words), len(cf_words))
                    if overlap_ratio > 0.3:
                        return cf_title, int(overlap_ratio * 100)
        
        if best_match:
            print(f"   🔍 Fuzzy match ({best_score}%): '{cb_movie}' -> '{best_match}'")
            return best_match, best_score
        
        return None, 0

    def calculate_dynamic_weights(self, exact_overlap_count, fuzzy_match_count, total_cb, avg_match_quality):
        """Tính weights động"""
        total_overlap = exact_overlap_count + fuzzy_match_count
        overlap_ratio = total_overlap / total_cb if total_cb > 0 else 0
        
        print(f"📊 Overlap stats: {exact_overlap_count} exact + {fuzzy_match_count} fuzzy = {total_overlap}/{total_cb} ({overlap_ratio:.1%})")
        
        # Base weights
        if overlap_ratio == 0:
            cf_weight, cb_weight = 0.3, 0.7
        elif overlap_ratio < 0.2:
            cf_weight, cb_weight = 0.4, 0.6
        else:
            cf_weight, cb_weight = 0.5, 0.5
        
        print(f"⚖️ Final weights - CF: {cf_weight:.2f}, CB: {cb_weight:.2f}")
        return cf_weight, cb_weight

    def generate_final_recommendations(self, cb_results, cf_mapping, cf_normalized, fuzzy_matches, cf_weight, cb_weight, top_n):
        """Tạo final recommendations"""
        hybrid_movies = []
        seen_titles = set()
        
        # Tạo mapping từ fuzzy matches
        fuzzy_mapping = {match['cb_movie']: match for match in fuzzy_matches}
        
        # Xử lý CB movies
        for i, cb_movie in enumerate(cb_results):
            if cb_movie in seen_titles:
                continue
            seen_titles.add(cb_movie)
            
            # Tính CB score
            cb_score = 1.0 - (i / len(cb_results))
            
            # Tìm CF score
            cf_norm = cf_normalized.get(cb_movie, 0.0)
            cf_raw = cf_mapping.get(cb_movie, 0.0)
            
            # Kiểm tra fuzzy match
            if cf_norm == 0.0 and cb_movie in fuzzy_mapping:
                match_data = fuzzy_mapping[cb_movie]
                cf_norm = match_data['cf_norm']
                cf_raw = match_data['cf_raw']
            
            # Tính final score
            final_score = (cf_norm * cf_weight + cb_score * cb_weight)
            
            # Xác định loại
            if cf_norm > 0.1 and cb_score > 0.1:
                rec_type = "Hybrid"
            elif cf_norm > 0.1:
                rec_type = "CF-Only"
            else:
                rec_type = "CB-Only"
            
            hybrid_movies.append({
                'title': cb_movie,
                'final_score': final_score,
                'cf_score': cf_norm,
                'cb_score': cb_score,
                'cf_raw': cf_raw,
                'rec_type': rec_type
            })
        
        # Thêm CF-only movies từ top CF
        cf_titles_sorted = sorted(cf_mapping.keys(), key=lambda x: cf_mapping[x], reverse=True)
        cf_added = 0
        
        for cf_title in cf_titles_sorted:
            if cf_title not in seen_titles and cf_added < 10:
                cf_norm = cf_normalized.get(cf_title, 0.0)
                
                hybrid_movies.append({
                    'title': cf_title,
                    'final_score': cf_norm * cf_weight,
                    'cf_score': cf_norm,
                    'cb_score': 0.0,
                    'cf_raw': cf_mapping[cf_title],
                    'rec_type': 'CF-Only'
                })
                seen_titles.add(cf_title)
                cf_added += 1
        
        # Tạo final result
        result_df = pd.DataFrame(hybrid_movies)
        result_df = result_df.drop_duplicates(subset=['title'])
        result_df = result_df.sort_values('final_score', ascending=False).head(top_n)
        
        # Thống kê
        hybrid_count = len(result_df[result_df['rec_type'] == 'Hybrid'])
        cb_only_count = len(result_df[result_df['rec_type'] == 'CB-Only'])
        cf_only_count = len(result_df[result_df['rec_type'] == 'CF-Only'])
        
        print(f"✅ FINAL HYBRID RESULTS:")
        print(f"   - Total: {len(result_df)} recommendations")
        print(f"   - Hybrid: {hybrid_count}")
        print(f"   - CB-Only: {cb_only_count}")
        print(f"   - CF-Only: {cf_only_count}")
        
        return result_df

    def direct_hybrid(self, userId, title, top_n=10, cf_weight=0.5, cb_weight=0.5):
        print(f"   User: {userId}, Movie: '{title}'")
        
        try:
            # Debug data sources
            cb_results = self.debug_data_sources(userId, title)
            
            if not cb_results:
                return pd.DataFrame()
            
            # Lấy CF predictions
            from Collaborative import trainML, collaborative_filtering
            print("📊 Training CF model...")
            model, _ = trainML()
            cf_results = collaborative_filtering(userId, model, top_n * 3)
            
            # Tạo CF mapping
            cf_mapping = {}
            cf_normalized = {}
            
            if not cf_results.empty and 'RaPredict' in cf_results.columns:
                cf_predictions = cf_results['RaPredict']
                cf_min, cf_max = cf_predictions.min(), cf_predictions.max()
                
                print(f"📊 CF scores range: {cf_min:.3f} to {cf_max:.3f}")
                
                for _, row in cf_results.iterrows():
                    title_cf = row['title']
                    if cf_max > cf_min:
                        normalized_score = (row['RaPredict'] - cf_min) / (cf_max - cf_min)
                    else:
                        normalized_score = 0.5
                    
                    cf_mapping[title_cf] = row['RaPredict']
                    cf_normalized[title_cf] = normalized_score
                
                print(f"📊 CF mapping created: {len(cf_mapping)} movies")
            
            # Tìm fuzzy matches
            exact_overlap = set(cb_results).intersection(set(cf_mapping.keys()))
            print(f"🔍 Exact overlap: {len(exact_overlap)} movies")
            
            fuzzy_matches = []
            for cb_movie in cb_results:
                if cb_movie not in exact_overlap:
                    best_match, cf_norm, cf_raw, quality = self.find_best_cf_match(
                        cb_movie, cf_mapping, cf_normalized, threshold=60
                    )
                    if best_match:
                        fuzzy_matches.append({
                            'cb_movie': cb_movie,
                            'cf_match': best_match,
                            'cf_norm': cf_norm,
                            'cf_raw': cf_raw,
                            'quality': quality
                        })
            
            # Tính weights
            avg_quality = sum(match['quality'] for match in fuzzy_matches) / len(fuzzy_matches) if fuzzy_matches else 0
            cf_weight, cb_weight = self.calculate_dynamic_weights(
                len(exact_overlap), len(fuzzy_matches), len(cb_results), avg_quality
            )
            
            # Generate final recommendations
            return self.generate_final_recommendations(
                cb_results, cf_mapping, cf_normalized, fuzzy_matches, cf_weight, cb_weight, top_n
            )
            
        except Exception as e:
            print(f"❌ Error in direct hybrid: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

# Khởi tạo
direct_hybrid_recommender = DirectHybridRecommender()