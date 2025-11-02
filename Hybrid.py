# Hybrid.py - Fixed version
import pandas as pd
import numpy as np
import warnings
import re
from fuzzywuzzy import fuzz, process
from Collaborative import collaborative_filtering, cf_system
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
        
        print(f"\nDEBUG DATA SOURCES:")
        print(f"   Movie: '{title}', User: {userId}")
        
        # 1. Xem CB recommendations
        cb_results = recommender_CB(title, 20)
        if cb_results:
            print(f" Top 10 CB results:")
            for i, movie in enumerate(cb_results[:10], 1):
                print(f"      {i:2d}. {movie}")
        else:
            print(" No CB results found!")
        
        return cb_results

    def safe_fuzzy_match(self, cb_movie, cf_titles, threshold=50):
        if cb_movie in cf_titles:
            return cb_movie, 100
        
        cb_lower = cb_movie.lower().strip()
        cf_titles_lower = [title.lower().strip() for title in cf_titles]
        
        cb_clean = re.sub(r'[^\w\s]', '', cb_lower)
        for cf_title in cf_titles:
            cf_clean = re.sub(r'[^\w\s]', '', cf_title.lower())
            if cb_clean == cf_clean:
                return cf_title, 100

        best_match = None
        best_score = 0
        
        for cf_title in cf_titles:
            cf_lower = cf_title.lower()
            
            cb_tokens = set(cb_lower.split())
            cf_tokens = set(cf_lower.split())
            
            if not cb_tokens or not cf_tokens:
                continue
                
            intersection = cb_tokens.intersection(cf_tokens)
            union = cb_tokens.union(cf_tokens)
            
            if union:
                token_ratio = len(intersection) / len(union) * 100
            else:
                token_ratio = 0
            
            partial_ratio = 0
            if cb_lower in cf_lower or cf_lower in cb_lower:
                partial_ratio = 90
            else:
                for i in range(len(cb_lower) - 3):
                    substring = cb_lower[i:i+4]
                    if substring in cf_lower:
                        partial_ratio = max(partial_ratio, 70)
            
            score = max(token_ratio, partial_ratio)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = cf_title
        
        if best_match:
            return best_match, best_score
        
        return None, 0

    def calculate_dynamic_weights(self, exact_overlap_count, fuzzy_match_count, total_cb, avg_match_quality):
        total_overlap = exact_overlap_count + fuzzy_match_count
        overlap_ratio = total_overlap / total_cb if total_cb > 0 else 0
        
        print(f"Overlap stats: {exact_overlap_count} exact + {fuzzy_match_count} fuzzy = {total_overlap}/{total_cb} ({overlap_ratio:.1%})")
        
        if overlap_ratio >= 0.5:
            cf_weight, cb_weight = 0.5, 0.5
        elif overlap_ratio >= 0.3:
            cf_weight, cb_weight = 0.4, 0.6
        elif overlap_ratio >= 0.1:
            cf_weight, cb_weight = 0.3, 0.7
        else:
            cf_weight, cb_weight = 0.2, 0.8
        
        # Adjust based on match quality
        if fuzzy_match_count > 0:
            quality_factor = avg_match_quality / 100
            cf_weight = cf_weight * (0.5 + 0.5 * quality_factor)
            cb_weight = 1 - cf_weight
        
        print(f"Smart weights - CF: {cf_weight:.2f}, CB: {cb_weight:.2f}")
        return cf_weight, cb_weight

    def normalize_cf_scores(self, cf_mapping):
        """Normalize CF scores thông minh hơn"""
        if not cf_mapping:
            return {}
            
        scores = list(cf_mapping.values())
        if not scores:
            return {}
            
        cf_min, cf_max = min(scores), max(scores)
        
        # Nếu range quá hẹp, mở rộng normalization
        score_range = cf_max - cf_min
        if score_range < 0.5:
            expansion = (0.5 - score_range) / 2
            cf_min = max(0, cf_min - expansion)
            cf_max = cf_max + expansion
        
        cf_normalized = {}
        for title, score in cf_mapping.items():
            if cf_max > cf_min:
                normalized = (score - cf_min) / (cf_max - cf_min)
                # Đảm bảo trong range [0, 1]
                normalized = max(0, min(1, normalized))
            else:
                normalized = 0.5
            cf_normalized[title] = normalized
            
        print(f"CF normalization: {cf_min:.3f}-{cf_max:.3f} -> 0-1 scale")
        return cf_normalized

    def generate_final_recommendations(self, cb_results, cf_mapping, cf_normalized, fuzzy_matches, cf_weight, cb_weight, top_n):
        """Tạo final recommendations với scoring tốt hơn"""
        hybrid_movies = []
        seen_titles = set()
        
        # Tạo mapping từ fuzzy matches
        fuzzy_mapping = {match['cb_movie']: match for match in fuzzy_matches}
        
        # Xử lý CB movies với scoring tốt hơn
        for i, cb_movie in enumerate(cb_results):
            if cb_movie in seen_titles:
                continue
            seen_titles.add(cb_movie)
            
            # Tính CB score (exponential decay for better ranking)
            cb_score = np.exp(-i / 8)  # Exponential decay
            
            # Tìm CF score
            cf_norm = cf_normalized.get(cb_movie, 0.0)
            cf_raw = cf_mapping.get(cb_movie, 0.0)
            
            # Kiểm tra fuzzy match
            if cf_norm == 0.0 and cb_movie in fuzzy_mapping:
                match_data = fuzzy_mapping[cb_movie]
                cf_norm = match_data['cf_norm'] * (match_data['quality'] / 100)  # Weight by match quality
                cf_raw = match_data['cf_raw']
            
            # Tính final score với balancing
            final_score = (cf_norm * cf_weight + cb_score * cb_weight)
            
            # Xác định loại
            if cf_norm > 0.1 and cb_score > 0.3:
                rec_type = "Hybrid"
            elif cf_norm > 0.3:
                rec_type = "CF-Only"
            else:
                rec_type = "CB-Only"
            
            hybrid_movies.append({
                'title': cb_movie,
                'final_score': round(final_score, 4),
                'cf_score': round(cf_norm, 4),
                'cb_score': round(cb_score, 4),
                'cf_raw': round(cf_raw, 4) if cf_raw else 0.0,
                'rec_type': rec_type
            })
        
        # Thêm CF-only movies chất lượng cao
        cf_titles_sorted = sorted(cf_mapping.keys(), 
                                key=lambda x: cf_normalized.get(x, 0), 
                                reverse=True)
        
        cf_added = 0
        for cf_title in cf_titles_sorted:
            if cf_title not in seen_titles and cf_added < 5:  # Thêm một ít CF-only
                cf_norm = cf_normalized.get(cf_title, 0.0)
                cf_raw = cf_mapping.get(cf_title, 0.0)
                
                # Chỉ thêm nếu CF score cao
                if cf_norm > 0.5:
                    hybrid_movies.append({
                        'title': cf_title,
                        'final_score': round(cf_norm * cf_weight, 4),
                        'cf_score': round(cf_norm, 4),
                        'cb_score': 0.0,
                        'cf_raw': round(cf_raw, 4),
                        'rec_type': 'CF-Only'
                    })
                    seen_titles.add(cf_title)
                    cf_added += 1
        
        # Tạo final result
        result_df = pd.DataFrame(hybrid_movies)
        result_df = result_df.drop_duplicates(subset=['title'])
        result_df = result_df.sort_values('final_score', ascending=False).head(top_n)
        result_df = result_df.reset_index(drop=True)
        
        # Thống kê
        hybrid_count = len(result_df[result_df['rec_type'] == "Hybrid"])
        cb_only_count = len(result_df[result_df['rec_type'] == "CB-Only"])
        cf_only_count = len(result_df[result_df['rec_type'] == "CF-Only"])
        
        print(f"FINAL HYBRID RESULTS:")
        print(f"   - Total: {len(result_df)} recommendations")
        print(f"   - Hybrid: {hybrid_count}")
        print(f"   - CB-Only: {cb_only_count}") 
        print(f"   - CF-Only: {cf_only_count}")
        
        return result_df

    def find_best_cf_match(self, cb_movie, cf_mapping, cf_normalized, threshold=50):
        """Tìm CF match tốt nhất - FIXED VERSION"""
        cf_titles = list(cf_mapping.keys())
        best_match, match_score = self.safe_fuzzy_match(cb_movie, cf_titles, threshold)
        
        if best_match:
            return best_match, cf_normalized.get(best_match, 0.0), cf_mapping.get(best_match, 0.0), match_score
        
        return None, 0.0, 0.0, 0

    def direct_hybrid(self, userId, title, top_n=10, cf_weight=0.5, cb_weight=0.5):
        """Direct hybrid recommendations - FIXED VERSION"""
        print(f"\nGenerating Hybrid Recommendations:")
        print(f"   User: {userId}, Movie: '{title}'")
        
        try:
            # Get CB recommendations
            cb_results = self.debug_data_sources(userId, title)
            
            if not cb_results:
                print("No CB results found!")
                return pd.DataFrame()
            
            # Get CF predictions
            print("Getting CF predictions...")
            cf_results = collaborative_filtering(userId, top=top_n * 3)
            
            # Create CF mapping
            cf_mapping = {}
            
            if not cf_results.empty and 'predicted_rating' in cf_results.columns:
                for _, row in cf_results.iterrows():
                    title_cf = row['title']
                    predicted_rating = row['predicted_rating']
                    cf_mapping[title_cf] = predicted_rating
                
                print(f"CF mapping created: {len(cf_mapping)} movies")
            
            # Smart normalization
            cf_normalized = self.normalize_cf_scores(cf_mapping)
            
            # Find matches với safe fuzzy matching
            exact_overlap = set(cb_results).intersection(set(cf_mapping.keys()))
            print(f"Exact overlap: {len(exact_overlap)} movies")
            
            fuzzy_matches = []
            for cb_movie in cb_results:
                if cb_movie not in exact_overlap:
                    best_match, cf_norm, cf_raw, quality = self.find_best_cf_match(
                        cb_movie, cf_mapping, cf_normalized, threshold=40  # Lower threshold
                    )
                    if best_match:
                        fuzzy_matches.append({
                            'cb_movie': cb_movie,
                            'cf_match': best_match,
                            'cf_norm': cf_norm,
                            'cf_raw': cf_raw,
                            'quality': quality
                        })
            
            print(f"Fuzzy matches found: {len(fuzzy_matches)} movies")
            
            # Calculate smart weights
            avg_quality = sum(match['quality'] for match in fuzzy_matches) / len(fuzzy_matches) if fuzzy_matches else 0
            cf_weight, cb_weight = self.calculate_dynamic_weights(
                len(exact_overlap), len(fuzzy_matches), len(cb_results), avg_quality
            )
            
            # Generate final recommendations
            final_results = self.generate_final_recommendations(
                cb_results, cf_mapping, cf_normalized, fuzzy_matches, cf_weight, cb_weight, top_n
            )
            
            if not final_results.empty:
                print(f"\nHYBRID RECOMMENDATIONS COMPLETE!")
                return final_results
            else:
                print("No hybrid recommendations generated!")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Error in direct hybrid: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

# Khởi tạo global instance
direct_hybrid_recommender = DirectHybridRecommender()