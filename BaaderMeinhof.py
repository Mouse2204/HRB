# BaaderMeinhof.py - FIXED VERSION
import time
import numpy as np
from typing import Dict, List, Set
from collections import defaultdict
import logging

logger = logging.getLogger("baader_meinhof_engine")

class BaaderMeinhofEngine:
    """
    Baader-Meinhoff Engine với Search Boost + Natural Decay
    """
    
    def __init__(self, decay_rate: float = 0.1, search_boost: float = 1.5, max_memory_hours: int = 168):
        self.decay_rate = decay_rate
        self.search_boost = search_boost  # Boost multiplier cho searched movies
        self.max_memory_hours = max_memory_hours
        
        # Lưu lịch sử
        self.user_exposure_history: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.content_exposure_count: Dict[str, int] = defaultdict(int)
        self.searched_movies: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"Baader-Meinhoff Engine initialized with search_boost={search_boost}")

    def _title_to_id(self, title: str) -> str:
        """Chuyển title thành ID chuẩn"""
        if not title:
            return ""
        return title.lower().replace(' ', '_').replace(':', '').replace('-', '').replace("'", "").replace('"', '').replace('.', '').replace('!', '').replace('?', '')

    def _id_to_title(self, movie_id: str) -> str:
        """Chuyển ID ngược lại thành title (cho debug)"""
        if not movie_id:
            return ""
        return movie_id.replace('_', ' ').title()

    def record_search(self, user_id: str, searched_movie: str):
        """Ghi nhận movie mà user đã search - sẽ được BOOST"""
        if not searched_movie:
            return
            
        user_id_str = str(user_id)
        movie_id = self._title_to_id(searched_movie)
        
        if user_id_str not in self.searched_movies:
            self.searched_movies[user_id_str] = set()
        
        self.searched_movies[user_id_str].add(movie_id)
        logger.debug(f"User {user_id} searched for: {searched_movie}")

    def apply_baader_meinhof_decay(self, user_id: str, rtb_results: List[Dict], searched_movie: str = None) -> List[Dict]:
        """Áp dụng Baader-Meinhoff decay với SEARCH BOOST"""
        current_time = time.time()
        decayed_results = []
        
        print(f"🎭 Applying Baader-Meinhoff decay with SEARCH BOOST for user {user_id}...")
        
        # Ghi nhận searched movie nếu có
        if searched_movie:
            self.record_search(user_id, searched_movie)
        
        for result in rtb_results:
            movie_title = result.get('movie', '')
            original_score = result.get('final_score', 0.5)
            
            if not movie_title:
                continue
                
            movie_id = self._title_to_id(movie_title)
            
            # Calculate decay/boost multiplier
            multiplier = self._calculate_score_multiplier(user_id, movie_id, current_time, searched_movie)
            
            # Apply multiplier to final score
            adjusted_score = original_score * multiplier
            
            # Tạo adjusted result
            adjusted_result = result.copy()
            adjusted_result['final_score'] = round(adjusted_score, 4)
            adjusted_result['baader_meinhof_multiplier'] = round(multiplier, 3)
            adjusted_result['original_score'] = original_score
            adjusted_result['adjustment_type'] = "BOOST" if multiplier > 1.0 else "DECAY" if multiplier < 1.0 else "NEUTRAL"
            
            decayed_results.append(adjusted_result)
        
        # Sort lại theo adjusted score
        decayed_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return decayed_results

    def _calculate_score_multiplier(self, user_id: str, movie_id: str, current_time: float, searched_movie: str) -> float:
        """Tính multiplier dựa trên SEARCH BOOST và decay"""
        if not movie_id:
            return 1.0
            
        user_id_str = str(user_id)
        base_multiplier = 1.0
        
        # 1. 🚀 SEARCH BOOST - QUAN TRỌNG NHẤT
        if searched_movie:
            searched_movie_id = self._title_to_id(searched_movie)
            
            # Boost trực tiếp cho movie được search
            if movie_id == searched_movie_id:
                base_multiplier *= self.search_boost
                print(f"   🎬 {self._id_to_title(movie_id)}: SEARCH_BOOST({self.search_boost}x) → {base_multiplier:.2f}x")
                return min(3.0, base_multiplier)  # Max boost 3x
            
            # Boost cho các movie liên quan
            elif self._is_related_movie(movie_id, searched_movie_id):
                related_boost = 1.2  # 20% boost cho related movies
                base_multiplier *= related_boost
                print(f"   🎬 {self._id_to_title(movie_id)}: RELATED_BOOST({related_boost}x) → {base_multiplier:.2f}x")
                return min(2.0, base_multiplier)
        
        # 2. PERSONAL EXPOSURE DECAY (chỉ áp dụng nếu không có search boost)
        if user_id_str in self.user_exposure_history and movie_id in self.user_exposure_history[user_id_str]:
            last_exposure_time = self.user_exposure_history[user_id_str][movie_id]
            hours_since_exposure = (current_time - last_exposure_time) / 3600
            
            if hours_since_exposure < 6:
                personal_decay = 0.2
            elif hours_since_exposure < 24:
                personal_decay = 0.5
            elif hours_since_exposure < 168:
                personal_decay = 0.8
            else:
                personal_decay = 0.95
                
            base_multiplier *= personal_decay
            print(f"   🎬 {self._id_to_title(movie_id)}: PERSONAL_DECAY({personal_decay:.1f}x) → {base_multiplier:.2f}x")
        
        # 3. GLOBAL FREQUENCY DECAY
        global_exposure_count = self.content_exposure_count.get(movie_id, 0)
        if global_exposure_count > 10:
            frequency_decay = max(0.6, 1.0 - (global_exposure_count - 10) * 0.05)
            base_multiplier *= frequency_decay
            print(f"   🎬 {self._id_to_title(movie_id)}: FREQUENCY_DECAY({frequency_decay:.1f}x) → {base_multiplier:.2f}x")
        
        return max(0.1, base_multiplier)  # Min decay 0.1x

    def _is_related_movie(self, movie_id: str, searched_movie_id: str) -> bool:
        """Kiểm tra xem movie có liên quan đến searched movie không"""
        if not movie_id or not searched_movie_id:
            return False
            
        movie_title = self._id_to_title(movie_id).lower()
        searched_title = self._id_to_title(searched_movie_id).lower()
        
        # Cùng series
        series_keywords = ['toy story', 'avengers', 'batman', 'spider-man', 'star wars', 'marvel', 'dc', 'harry potter', 'lord of the rings', 'fast and furious']
        for keyword in series_keywords:
            if keyword in movie_title and keyword in searched_title:
                return True
        
        # Cùng keyword chính
        movie_words = set(movie_title.split())
        searched_words = set(searched_title.split())
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'is', 'on', 'with', 'as', 'at', 'by', 'an'}
        meaningful_common = movie_words.intersection(searched_words) - stop_words
        
        return len(meaningful_common) >= 2

    def update_exposure_history(self, user_id: str, movie_titles: List[str]):
        """Cập nhật exposure history"""
        user_id_str = str(user_id)
        current_time = time.time()
        
        print(f"🎭 Updating exposure history for user {user_id}: {len(movie_titles)} movies")
        
        for movie_title in movie_titles:
            if not movie_title:
                continue
            movie_id = self._title_to_id(movie_title)
            self.user_exposure_history[user_id_str][movie_id] = current_time
            self.content_exposure_count[movie_id] += 1
            print(f"   📝 {movie_title} -> exposure recorded")
        
        # Debug: verify storage
        user_exposure_count = len(self.user_exposure_history[user_id_str])
        print(f"✅ User {user_id} now has {user_exposure_count} exposures in history")

    def get_user_exposure_stats(self, user_id: str) -> Dict:
        """Lấy statistics về exposure history của user"""
        user_id_str = str(user_id)
        
        stats = {
            "total_exposures": 0, 
            "recent_exposures_6h": 0,
            "recent_exposures_24h": 0, 
            "decay_rate": self.decay_rate,
            "search_boost": self.search_boost,
            "global_content_count": len(self.content_exposure_count),
            "searched_movies_count": 0,
            "most_exposed_content": {}
        }
        
        if user_id_str in self.user_exposure_history:
            current_time = time.time()
            user_history = self.user_exposure_history[user_id_str]
            
            recent_count_6h = 0
            recent_count_24h = 0
            
            for movie_id, exposure_time in user_history.items():
                hours_since = (current_time - exposure_time) / 3600
                if hours_since < 6:
                    recent_count_6h += 1
                if hours_since < 24:
                    recent_count_24h += 1
            
            stats.update({
                "total_exposures": len(user_history),
                "recent_exposures_6h": recent_count_6h,
                "recent_exposures_24h": recent_count_24h
            })
        
        # Thêm searched movies count
        if user_id_str in self.searched_movies:
            stats["searched_movies_count"] = len(self.searched_movies[user_id_str])
        
        # Most exposed content
        stats["most_exposed_content"] = dict(sorted(self.content_exposure_count.items(), 
                                                  key=lambda x: x[1], reverse=True)[:3])
        
        return stats

    def cleanup_old_entries(self, max_age_hours: int = 720):
        """Dọn dẹp entries cũ hơn max_age_hours"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        cleaned_count = 0
        for user_id in list(self.user_exposure_history.keys()):
            original_count = len(self.user_exposure_history[user_id])
            self.user_exposure_history[user_id] = {
                movie_id: exposure_time 
                for movie_id, exposure_time in self.user_exposure_history[user_id].items()
                if exposure_time > cutoff_time
            }
            cleaned_count += (original_count - len(self.user_exposure_history[user_id]))
            
            # Remove empty user entries
            if not self.user_exposure_history[user_id]:
                del self.user_exposure_history[user_id]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old Baader-Meinhoff entries")
        
        return cleaned_count

# Global instance
baader_meinhof_engine = BaaderMeinhofEngine(search_boost=1.5)