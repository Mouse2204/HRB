import pickle
import os
import hashlib
from functools import wraps
from datetime import datetime, timedelta

class cacheManager:
    def __init__(self, cache_dir="cache", expiry_hours=24):
        self.cache_dir = cache_dir
        self.expiry_hours = expiry_hours
        os.makedirs(cache_dir, exist_ok=True)
        # THÊM: Load cache index khi khởi tạo
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index từ file"""
        index_file = os.path.join(self.cache_dir, "cache_index.pkl")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def _save_cache_index(self):
        """Lưu cache index vào file"""
        index_file = os.path.join(self.cache_dir, "cache_index.pkl")
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except:
            pass
    
    def _get_cache_key(self, name, *args, **kwargs):
        """Tạo cache key duy nhất"""
        key_str = f"{name}_{str(args)}_{str(kwargs)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Kiểm tra trong cache index trước
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                if datetime.now() - cache_info['timestamp'] < timedelta(hours=self.expiry_hours):
                    if os.path.exists(cache_file):
                        try:
                            with open(cache_file, 'rb') as f:
                                print(f"📁 Loading from cache: {func.__name__}")
                                return pickle.load(f)
                        except:
                            # Xóa cache bị lỗi
                            del self.cache_index[cache_key]
            
            # Chạy function và cache kết quả
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                # Cập nhật cache index
                self.cache_index[cache_key] = {
                    'timestamp': datetime.now(),
                    'function': func.__name__
                }
                self._save_cache_index()
            except Exception as e:
                print(f"⚠️ Cache save failed: {e}")
                
            return result
        return wrapper

# SINGLETON INSTANCE - quan trọng!
cache_manager = cacheManager()