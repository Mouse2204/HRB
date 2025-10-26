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
    
    def _getcache(self, name, *args, **kwargs):
        key_str = f"{name}_{str(args)}_{str(kwargs)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._getcache(func.__name__, *args, **kwargs)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(hours=self.expiry_hours):
                    try:
                        with open(cache_file, 'rb') as f:
                            print(f"📁 Loading from cache: {func.__name__}")
                            return pickle.load(f)
                    except:
                        pass
            
            result = func(*args, **kwargs)

            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass
                
            return result
        return wrapper

cache_manager = cacheManager()
        
