import functools
import hashlib
import json

_MEMORY_CACHE = {}

def hash_dict(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode('utf-8')).hexdigest()

def simple_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}_{hash(args)}_{hash_dict(kwargs)}"
        if key not in _MEMORY_CACHE:
            _MEMORY_CACHE[key] = func(*args, **kwargs)
        return _MEMORY_CACHE[key]
    return wrapper

def clear_cache():
    global _MEMORY_CACHE
    _MEMORY_CACHE.clear()
