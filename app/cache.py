import re
import json
import time
import hashlib
import threading
from typing import Optional, Dict, Any
from app.config import settings


def normalize_query(query: str) -> str:
    """Normalizes query for semantic matching (lowercase, alphanumeric, collapsed spaces)."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    return re.sub(r"\s+", " ", q)


class InMemoryLRUCache:
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, capacity: int = 500, default_ttl: int = 3600):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            entry = self.cache.get(key)
            if not entry:
                return None
            if time.time() > entry["expires_at"]:
                del self.cache[key]
                return None
            # Move to MRU
            self.cache[key] = entry
            return entry["data"]

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        with self.lock:
            ttl_seconds = ttl or self.default_ttl
            if len(self.cache) >= self.capacity and key not in self.cache:
                # Evict oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = {
                "data": data,
                "expires_at": time.time() + ttl_seconds
            }

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()


class SemanticCache:
    """Unified cache dispatcher supporting Redis and In-Memory fallback."""

    def __init__(self):
        self.redis_client = None
        self.memory_cache = InMemoryLRUCache()
        self.cache_type = "in-memory"

        if settings.REDIS_URL:
            try:
                import redis
                self.redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
                self.redis_client.ping()
                self.cache_type = "redis"
            except Exception:
                self.redis_client = None
                self.cache_type = "in-memory"

    def _make_key(self, query: str, mode: str) -> str:
        norm = normalize_query(query)
        hash_val = hashlib.sha256(f"{norm}:{mode}".encode("utf-8")).hexdigest()
        return f"rag:cache:{hash_val}"

    def get(self, query: str, mode: str = "hybrid") -> Optional[Dict[str, Any]]:
        key = self._make_key(query, mode)
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass
        return self.memory_cache.get(key)

    def set(self, query: str, data: Dict[str, Any], mode: str = "hybrid", ttl: int = 3600) -> None:
        key = self._make_key(query, mode)
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(data))
                return
            except Exception:
                pass
        self.memory_cache.set(key, data, ttl=ttl)

    def clear(self) -> None:
        """Clears all cached queries from Redis and memory."""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass
        self.memory_cache.clear()


# Global cache instance
cache = SemanticCache()
