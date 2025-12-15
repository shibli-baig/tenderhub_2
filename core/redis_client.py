"""
Redis client configuration for session storage and caching.
"""

import os
import redis
import logging
from typing import Optional

# Suppress verbose Redis logs
logging.getLogger('redis').setLevel(logging.WARNING)

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Initialize Redis client
try:
    redis_client = redis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    # Test connection
    redis_client.ping()
    # Only show success in debug mode
    logging.debug("✓ Redis connection established")
except (redis.ConnectionError, redis.TimeoutError) as e:
    # Suppress warning in development
    logging.debug(f"Redis connection failed: {e}. Using in-memory storage.")
    redis_client = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client instance."""
    return redis_client


def is_redis_available() -> bool:
    """Check if Redis is available."""
    return redis_client is not None
