"""
Caching utilities using Redis with fallback to in-memory caching.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from core.redis_client import get_redis_client, is_redis_available


# URL to State mapping for tender sources
URL_TO_STATE = {
    "https://eproc.rajasthan.gov.in": "Rajasthan",
    "https://mptenders.gov.in": "Madhya Pradesh",
    "https://jharkhandtenders.gov.in/": "Jharkhand",
    "https://meghalayatenders.gov.in/": "Meghalaya",
    "http://www.eprocure.goa.gov.in": "Goa",
    "https://tntenders.gov.in/": "Tamil Nadu",
    "https://hptenders.gov.in": "Himachal Pradesh",
    "https://assamtenders.gov.in/": "Assam",
    "http://www.jktenders.gov.in": "Jammu and Kashmir",
    "https://govtprocurement.delhi.gov.in/nicgep/app": "Delhi",
    "https://etender.up.nic.in/nicgep/app": "Uttar Pradesh",
    "http://uktenders.gov.in": "Uttarakhand",
    "https://etenders.hry.nic.in": "Haryana",
    "https://etenders.kerala.gov.in": "Kerala",
    "https://mahatenders.gov.in/nicgep/app": "Maharashtra",
    "https://wbtenders.gov.in/nicgep/app": "West Bengal",
    "https://manipurtenders.gov.in/": "Manipur",
    "https://eproc.punjab.gov.in/nicgep/app": "Punjab",
    "https://etenders.chd.nic.in/nicgep/app": "Chandigarh",
    "https://nagalandtenders.gov.in": "Nagaland"
}

# Reverse mapping for state name to URL (for backend filtering)
STATE_TO_URL = {v: k for k, v in URL_TO_STATE.items()}


class FilterCache:
    """Cache for filter options (categories, states, sources, etc.)"""

    def __init__(self):
        self._memory_cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._ttl = 3600  # 1 hour in seconds

    def get_filter_options(self, db) -> Dict[str, list]:
        """
        Get filter options with caching.
        Returns dict with categories, states, sources, and product_categories.
        """
        from database import TenderDB
        from sqlalchemy import func, distinct, or_

        cache_key = 'tender:filters'

        # Try Redis first
        redis_client = get_redis_client()
        if redis_client and is_redis_available():
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"⚠ Redis cache retrieval failed: {e}, checking in-memory")

        # Check in-memory cache
        if cache_key in self._memory_cache:
            cache_age = (datetime.now() - self._cache_time.get(cache_key, datetime.now())).seconds
            if cache_age < self._ttl:
                return self._memory_cache[cache_key]

        # Query database if cache miss
        sources_raw = [s[0] for s in db.query(TenderDB.source).distinct().all() if s[0]]

        # Extract unique Product Category values from work_item_details JSONB
        # Handle both "Product Category" and "PRODUCT CATEGORY" key variations
        # Normalize values (trim whitespace) to ensure consistency
        product_categories_set = set()
        
        # Query for "Product Category" key
        pc_results = db.query(
            distinct(TenderDB.work_item_details['Product Category'].astext)
        ).filter(
            TenderDB.work_item_details['Product Category'].isnot(None),
            TenderDB.work_item_details['Product Category'].astext != '',
            TenderDB.work_item_details['Product Category'].astext != 'NA'
        ).all()
        
        for result in pc_results:
            if result[0]:
                # Trim whitespace and add to set
                normalized_value = result[0].strip()
                if normalized_value:
                    product_categories_set.add(normalized_value)
        
        # Query for "PRODUCT CATEGORY" key (uppercase variation)
        pc_upper_results = db.query(
            distinct(TenderDB.work_item_details['PRODUCT CATEGORY'].astext)
        ).filter(
            TenderDB.work_item_details['PRODUCT CATEGORY'].isnot(None),
            TenderDB.work_item_details['PRODUCT CATEGORY'].astext != '',
            TenderDB.work_item_details['PRODUCT CATEGORY'].astext != 'NA'
        ).all()
        
        for result in pc_upper_results:
            if result[0]:
                # Trim whitespace and add to set
                normalized_value = result[0].strip()
                if normalized_value:
                    product_categories_set.add(normalized_value)
        
        # Debug: Log found product categories, especially "Consultancy" variations
        if product_categories_set:
            consultancy_variants = [pc for pc in product_categories_set if 'consultancy' in pc.lower()]
            if consultancy_variants:
                print(f"DEBUG: Found Consultancy variants in Product Categories: {consultancy_variants}")

        filters = {
            'categories': [c[0] for c in db.query(TenderDB.category).distinct().all() if c[0]],
            'states': [s[0] for s in db.query(TenderDB.state).distinct().all() if s[0]],
            'sources': sources_raw,  # Keep URLs as-is
            'product_categories': sorted(list(product_categories_set))
        }

        # Store in Redis
        if redis_client and is_redis_available():
            try:
                redis_client.setex(cache_key, self._ttl, json.dumps(filters))
            except Exception as e:
                print(f"⚠ Redis cache storage failed: {e}")

        # Store in memory cache as fallback
        self._memory_cache[cache_key] = filters
        self._cache_time[cache_key] = datetime.now()

        return filters

    def invalidate_filters(self):
        """Invalidate filter cache (call when new tenders are added)"""
        cache_key = 'tender:filters'

        # Clear from Redis
        redis_client = get_redis_client()
        if redis_client and is_redis_available():
            try:
                redis_client.delete(cache_key)
            except Exception as e:
                print(f"⚠ Redis cache invalidation failed: {e}")

        # Clear from memory
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        if cache_key in self._cache_time:
            del self._cache_time[cache_key]


# Global instance
filter_cache = FilterCache()
