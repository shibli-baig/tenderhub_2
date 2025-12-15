"""
Tender Recommendation Engine
Matches user's certificates and projects against tenders using keyword-based scoring.
Enhanced with fuzzy matching, synonym detection, and partial word matching.
"""

import re
import logging
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
from fuzzywuzzy import fuzz
from sqlalchemy.orm import Session
from database import CertificateDB, ProjectDB, TenderDB, TenderEmbeddingDB
from core.openai_wrapper import OpenAIWrapper

logger = logging.getLogger(__name__)


class TenderEmbeddingManager:
    """
    Manages OpenAI embeddings for tenders and user profiles with 3-tier caching.

    Caching Strategy:
    1. Memory Cache: Fast in-memory dictionary (session-scoped)
    2. Database Cache: PostgreSQL persistent cache (TenderEmbeddingDB table)
    3. API Generation: OpenAI API call (last resort, costs $0.13 per 1M tokens)

    Cost Optimization:
    - Batch API calls (up to 2048 texts per request)
    - SHA-256 hash-based deduplication
    - Access tracking for cache analytics
    - 3072-dimension embeddings for high accuracy
    """

    # Model configuration
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS = 3072

    # Cache configuration
    MEMORY_CACHE_SIZE = 1000  # Maximum items in memory cache
    DB_CACHE_TTL_DAYS = 30  # Cache validity period
    BATCH_SIZE = 2048  # Maximum texts per batch API call

    # Similarity thresholds
    MIN_SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity for match (25%) - lowered from 0.3 for looser matching

    def __init__(self, db: Session, openai_client: Optional[OpenAIWrapper] = None):
        """
        Initialize embedding manager with database and OpenAI client.

        Args:
            db: SQLAlchemy database session
            openai_client: OpenAIWrapper instance (creates new if None)
        """
        self.db = db
        self.openai_client = openai_client or OpenAIWrapper()

        # Memory cache: {text_hash: (embedding_array, timestamp)}
        self.memory_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}

        # Statistics
        self.stats = {
            "memory_hits": 0,
            "db_hits": 0,
            "api_calls": 0,
            "total_requests": 0
        }

        logger.info(f"TenderEmbeddingManager initialized with model: {self.EMBEDDING_MODEL}")

    def _compute_text_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of text for cache lookup.

        Args:
            text: Input text to hash

        Returns:
            64-character hexadecimal hash string
        """
        # Normalize text before hashing to improve cache hits
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Serialize numpy embedding array to bytes for database storage.

        Args:
            embedding: NumPy array of float32 values

        Returns:
            Binary representation (12,288 bytes for 3072 dimensions)
        """
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """
        Deserialize bytes to numpy embedding array.

        Args:
            data: Binary data from database

        Returns:
            NumPy array of shape (3072,)
        """
        return np.frombuffer(data, dtype=np.float32)

    def get_embedding(self, text: str, text_type: str = 'tender') -> Optional[np.ndarray]:
        """
        Get embedding for text using 3-tier caching strategy.

        Tier 1: Check memory cache (fastest)
        Tier 2: Check database cache (fast)
        Tier 3: Generate via API (slowest, costs money)

        Args:
            text: Text to embed
            text_type: 'tender' or 'profile' for cache organization

        Returns:
            NumPy array of shape (3072,) or None on error
        """
        self.stats["total_requests"] += 1

        if not text or not text.strip():
            logger.warning("Empty text provided to get_embedding()")
            return None

        text_hash = self._compute_text_hash(text)

        # Tier 1: Memory cache lookup
        if text_hash in self.memory_cache:
            embedding, timestamp = self.memory_cache[text_hash]
            self.stats["memory_hits"] += 1
            logger.debug(f"Memory cache HIT for {text_type} (hash: {text_hash[:8]}...)")
            return embedding

        # Tier 2: Database cache lookup
        db_entry = self.db.query(TenderEmbeddingDB).filter(
            TenderEmbeddingDB.text_hash == text_hash,
            TenderEmbeddingDB.text_type == text_type
        ).first()

        if db_entry:
            # Check if cache is still valid
            age = datetime.utcnow() - db_entry.created_at
            if age.days < self.DB_CACHE_TTL_DAYS:
                embedding = self._deserialize_embedding(db_entry.embedding)

                # Update access statistics
                db_entry.last_accessed = datetime.utcnow()
                db_entry.access_count += 1
                self.db.commit()

                # Store in memory cache
                self.memory_cache[text_hash] = (embedding, datetime.utcnow())
                self._cleanup_memory_cache()

                self.stats["db_hits"] += 1
                logger.debug(f"Database cache HIT for {text_type} (hash: {text_hash[:8]}..., age: {age.days} days)")
                return embedding
            else:
                # Cache expired, delete it
                logger.info(f"Cache expired for {text_type} (hash: {text_hash[:8]}..., age: {age.days} days)")
                self.db.delete(db_entry)
                self.db.commit()

        # Tier 3: Generate via OpenAI API
        logger.info(f"Cache MISS - Generating embedding via API for {text_type} (text length: {len(text)} chars)")
        embedding = self._generate_embedding_via_api(text, text_type, text_hash)

        return embedding

    def _generate_embedding_via_api(self, text: str, text_type: str, text_hash: str) -> Optional[np.ndarray]:
        """
        Generate embedding by calling OpenAI API and cache the result.

        Args:
            text: Text to embed
            text_type: 'tender' or 'profile'
            text_hash: Pre-computed hash for caching

        Returns:
            NumPy array of shape (3072,) or None on error
        """
        try:
            self.stats["api_calls"] += 1

            # Call OpenAI API (will use openai_wrapper.py's create_embedding method)
            response = self.openai_client.create_embedding(
                text=text,
                model=self.EMBEDDING_MODEL
            )

            if not response or 'embedding' not in response:
                logger.error(f"Failed to generate embedding via API for {text_type}")
                return None

            embedding_list = response['embedding']
            embedding = np.array(embedding_list, dtype=np.float32)

            # Verify dimensions
            if len(embedding) != self.EMBEDDING_DIMENSIONS:
                logger.error(f"Unexpected embedding dimensions: {len(embedding)} != {self.EMBEDDING_DIMENSIONS}")
                return None

            logger.info(f"Successfully generated embedding via API for {text_type} (dimensions: {len(embedding)})")

            # Store in database cache
            db_entry = TenderEmbeddingDB(
                text_hash=text_hash,
                text_type=text_type,
                embedding=self._serialize_embedding(embedding),
                model=self.EMBEDDING_MODEL,
                dimensions=self.EMBEDDING_DIMENSIONS,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1
            )
            self.db.add(db_entry)
            self.db.commit()
            logger.debug(f"Stored embedding in database cache (hash: {text_hash[:8]}...)")

            # Store in memory cache
            self.memory_cache[text_hash] = (embedding, datetime.utcnow())
            self._cleanup_memory_cache()

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding via API: {e}", exc_info=True)
            return None

    def _cleanup_memory_cache(self):
        """
        Remove oldest entries from memory cache if size exceeds limit.
        Uses LRU (Least Recently Used) eviction strategy.
        """
        if len(self.memory_cache) > self.MEMORY_CACHE_SIZE:
            # Sort by timestamp (oldest first)
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1][1]  # x[1][1] is timestamp
            )

            # Remove oldest 20% of entries
            num_to_remove = int(self.MEMORY_CACHE_SIZE * 0.2)
            for text_hash, _ in sorted_items[:num_to_remove]:
                del self.memory_cache[text_hash]

            logger.debug(f"Memory cache cleanup: removed {num_to_remove} oldest entries")

    def batch_generate_embeddings(self, texts: List[str], text_type: str = 'tender') -> List[Tuple[str, Optional[np.ndarray]]]:
        """
        Generate embeddings for multiple texts in batch for efficiency.

        Uses 3-tier caching for each text, then batches API calls for cache misses.

        Args:
            texts: List of texts to embed
            text_type: 'tender' or 'profile'

        Returns:
            List of (text, embedding) tuples in same order as input
        """
        logger.info(f"Batch embedding request for {len(texts)} {text_type} texts")

        results: List[Tuple[str, Optional[np.ndarray]]] = []
        texts_needing_api: List[Tuple[int, str, str]] = []  # (index, text, hash)

        # First pass: Check cache for each text
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                results.append((text, None))
                continue

            text_hash = self._compute_text_hash(text)

            # Try to get from cache (memory or database)
            embedding = self.get_embedding(text, text_type)

            if embedding is not None:
                results.append((text, embedding))
            else:
                # Need to generate via API
                texts_needing_api.append((idx, text, text_hash))
                results.append((text, None))  # Placeholder

        # Second pass: Batch generate embeddings for cache misses
        if texts_needing_api:
            logger.info(f"Generating {len(texts_needing_api)} embeddings via batch API call")

            try:
                # Extract just the texts for batch API call
                batch_texts = [text for _, text, _ in texts_needing_api]

                # Call batch API
                self.stats["api_calls"] += 1
                batch_response = self.openai_client.batch_create_embeddings(
                    texts=batch_texts,
                    model=self.EMBEDDING_MODEL
                )

                if not batch_response or 'embeddings' not in batch_response:
                    logger.error("Failed to generate batch embeddings via API")
                    # Keep None placeholders in results
                else:
                    embeddings_list = batch_response['embeddings']
                    logger.info(f"Successfully generated {len(embeddings_list)} embeddings via batch API")

                    # Store each embedding in cache and update results
                    for i, (idx, text, text_hash) in enumerate(texts_needing_api):
                        if i < len(embeddings_list):
                            embedding = np.array(embeddings_list[i], dtype=np.float32)

                            # Verify dimensions
                            if len(embedding) == self.EMBEDDING_DIMENSIONS:
                                # Store in database cache
                                db_entry = TenderEmbeddingDB(
                                    text_hash=text_hash,
                                    text_type=text_type,
                                    embedding=self._serialize_embedding(embedding),
                                    model=self.EMBEDDING_MODEL,
                                    dimensions=self.EMBEDDING_DIMENSIONS,
                                    created_at=datetime.utcnow(),
                                    last_accessed=datetime.utcnow(),
                                    access_count=1
                                )
                                self.db.add(db_entry)

                                # Store in memory cache
                                self.memory_cache[text_hash] = (embedding, datetime.utcnow())

                                # Update result
                                results[idx] = (text, embedding)
                            else:
                                logger.error(f"Unexpected embedding dimensions: {len(embedding)} != {self.EMBEDDING_DIMENSIONS}")

                    # Commit all database entries at once
                    self.db.commit()
                    self._cleanup_memory_cache()

            except Exception as e:
                logger.error(f"Error in batch embedding generation: {e}", exc_info=True)
                # Keep None placeholders in results

        success_count = sum(1 for _, emb in results if emb is not None)
        logger.info(f"Batch embedding completed: {success_count}/{len(texts)} successful")

        return results

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Cosine similarity ranges from -1 to 1:
        - 1: Identical vectors (perfect match)
        - 0: Orthogonal vectors (no similarity)
        - -1: Opposite vectors (opposite meaning)

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1 (we use absolute value)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            logger.warning("Zero-norm vector in cosine_similarity calculation")
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to [0, 1] range (use absolute value to handle negative similarities)
        return max(0.0, min(1.0, float(abs(similarity))))

    def get_user_profile_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Generate embedding for user's complete profile (all projects).

        Combines text from:
        - project_description
        - complete_scope_of_work
        - services_rendered

        Args:
            user_id: User ID to generate profile for

        Returns:
            NumPy array of shape (3072,) or None if no projects
        """
        logger.info(f"Generating user profile embedding for user_id: {user_id}")

        # Fetch user's projects
        projects = self.db.query(ProjectDB).filter(
            ProjectDB.user_id == user_id,
            ProjectDB.completion_status.in_(['complete', 'completed_by_user'])
        ).all()

        if not projects:
            logger.warning(f"No completed projects found for user {user_id}")
            return None

        # Combine all project texts
        text_parts = []

        for project in projects:
            if project.project_description:
                text_parts.append(project.project_description)

            if project.complete_scope_of_work:
                text_parts.append(project.complete_scope_of_work)

            if project.services_rendered and isinstance(project.services_rendered, dict):
                for service_list in project.services_rendered.values():
                    if isinstance(service_list, list):
                        text_parts.extend([s for s in service_list if s])

        if not text_parts:
            logger.warning(f"No project text found for user {user_id}")
            return None

        # Combine all text (limit to ~8000 tokens ≈ 32,000 characters)
        combined_text = " ".join(text_parts)
        if len(combined_text) > 32000:
            combined_text = combined_text[:32000]
            logger.info(f"Truncated user profile text to 32,000 characters")

        logger.info(f"User profile text: {len(text_parts)} parts, {len(combined_text)} total chars")

        # Generate embedding (uses caching)
        embedding = self.get_embedding(combined_text, text_type='profile')

        return embedding

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with hit rates and call counts
        """
        total = self.stats["total_requests"]

        if total == 0:
            return {
                "total_requests": 0,
                "memory_hit_rate": 0.0,
                "db_hit_rate": 0.0,
                "api_call_rate": 0.0,
                "cache_hit_rate": 0.0
            }

        return {
            "total_requests": total,
            "memory_hits": self.stats["memory_hits"],
            "db_hits": self.stats["db_hits"],
            "api_calls": self.stats["api_calls"],
            "memory_hit_rate": (self.stats["memory_hits"] / total) * 100,
            "db_hit_rate": (self.stats["db_hits"] / total) * 100,
            "api_call_rate": (self.stats["api_calls"] / total) * 100,
            "cache_hit_rate": ((self.stats["memory_hits"] + self.stats["db_hits"]) / total) * 100,
            "memory_cache_size": len(self.memory_cache)
        }


class TenderRecommendationScorer:
    """
    Scores tenders against user's expertise profile built from certificates and projects.
    Uses keyword matching with fuzzy logic for recommendations.
    """

    # Scoring weights (total = 100)
    # Updated for semantic similarity integration
    WEIGHT_SCOPE_TEXT = 35       # Text matching (was 40)
    WEIGHT_SEMANTIC = 15         # NEW: AI semantic similarity
    WEIGHT_SECTOR = 20           # Sector/category matching
    WEIGHT_SERVICES = 15         # Services matching (was 20)
    WEIGHT_LOCATION = 10         # Geographic matching
    WEIGHT_AUTHORITY = 5         # Client/authority matching

    # Matching thresholds (lowered further for looser matching per user request)
    FUZZY_THRESHOLD = 55  # Minimum fuzzy match score (0-100) - lowered from 60 to 55
    FUZZY_STRONG_THRESHOLD = 75  # Strong match threshold - lowered from 80 to 75
    FUZZY_EXACT_THRESHOLD = 95  # Near-exact match threshold
    PHRASE_BONUS_MULTIPLIER = 1.5

    # Minimum word length for keyword extraction
    MIN_WORD_LENGTH = 3

    # Stop words to exclude
    STOP_WORDS = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was',
        'one', 'our', 'out', 'has', 'been', 'have', 'this', 'that', 'with', 'from',
        'will', 'would', 'there', 'their', 'what', 'which', 'when', 'where', 'who',
        'how', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'ltd', 'pvt', 'company', 'limited', 'private', 'department', 'govt', 'government'
    }

    # Domain-specific synonym mappings for semantic matching
    SECTOR_SYNONYMS = {
        # Petroleum/Oil & Gas (expanded with industry terms)
        'petroleum': {'oil', 'gas', 'petrol', 'hydrocarbon', 'fossil', 'energy', 'refinery', 'pipeline', 'lng', 'lpg', 'iocl', 'bpcl', 'hpcl', 'ongc', 'upstream', 'downstream', 'refining', 'petrochemical', 'fuel', 'naphtha'},
        'oil': {'petroleum', 'gas', 'petrol', 'hydrocarbon', 'fossil', 'energy', 'refinery', 'crude', 'iocl', 'bpcl', 'hpcl', 'ongc', 'fuel', 'naphtha'},
        'gas': {'petroleum', 'oil', 'natural', 'lng', 'lpg', 'hydrocarbon', 'energy', 'pipeline', 'iocl', 'bpcl', 'hpcl', 'ongc'},
        'petrol': {'petroleum', 'oil', 'gas', 'fuel', 'energy', 'iocl', 'bpcl', 'hpcl'},
        'hydrocarbon': {'petroleum', 'oil', 'gas', 'fossil', 'energy', 'refining', 'petrochemical'},
        'naphtha': {'petroleum', 'oil', 'gas', 'refining', 'petrochemical', 'fuel'},
        'refining': {'petroleum', 'oil', 'refinery', 'petrochemical', 'processing'},
        'petrochemical': {'petroleum', 'oil', 'chemical', 'refining', 'processing'},
        'upstream': {'petroleum', 'oil', 'gas', 'exploration', 'drilling', 'production'},
        'downstream': {'petroleum', 'oil', 'gas', 'refining', 'distribution', 'marketing'},
        'iocl': {'petroleum', 'oil', 'gas', 'indian oil', 'refinery', 'fuel'},
        'bpcl': {'petroleum', 'oil', 'gas', 'bharat petroleum', 'refinery', 'fuel'},
        'hpcl': {'petroleum', 'oil', 'gas', 'hindustan petroleum', 'refinery', 'fuel'},
        'ongc': {'petroleum', 'oil', 'gas', 'exploration', 'drilling', 'upstream'},

        # Infrastructure/Construction
        'infrastructure': {'construction', 'building', 'development', 'civil', 'works', 'engineering', 'project'},
        'construction': {'infrastructure', 'building', 'development', 'civil', 'works', 'engineering'},
        'building': {'construction', 'infrastructure', 'development', 'civil', 'works'},
        'civil': {'construction', 'infrastructure', 'building', 'engineering', 'works'},

        # Water/Sanitation
        'water': {'irrigation', 'sanitation', 'sewage', 'drainage', 'supply', 'treatment', 'waste', 'hydro'},
        'irrigation': {'water', 'agriculture', 'farming', 'canal', 'drainage'},
        'sanitation': {'water', 'sewage', 'waste', 'drainage', 'treatment'},

        # Power/Energy
        'power': {'energy', 'electricity', 'electrical', 'grid', 'transmission', 'distribution', 'generation'},
        'energy': {'power', 'electricity', 'electrical', 'renewable', 'solar', 'wind'},
        'electricity': {'power', 'energy', 'electrical', 'grid', 'transmission'},
        'solar': {'energy', 'power', 'renewable', 'photovoltaic', 'pv'},
        'wind': {'energy', 'power', 'renewable', 'turbine'},

        # IT/Technology
        'software': {'technology', 'it', 'digital', 'application', 'system', 'computer', 'program'},
        'hardware': {'technology', 'it', 'equipment', 'device', 'computer', 'system'},
        'it': {'information', 'technology', 'software', 'hardware', 'digital', 'computer'},
        'digital': {'technology', 'it', 'software', 'electronic', 'computer'},

        # Healthcare
        'medical': {'health', 'healthcare', 'hospital', 'clinic', 'pharmaceutical', 'medicine'},
        'health': {'medical', 'healthcare', 'hospital', 'clinic', 'wellness'},
        'hospital': {'medical', 'health', 'healthcare', 'clinic', 'care'},

        # Transportation
        'road': {'highway', 'transport', 'transportation', 'infrastructure', 'traffic'},
        'highway': {'road', 'transport', 'transportation', 'infrastructure'},
        'railway': {'rail', 'metro', 'train', 'transport', 'transportation'},
        'transport': {'transportation', 'road', 'railway', 'highway', 'logistics'},

        # Consulting/Services
        'consulting': {'consultancy', 'consultant', 'advisory', 'service', 'professional'},
        'consultancy': {'consulting', 'consultant', 'advisory', 'service'},
        'consultant': {'consulting', 'consultancy', 'advisory', 'professional'},
    }

    # Common abbreviation expansions (expanded with petroleum industry)
    ABBREVIATION_MAP = {
        'o&g': 'oil gas',
        'iocl': 'indian oil corporation',
        'bpcl': 'bharat petroleum corporation',
        'hpcl': 'hindustan petroleum corporation',
        'ongc': 'oil natural gas corporation',
        'it': 'information technology',
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'iot': 'internet of things',
        'erp': 'enterprise resource planning',
        'crm': 'customer relationship management',
        'hvac': 'heating ventilation air conditioning',
        'pm': 'project management',
        'qa': 'quality assurance',
        'qc': 'quality control',
        'r&d': 'research development',
        'hr': 'human resources',
        'pv': 'photovoltaic',
        'lng': 'liquefied natural gas',
        'lpg': 'liquefied petroleum gas',
    }

    def __init__(
        self,
        db: Session,
        user_id: str,
        embedding_manager: Optional[TenderEmbeddingManager] = None,
        user_profile_embedding: Optional[np.ndarray] = None
    ):
        """
        Initialize scorer with database session and user ID.

        Args:
            db: SQLAlchemy database session
            user_id: User ID to build profile for
            embedding_manager: Optional TenderEmbeddingManager for semantic similarity
            user_profile_embedding: Optional pre-computed user profile embedding
        """
        self.db = db
        self.user_id = user_id
        self.user_profile: Optional[Dict[str, Any]] = None
        self.embedding_manager = embedding_manager
        self.user_profile_embedding = user_profile_embedding

    def build_user_profile(self) -> Dict[str, Any]:
        """
        Build user expertise profile from certificates and projects.

        Returns:
            Dict containing sectors, services, locations, keywords, phrases, clients, value_range
        """
        if self.user_profile:
            return self.user_profile

        profile = {
            "sectors": set(),
            "services": set(),
            "locations": set(),
            "keywords": set(),
            "phrases": set(),
            "clients": set(),
            "value_range": {"min": float('inf'), "max": 0}
        }

        # Extract from certificates
        certificates = self.db.query(CertificateDB).filter(
            CertificateDB.user_id == self.user_id,
            CertificateDB.confidence_score >= 0.7
        ).all()

        for cert in certificates:
            # Sectors
            if cert.sectors:
                profile["sectors"].update([s.strip() for s in cert.sectors if s and s.strip()])
            if cert.sub_sectors:
                profile["sectors"].update([s.strip() for s in cert.sub_sectors if s and s.strip()])

            # Services
            if cert.services_rendered:
                if isinstance(cert.services_rendered, list):
                    profile["services"].update([s.strip() for s in cert.services_rendered if s and str(s).strip()])
                elif isinstance(cert.services_rendered, str):
                    profile["services"].add(cert.services_rendered.strip())

            # Location
            if cert.location:
                profile["locations"].add(cert.location.strip())

            # Keywords from scope
            if cert.scope_of_work:
                keywords, phrases = self._extract_keywords_and_phrases(cert.scope_of_work)
                profile["keywords"].update(keywords)
                profile["phrases"].update(phrases)

            if cert.extracted_text:
                keywords, phrases = self._extract_keywords_and_phrases(cert.extracted_text)
                profile["keywords"].update(keywords)
                profile["phrases"].update(phrases)

            # Client
            if cert.client_name:
                profile["clients"].add(cert.client_name.strip())

            # Value range
            if cert.project_value:
                profile["value_range"]["min"] = min(profile["value_range"]["min"], cert.project_value)
                profile["value_range"]["max"] = max(profile["value_range"]["max"], cert.project_value)

        # Extract from projects
        projects = self.db.query(ProjectDB).filter(
            ProjectDB.user_id == self.user_id,
            ProjectDB.completion_status.in_(['complete', 'completed_by_user'])
        ).all()

        for project in projects:
            # Sectors
            if project.sector:
                profile["sectors"].add(project.sector.strip())
            if project.sub_sector:
                profile["sectors"].add(project.sub_sector.strip())

            # Services
            if project.services_rendered and isinstance(project.services_rendered, dict):
                for service_list in project.services_rendered.values():
                    if isinstance(service_list, list):
                        profile["services"].update([s.strip() for s in service_list if s and str(s).strip()])

            # Locations
            if project.states:
                profile["locations"].update([s.strip() for s in project.states if s and str(s).strip()])
            if project.cities:
                profile["locations"].update([c.strip() for c in project.cities if c and str(c).strip()])

            # Keywords from descriptions
            if project.project_description:
                keywords, phrases = self._extract_keywords_and_phrases(project.project_description)
                profile["keywords"].update(keywords)
                profile["phrases"].update(phrases)

            if project.complete_scope_of_work:
                keywords, phrases = self._extract_keywords_and_phrases(project.complete_scope_of_work)
                profile["keywords"].update(keywords)
                profile["phrases"].update(phrases)

            # Client
            if project.client_name:
                profile["clients"].add(project.client_name.strip())

            # Value range
            if project.consultancy_fee:
                profile["value_range"]["min"] = min(profile["value_range"]["min"], project.consultancy_fee)
                profile["value_range"]["max"] = max(profile["value_range"]["max"], project.consultancy_fee)
            if project.project_cost:
                profile["value_range"]["min"] = min(profile["value_range"]["min"], project.project_cost)
                profile["value_range"]["max"] = max(profile["value_range"]["max"], project.project_cost)

        # Handle empty value range
        if profile["value_range"]["min"] == float('inf'):
            profile["value_range"] = {"min": 0, "max": 0}

        self.user_profile = profile
        return profile

    def _extract_keywords_and_phrases(self, text: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract keywords and 2-3 word phrases from text with abbreviation expansion.

        Args:
            text: Input text to extract from

        Returns:
            Tuple of (keywords_set, phrases_set)
        """
        if not text:
            return set(), set()

        # Clean and normalize text
        text = text.lower()

        # Expand abbreviations BEFORE removing punctuation
        for abbr, expansion in self.ABBREVIATION_MAP.items():
            # Use word boundaries to avoid partial matches
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)

        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        words = text.split()

        # Extract keywords (single words) - lowered threshold to catch short critical words like "oil"
        keywords = {
            word for word in words
            if len(word) >= 2 and word not in self.STOP_WORDS  # Changed from MIN_WORD_LENGTH (3) to 2
        }

        # Extract partial word stems for compound words (e.g., "petro" from "petroleum")
        for word in list(keywords):  # Use list() to avoid modifying set during iteration
            if len(word) >= 4:  # Changed from 6 to 4 to catch more stems (e.g., "pipe" from "pipeline")
                # Add first 4 characters as stem for partial matching (changed from 5)
                keywords.add(word[:4])

        # Extract 2-3 word phrases
        phrases = set()
        for i in range(len(words) - 1):
            # 2-word phrases
            phrase = f"{words[i]} {words[i+1]}"
            if all(word not in self.STOP_WORDS for word in [words[i], words[i+1]]):
                phrases.add(phrase)

            # 3-word phrases
            if i < len(words) - 2:
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if all(word not in self.STOP_WORDS for word in [words[i], words[i+1], words[i+2]]):
                    phrases.add(phrase)

        return keywords, phrases

    def _fuzzy_match_score(self, str1: str, str2: str) -> float:
        """
        Calculate fuzzy match score with multi-tier thresholds.

        Args:
            str1, str2: Strings to compare

        Returns:
            Match score 0.0-1.0 (0% to 100% of full weight)
        """
        ratio = fuzz.ratio(str1.lower(), str2.lower())

        if ratio >= self.FUZZY_EXACT_THRESHOLD:
            return 1.0  # Near-exact match
        elif ratio >= self.FUZZY_STRONG_THRESHOLD:
            return 0.9  # Strong fuzzy match
        elif ratio >= self.FUZZY_THRESHOLD:
            return 0.7  # Medium fuzzy match
        else:
            return 0.0  # No match

    def _expand_with_synonyms(self, keywords: Set[str]) -> Set[str]:
        """
        Expand keyword set with domain-specific synonyms.

        Args:
            keywords: Original keyword set

        Returns:
            Expanded keyword set including synonyms
        """
        expanded = set(keywords)

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower in self.SECTOR_SYNONYMS:
                expanded.update(self.SECTOR_SYNONYMS[keyword_lower])

        return expanded

    def _partial_word_match(self, user_word: str, tender_words: Set[str]) -> bool:
        """
        Check if user_word partially matches any tender word.

        Uses substring matching and common prefix matching for compound words.

        Args:
            user_word: Word from user profile
            tender_words: Set of words from tender

        Returns:
            True if partial match found
        """
        user_lower = user_word.lower()

        for tender_word in tender_words:
            tender_lower = tender_word.lower()

            # Skip if either word is too short for partial matching
            if len(user_lower) < 4 or len(tender_lower) < 4:
                continue

            # Check substring match (e.g., "petrol" in "petroleum")
            if user_lower in tender_lower or tender_lower in user_lower:
                return True

            # Check common prefix for related words (e.g., "petro..." matches)
            if len(user_lower) >= 5 and len(tender_lower) >= 5:
                if user_lower[:5] == tender_lower[:5]:
                    return True

        return False

    def _passes_hybrid_filter(self, tender: TenderDB) -> Tuple[bool, str]:
        """
        Hybrid filter: Requires ≥1 keyword match OR same sector OR same state.

        Changed from AND to OR logic for looser matching to catch more relevant tenders.
        User requested: "fuzzy and loose matching" to show more results.

        Args:
            tender: TenderDB object to check

        Returns:
            Tuple of (passes_filter, reason_if_rejected)
        """
        if not self.user_profile:
            self.build_user_profile()

        # Extract tender text for keyword matching
        tender_text_parts = []
        if tender.title:
            tender_text_parts.append(tender.title)
        if tender.summary:
            tender_text_parts.append(tender.summary)
        if tender.work_item_details:
            if isinstance(tender.work_item_details, str):
                tender_text_parts.append(tender.work_item_details)
            elif isinstance(tender.work_item_details, dict):
                tender_text_parts.append(str(tender.work_item_details))

        if not tender_text_parts:
            return False, "No tender text to match"

        tender_text = " ".join(tender_text_parts)
        tender_keywords, _ = self._extract_keywords_and_phrases(tender_text)

        # REQUIREMENT 1: At least 1 keyword match
        keyword_matches = 0

        # Check exact keyword matches
        exact_matches = self.user_profile["keywords"] & tender_keywords
        keyword_matches += len(exact_matches)

        # Check synonym-based matches
        if keyword_matches == 0:
            expanded_user_keywords = self._expand_with_synonyms(self.user_profile["keywords"])
            expanded_tender_keywords = self._expand_with_synonyms(tender_keywords)
            synonym_matches = expanded_user_keywords & expanded_tender_keywords
            keyword_matches += len(synonym_matches)

        # Check partial word matches
        if keyword_matches == 0:
            for user_keyword in self.user_profile["keywords"]:
                if self._partial_word_match(user_keyword, tender_keywords):
                    keyword_matches += 1
                    break  # Just need 1 match

        # Check for matches - now using OR logic
        # REQUIREMENT: keyword match OR sector match OR location match
        sector_match = False
        location_match = False

        # Check sector match (expanded to check full tender text, not just structured fields)
        if self.user_profile["sectors"]:
            tender_sectors = set()
            if tender.category:
                tender_sectors.add(tender.category.lower().strip())
            if tender.tender_type:
                tender_sectors.add(tender.tender_type.lower().strip())
            if tender.tender_category:
                tender_sectors.add(tender.tender_category.lower().strip())

            # NEW: Also extract sectors from tender text (title + summary)
            tender_text_lower = tender_text.lower()
            for user_sector in self.user_profile["sectors"]:
                if user_sector.lower() in tender_text_lower:
                    tender_sectors.add(user_sector.lower())

            if tender_sectors:
                user_sectors_lower = {s.lower() for s in self.user_profile["sectors"]}

                # Exact match
                if tender_sectors & user_sectors_lower:
                    sector_match = True
                else:
                    # Synonym match
                    expanded_user = self._expand_with_synonyms(user_sectors_lower)
                    expanded_tender = self._expand_with_synonyms(tender_sectors)
                    if expanded_user & expanded_tender:
                        sector_match = True

        # Check location match
        if self.user_profile["locations"] and tender.state:
            tender_state = tender.state.lower().strip()
            user_locations_lower = {loc.lower() for loc in self.user_profile["locations"]}

            if tender_state in user_locations_lower:
                location_match = True

        # Changed to OR logic: Must have keyword OR sector OR location match
        if keyword_matches == 0 and not sector_match and not location_match:
            return False, "No keyword, sector, or location match found"

        # Passed! Build match description
        match_parts = []
        if keyword_matches > 0:
            match_parts.append(f"{keyword_matches} keywords")
        if sector_match:
            match_parts.append("sector")
        if location_match:
            match_parts.append("location")

        logger.debug(f"Hybrid filter PASSED: {' + '.join(match_parts)}")
        return True, " + ".join(match_parts)

    def _score_semantic_similarity(self, tender: TenderDB) -> Tuple[float, float]:
        """
        Score semantic similarity using OpenAI embeddings (0-15 points).

        Uses AI to understand meaning and context beyond keyword matching.
        For example, "water treatment plant" semantically matches "sewage processing facility"
        even without shared keywords.

        Args:
            tender: TenderDB object to score

        Returns:
            Tuple of (score, similarity_percentage) where:
                - score: 0-15 points based on WEIGHT_SEMANTIC
                - similarity_percentage: 0-100% raw cosine similarity
        """
        # Check if semantic scoring is available
        if not self.embedding_manager or not self.user_profile_embedding:
            logger.debug("Semantic scoring disabled (no embedding manager or user profile embedding)")
            return 0.0, 0.0

        # Collect tender text
        tender_text_parts = []
        if tender.title:
            tender_text_parts.append(tender.title)
        if tender.summary:
            tender_text_parts.append(tender.summary)
        if tender.work_item_details:
            if isinstance(tender.work_item_details, str):
                tender_text_parts.append(tender.work_item_details)
            elif isinstance(tender.work_item_details, dict):
                tender_text_parts.append(str(tender.work_item_details))

        if not tender_text_parts:
            logger.debug(f"No tender text for semantic scoring (tender {tender.id})")
            return 0.0, 0.0

        tender_text = " ".join(tender_text_parts)

        # Limit text length (OpenAI has token limits)
        if len(tender_text) > 8000:
            tender_text = tender_text[:8000]

        try:
            # Get tender embedding (uses 3-tier caching)
            tender_embedding = self.embedding_manager.get_embedding(tender_text, text_type='tender')

            if tender_embedding is None:
                logger.warning(f"Failed to generate embedding for tender {tender.id}")
                return 0.0, 0.0

            # Calculate cosine similarity
            similarity = self.embedding_manager.cosine_similarity(
                self.user_profile_embedding,
                tender_embedding
            )

            # Convert similarity (0-1) to score (0-15 points)
            # Apply minimum threshold
            if similarity < TenderEmbeddingManager.MIN_SIMILARITY_THRESHOLD:
                logger.debug(f"Semantic similarity below threshold: {similarity:.3f} < {TenderEmbeddingManager.MIN_SIMILARITY_THRESHOLD}")
                return 0.0, similarity * 100

            # Linear scaling: 0.3 similarity = 0 points, 1.0 similarity = 15 points
            # Formula: score = ((similarity - min_threshold) / (1.0 - min_threshold)) * max_points
            normalized_similarity = (similarity - TenderEmbeddingManager.MIN_SIMILARITY_THRESHOLD) / (
                1.0 - TenderEmbeddingManager.MIN_SIMILARITY_THRESHOLD
            )
            score = normalized_similarity * self.WEIGHT_SEMANTIC

            logger.debug(f"Semantic similarity: {similarity:.3f} ({similarity*100:.1f}%) → {score:.2f}/{self.WEIGHT_SEMANTIC} points")

            return score, similarity * 100

        except Exception as e:
            logger.error(f"Error calculating semantic similarity for tender {tender.id}: {e}", exc_info=True)
            return 0.0, 0.0

    def score_tender(self, tender: TenderDB) -> Tuple[float, Dict[str, Any]]:
        """
        Score a single tender against user's profile with comprehensive logging.

        Enhanced Scoring System:
        - Hybrid Filter: ≥1 keyword + (sector OR location) - EARLY REJECTION
        - Scope Text: 35 points (keyword/phrase matching)
        - Semantic: 15 points (AI similarity)
        - Sector: 20 points (category matching)
        - Services: 15 points (service type matching)
        - Location: 10 points (state matching)
        - Authority: 5 points (client matching)
        Total: 100 points

        Args:
            tender: TenderDB object to score

        Returns:
            Tuple of (score, metadata) where score is 0-100 and metadata contains match details
        """
        if not self.user_profile:
            self.build_user_profile()

        logger.debug(f"Scoring tender {tender.id}: '{tender.title[:50]}...'")
        logger.debug(f"  Tender category: {tender.category}, type: {tender.tender_type}")

        # HYBRID FILTER: Early rejection if doesn't meet minimum criteria
        passes_filter, filter_reason = self._passes_hybrid_filter(tender)
        if not passes_filter:
            logger.debug(f"  REJECTED by hybrid filter: {filter_reason}")
            metadata = {
                "match_reasons": [],
                "sector_match": 0,
                "scope_match": 0,
                "semantic_similarity": 0,
                "services_match": 0,
                "location_match": 0,
                "authority_match": 0,
                "hybrid_filter_passed": False,
                "hybrid_filter_reason": filter_reason
            }
            return 0.0, metadata

        # Initialize scoring
        score = 0.0
        metadata = {
            "match_reasons": [filter_reason],
            "sector_match": 0,
            "scope_match": 0,
            "semantic_similarity": 0,
            "services_match": 0,
            "location_match": 0,
            "authority_match": 0,
            "hybrid_filter_passed": True,
            "hybrid_filter_reason": filter_reason
        }

        # 1. Scope/Description Text Match (35 points)
        scope_score, has_description = self._score_scope_match(tender)
        score += scope_score
        metadata["scope_match"] = scope_score
        if scope_score > 0:
            metadata["match_reasons"].append(f"Scope match: {int((scope_score/self.WEIGHT_SCOPE_TEXT)*100)}%")
        logger.debug(f"  Scope score: {scope_score:.2f}/{self.WEIGHT_SCOPE_TEXT}")

        # 2. Semantic Similarity (15 points) - AI-powered
        semantic_score, similarity_pct = self._score_semantic_similarity(tender)
        score += semantic_score
        metadata["semantic_similarity"] = similarity_pct
        if semantic_score > 0:
            metadata["match_reasons"].append(f"AI semantic: {similarity_pct:.1f}%")
        logger.debug(f"  Semantic score: {semantic_score:.2f}/{self.WEIGHT_SEMANTIC}")

        # 3. Sector Match (20 points)
        sector_score = self._score_sector_match(tender)
        score += sector_score
        metadata["sector_match"] = sector_score
        if sector_score > 0:
            metadata["match_reasons"].append(f"Sector: {int((sector_score/self.WEIGHT_SECTOR)*100)}%")
        logger.debug(f"  Sector score: {sector_score:.2f}/{self.WEIGHT_SECTOR}")

        # 4. Services Match (15 points)
        services_score = self._score_services_match(tender)
        score += services_score
        metadata["services_match"] = services_score
        if services_score > 0:
            metadata["match_reasons"].append(f"Services: {int((services_score/self.WEIGHT_SERVICES)*100)}%")
        logger.debug(f"  Services score: {services_score:.2f}/{self.WEIGHT_SERVICES}")

        # 5. Location Match (10 points)
        location_score = self._score_location_match(tender)
        score += location_score
        metadata["location_match"] = location_score
        if location_score > 0:
            metadata["match_reasons"].append(f"Location: {int((location_score/self.WEIGHT_LOCATION)*100)}%")
        logger.debug(f"  Location score: {location_score:.2f}/{self.WEIGHT_LOCATION}")

        # 6. Authority/Client Match (5 points)
        authority_score = self._score_authority_match(tender)
        score += authority_score
        metadata["authority_match"] = authority_score
        if authority_score > 0:
            metadata["match_reasons"].append(f"Authority: {int((authority_score/self.WEIGHT_AUTHORITY)*100)}%")
        logger.debug(f"  Authority score: {authority_score:.2f}/{self.WEIGHT_AUTHORITY}")

        logger.debug(f"  FINAL SCORE: {score:.2f}/100 - Match reasons: {metadata['match_reasons']}")

        return round(score, 2), metadata

    def _score_sector_match(self, tender: TenderDB) -> float:
        """
        Score sector/category matching with synonym expansion (0-20 points).
        Enhanced with domain-specific synonyms for better matching.
        """
        if not self.user_profile["sectors"]:
            return 0.0

        # Get tender sectors
        tender_sectors = set()
        if tender.category:
            tender_sectors.add(tender.category.lower().strip())
        if tender.tender_type:
            tender_sectors.add(tender.tender_type.lower().strip())
        if tender.tender_category:
            tender_sectors.add(tender.tender_category.lower().strip())

        if not tender_sectors:
            return 0.0

        user_sectors_lower = {s.lower() for s in self.user_profile["sectors"]}

        # 1. Check for exact matches first
        exact_matches = tender_sectors & user_sectors_lower
        if exact_matches:
            logger.debug(f"Exact sector match found: {exact_matches}")
            return self.WEIGHT_SECTOR

        # 2. Expand both user and tender sectors with synonyms
        expanded_user_sectors = self._expand_with_synonyms(user_sectors_lower)
        expanded_tender_sectors = self._expand_with_synonyms(tender_sectors)

        # Check for synonym matches
        synonym_matches = expanded_user_sectors & expanded_tender_sectors
        if synonym_matches:
            logger.debug(f"Synonym sector match found: {synonym_matches}")
            return self.WEIGHT_SECTOR * 0.95  # 95% for synonym match

        # 3. Check for fuzzy matches using multi-tier scoring
        max_fuzzy_score = 0
        best_match_pair = None
        for user_sector in self.user_profile["sectors"]:
            for tender_sector in tender_sectors:
                fuzzy_match_score = self._fuzzy_match_score(user_sector, tender_sector)
                if fuzzy_match_score > max_fuzzy_score:
                    max_fuzzy_score = fuzzy_match_score
                    best_match_pair = (user_sector, tender_sector)

        if max_fuzzy_score > 0:
            logger.debug(f"Fuzzy sector match: {best_match_pair} -> {max_fuzzy_score*100}%")
            return self.WEIGHT_SECTOR * max_fuzzy_score

        # 4. Check for partial word matches (e.g., "petrol" matches "petroleum")
        for user_sector in user_sectors_lower:
            if self._partial_word_match(user_sector, tender_sectors):
                logger.debug(f"Partial sector match: {user_sector}")
                return self.WEIGHT_SECTOR * 0.8  # 80% for partial match

        return 0.0

    def _score_scope_match(self, tender: TenderDB) -> Tuple[float, bool]:
        """
        Score scope/description text matching with synonym expansion (0-40 points).
        Enhanced to match keywords with their domain synonyms.

        Returns:
            Tuple of (score, has_description_bool)
        """
        if not self.user_profile["keywords"]:
            return 0.0, False

        # Collect tender text
        tender_text_parts = []
        if tender.title:
            tender_text_parts.append(tender.title)
        if tender.summary:
            tender_text_parts.append(tender.summary)
        if tender.work_item_details:
            if isinstance(tender.work_item_details, str):
                tender_text_parts.append(tender.work_item_details)
            elif isinstance(tender.work_item_details, dict):
                tender_text_parts.append(str(tender.work_item_details))

        if not tender_text_parts:
            return 0.0, False

        tender_text = " ".join(tender_text_parts)
        has_description = len(tender_text) > 100  # Has meaningful description if >100 chars

        # Extract keywords and phrases from tender
        tender_keywords, tender_phrases = self._extract_keywords_and_phrases(tender_text)

        if not tender_keywords:
            return 0.0, has_description

        # 1. Calculate exact keyword overlap
        exact_keyword_overlap = len(self.user_profile["keywords"] & tender_keywords)

        # 2. Expand keywords with synonyms for semantic matching
        expanded_user_keywords = self._expand_with_synonyms(self.user_profile["keywords"])
        expanded_tender_keywords = self._expand_with_synonyms(tender_keywords)

        # Calculate synonym-based overlap
        synonym_overlap = len(expanded_user_keywords & expanded_tender_keywords)

        # 3. Check for partial word matches (e.g., "petrol" matches "petroleum")
        partial_matches = 0
        for user_keyword in self.user_profile["keywords"]:
            if user_keyword not in tender_keywords:  # Only check non-exact matches
                if self._partial_word_match(user_keyword, tender_keywords):
                    partial_matches += 1

        # Calculate total keyword matches with weights
        # Exact matches: 100%, Synonym matches: 80%, Partial matches: 60%
        total_matches = exact_keyword_overlap + (synonym_overlap - exact_keyword_overlap) * 0.8 + partial_matches * 0.6
        keyword_overlap_pct = min(total_matches / len(self.user_profile["keywords"]), 1.0)

        logger.debug(f"Scope matching: exact={exact_keyword_overlap}, synonym={synonym_overlap}, partial={partial_matches}")

        # 4. Calculate phrase overlap (bonus)
        phrase_overlap = len(self.user_profile["phrases"] & tender_phrases)
        phrase_bonus = 0.0
        if phrase_overlap > 0 and self.user_profile["phrases"]:
            phrase_overlap_pct = phrase_overlap / len(self.user_profile["phrases"])
            phrase_bonus = phrase_overlap_pct * self.PHRASE_BONUS_MULTIPLIER

        # Total scope score
        base_score = keyword_overlap_pct * self.WEIGHT_SCOPE_TEXT
        bonus_score = min(phrase_bonus * 10, 10)  # Cap bonus at 10 points

        return min(base_score + bonus_score, self.WEIGHT_SCOPE_TEXT), has_description

    def _score_services_match(self, tender: TenderDB) -> float:
        """Score services matching (0-20 points)."""
        if not self.user_profile["services"]:
            return 0.0

        # Extract service-related keywords from tender
        service_text_parts = []
        if tender.work_item_details:
            if isinstance(tender.work_item_details, str):
                service_text_parts.append(tender.work_item_details)
            elif isinstance(tender.work_item_details, dict):
                service_text_parts.append(str(tender.work_item_details))
        if tender.summary:
            service_text_parts.append(tender.summary)

        if not service_text_parts:
            return 0.0

        service_text = " ".join(service_text_parts).lower()

        # Check how many user services appear in tender text
        matched_services = 0
        for user_service in self.user_profile["services"]:
            user_service_lower = user_service.lower()
            # Check exact substring match or fuzzy match
            if user_service_lower in service_text:
                matched_services += 1
            else:
                # Try fuzzy matching with tender keywords
                for word in service_text.split():
                    if len(word) >= self.MIN_WORD_LENGTH:
                        if fuzz.ratio(user_service_lower, word) >= self.FUZZY_THRESHOLD:
                            matched_services += 1
                            break

        if matched_services == 0:
            return 0.0

        match_ratio = matched_services / len(self.user_profile["services"])
        return match_ratio * self.WEIGHT_SERVICES

    def _score_location_match(self, tender: TenderDB) -> float:
        """Score location matching (0-10 points)."""
        if not self.user_profile["locations"] or not tender.state:
            return 0.0

        tender_state = tender.state.lower().strip()
        user_locations_lower = {loc.lower() for loc in self.user_profile["locations"]}

        # Check for exact match
        if tender_state in user_locations_lower:
            return self.WEIGHT_LOCATION

        # Check for fuzzy match
        max_fuzzy_score = 0
        for user_location in self.user_profile["locations"]:
            fuzzy_score = fuzz.ratio(user_location.lower(), tender_state)
            if fuzzy_score > max_fuzzy_score:
                max_fuzzy_score = fuzzy_score

        if max_fuzzy_score >= self.FUZZY_THRESHOLD:
            return self.WEIGHT_LOCATION * 0.5  # 50% for fuzzy match

        return 0.0

    def _score_authority_match(self, tender: TenderDB) -> float:
        """Score authority/client matching (0-5 points)."""
        if not self.user_profile["clients"] or not tender.authority:
            return 0.0

        tender_authority = tender.authority.lower().strip()

        # Check against user's past clients
        for client in self.user_profile["clients"]:
            client_lower = client.lower()
            # Check if client name appears in authority or vice versa
            if client_lower in tender_authority or tender_authority in client_lower:
                return self.WEIGHT_AUTHORITY
            # Fuzzy match
            if fuzz.ratio(client_lower, tender_authority) >= self.FUZZY_THRESHOLD:
                return self.WEIGHT_AUTHORITY * 0.5

        return 0.0

    def _score_title_match(self, tender: TenderDB) -> float:
        """
        Score title-only matching for tenders with minimal description.
        Returns 0-1 score.
        """
        if not tender.title or not self.user_profile["keywords"]:
            return 0.0

        title_keywords, _ = self._extract_keywords_and_phrases(tender.title)

        if not title_keywords:
            return 0.0

        keyword_overlap = len(self.user_profile["keywords"] & title_keywords)

        if keyword_overlap == 0:
            return 0.0

        return keyword_overlap / len(self.user_profile["keywords"])

    def categorize_results(self, scored_tenders: List[Tuple[TenderDB, float, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize scored tenders into 'more_reco' (>65%) and 'less_reco' (30-65%).

        Very loose thresholds for maximum coverage per user request:
        - >65%: Highly Recommended (strong match) - lowered from 70%
        - 30-65%: Other Recommendations (potential match) - lowered from 45%
        - <30%: Not shown (filtered out)

        Args:
            scored_tenders: List of (tender, score, metadata) tuples

        Returns:
            Dict with 'more_reco' and 'less_reco' arrays containing tender data
        """
        more_reco = []
        less_reco = []

        for tender, score, metadata in scored_tenders:
            if score > 65:  # Lowered from 70 to 65 for more results
                more_reco.append({
                    "tender": tender,
                    "score": score,
                    "match_reasons": metadata["match_reasons"],
                    "semantic_similarity": metadata.get("semantic_similarity", 0)
                })
            elif score >= 30:  # Lowered from 45 to 30 for maximum coverage
                less_reco.append({
                    "tender": tender,
                    "score": score,
                    "match_reasons": metadata["match_reasons"],
                    "semantic_similarity": metadata.get("semantic_similarity", 0)
                })

        # Sort by score (descending)
        more_reco.sort(key=lambda x: x["score"], reverse=True)
        less_reco.sort(key=lambda x: x["score"], reverse=True)

        return {
            "more_reco": more_reco,
            "less_reco": less_reco,
            "total_more": len(more_reco),
            "total_less": len(less_reco)
        }
