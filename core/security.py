"""
Security utilities for authentication and password management.
"""

import bcrypt
import secrets
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from core.redis_client import get_redis_client, is_redis_available

# Fallback in-memory session storage (only used if Redis is unavailable)
user_sessions = {}

# Session expiration
SESSION_EXPIRE_DAYS = int(os.getenv('SESSION_EXPIRE_DAYS', '7'))


def hash_password(password: str) -> str:
    """Hash password using bcrypt with automatic salt generation."""
    # bcrypt automatically handles salting
    salt = bcrypt.gensalt(rounds=12)  # Higher rounds = slower but more secure
    pwd_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
    return pwd_hash.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify password against hash.
    Supports both old SHA-256 format and new bcrypt format for backward compatibility.
    """
    try:
        # Try bcrypt first (new format)
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed.encode('utf-8')
        )
    except Exception:
        # Fall back to old SHA-256 format (salt:hash)
        try:
            if ':' in hashed:
                salt, pwd_hash = hashed.split(':')
                computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
                return computed_hash == pwd_hash
        except Exception:
            pass
        return False


def create_session(user_id: str) -> str:
    """Create user session and return session token."""
    session_token = secrets.token_urlsafe(32)
    session_data = {
        'user_id': user_id,
        'created_at': datetime.utcnow().isoformat(),
        'expires_at': (datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS)).isoformat(),
        'pin_verified': False  # Feature lock: tracks if PIN has been verified this session
    }

    # Try to store in Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            # Store in Redis with automatic expiration
            redis_client.setex(
                f'session:{session_token}',
                timedelta(days=SESSION_EXPIRE_DAYS),
                json.dumps(session_data)
            )
            return session_token
        except Exception as e:
            print(f"⚠ Redis session storage failed: {e}, falling back to in-memory")

    # Fallback to in-memory storage
    user_sessions[session_token] = {
        'user_id': user_id,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS),
        'pin_verified': False
    }
    return session_token


def create_test_session(user_id: str, restricted: bool = True, test_token: str = None, test_base_url: str = None) -> str:
    """
    Create a test session for isolated testing (e.g., public project viewing).

    Args:
        user_id: User ID to create session for
        restricted: If True, marks session as navigation-restricted
        test_token: The test token used to access (for redirect back)
        test_base_url: Base URL to redirect back to (e.g., /public-projects/nkbpl.pratyaksh)

    Returns:
        session_token: Session token for the test user
    """
    session_token = secrets.token_urlsafe(32)
    session_data = {
        'user_id': user_id,
        'created_at': datetime.utcnow().isoformat(),
        'expires_at': (datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS)).isoformat(),
        'pin_verified': False,
        'is_test_session': True,  # Mark as test session
        'restricted_navigation': restricted,  # Block navigation
        'test_token': test_token,  # Store for redirects
        'test_base_url': test_base_url,  # Store base URL for redirects
        'quarantined': restricted  # Explicit quarantine flag
    }

    # Store in Redis (or fallback to in-memory)
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            redis_client.setex(
                f'session:{session_token}',
                timedelta(days=SESSION_EXPIRE_DAYS),
                json.dumps(session_data)
            )
            return session_token
        except Exception as e:
            print(f"⚠ Redis test session storage failed: {e}, falling back to in-memory")

    # Fallback to in-memory storage
    user_sessions[session_token] = {
        'user_id': user_id,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS),
        'pin_verified': False,
        'is_test_session': True,
        'restricted_navigation': restricted,
        'test_token': test_token,
        'test_base_url': test_base_url,
        'quarantined': restricted
    }
    return session_token


def get_session(session_token: str) -> Optional[dict]:
    """Get session data from token."""
    if not session_token:
        return None

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            session_data = redis_client.get(f'session:{session_token}')
            if session_data:
                return json.loads(session_data)
            return None
        except Exception as e:
            print(f"⚠ Redis session retrieval failed: {e}, checking in-memory")

    # Fallback to in-memory storage
    if session_token not in user_sessions:
        return None

    session_data = user_sessions.get(session_token)
    if session_data and session_data['expires_at'] < datetime.utcnow():
        del user_sessions[session_token]
        return None

    return session_data


def update_session(session_token: str, updates: dict) -> bool:
    """
    Update session data with new fields.
    Used for feature lock PIN verification tracking.
    Returns True if successful, False otherwise.
    """
    if not session_token:
        return False

    # Get current session data
    session_data = get_session(session_token)
    if not session_data:
        return False

    # Merge updates into session data
    session_data.update(updates)

    # Try to store in Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            redis_client.setex(
                f'session:{session_token}',
                timedelta(days=SESSION_EXPIRE_DAYS),
                json.dumps(session_data)
            )
            return True
        except Exception as e:
            print(f"⚠ Redis session update failed: {e}, updating in-memory")

    # Fallback to in-memory storage
    if session_token in user_sessions:
        user_sessions[session_token].update(updates)
        return True

    return False


def delete_session(session_token: str) -> None:
    """Delete a session."""
    if not session_token:
        return

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            redis_client.delete(f'session:{session_token}')
            return
        except Exception as e:
            print(f"⚠ Redis session deletion failed: {e}, checking in-memory")

    # Fallback to in-memory storage
    if session_token in user_sessions:
        del user_sessions[session_token]


# ==================== Expert-Verse Authentication ====================

# Fallback in-memory expert session storage
expert_sessions = {}


def create_expert_session(expert_id: str) -> str:
    """Create expert session and return session token."""
    session_token = secrets.token_urlsafe(32)
    session_data = {
        'expert_id': expert_id,
        'user_type': 'expert',  # Identify this as expert session
        'created_at': datetime.utcnow().isoformat(),
        'expires_at': (datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS)).isoformat()
    }

    # Try to store in Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            # Store in Redis with automatic expiration (use expert: prefix)
            redis_client.setex(
                f'expert_session:{session_token}',
                timedelta(days=SESSION_EXPIRE_DAYS),
                json.dumps(session_data)
            )
            return session_token
        except Exception as e:
            print(f"⚠ Redis expert session storage failed: {e}, falling back to in-memory")

    # Fallback to in-memory storage
    expert_sessions[session_token] = {
        'expert_id': expert_id,
        'user_type': 'expert',
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS)
    }
    return session_token


def get_expert_session(session_token: str) -> Optional[dict]:
    """Get expert session data from token."""
    if not session_token:
        return None

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            session_data = redis_client.get(f'expert_session:{session_token}')
            if session_data:
                return json.loads(session_data)
            return None
        except Exception as e:
            print(f"⚠ Redis expert session retrieval failed: {e}, checking in-memory")

    # Fallback to in-memory storage
    if session_token not in expert_sessions:
        return None

    session_data = expert_sessions.get(session_token)
    if session_data and session_data['expires_at'] < datetime.utcnow():
        del expert_sessions[session_token]
        return None

    return session_data


def delete_expert_session(session_token: str) -> None:
    """Delete an expert session."""
    if not session_token:
        return

    # Try Redis first
    redis_client = get_redis_client()
    if redis_client and is_redis_available():
        try:
            redis_client.delete(f'expert_session:{session_token}')
            return
        except Exception as e:
            print(f"⚠ Redis expert session deletion failed: {e}, checking in-memory")

    # Fallback to in-memory storage
    if session_token in expert_sessions:
        del expert_sessions[session_token]
