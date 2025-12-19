"""
SEO Configuration Module for BidSuite

This module contains all SEO-related configuration including site metadata,
social media handles, default images, and organization details.
"""

import os
from typing import Dict, Optional
from urllib.parse import urljoin

# Get site URL from environment or use default
SITE_URL = os.getenv("SITE_URL", "https://bidsuite.com").rstrip("/")

# Site Metadata
SITE_NAME = "BidSuite"
SITE_DESCRIPTION = "Operational command centre for government tender strategy. BidSuite unifies discovery, evaluation, collaboration, and compliance into a single experience built for leadership teams navigating high-value public infrastructure opportunities."
SITE_TAGLINE = "Government Tender Intelligence for modern enterprises"

# Default Images
DEFAULT_OG_IMAGE = urljoin(SITE_URL, "/static/images/og-default.jpg")
DEFAULT_FAVICON = urljoin(SITE_URL, "/static/images/favicon.ico")
DEFAULT_LOGO = urljoin(SITE_URL, "/static/images/logo.png")

# Social Media Handles
SOCIAL_MEDIA = {
    "twitter": "@BidSuite",  # Update with actual handle
    "facebook": "BidSuite",  # Update with actual handle
    "linkedin": "company/bidsuite",  # Update with actual handle
    "instagram": None,  # Add if available
}

# Organization Details
ORGANIZATION = {
    "name": "BidSuite",
    "legalName": "BidSuite",  # Update with actual legal name
    "url": SITE_URL,
    "logo": DEFAULT_LOGO,
    "description": SITE_DESCRIPTION,
    "foundingDate": "2024",  # Update with actual founding date
    "contactPoint": {
        "contactType": "Customer Service",
        "email": "support@bidsuite.com",  # Update with actual email
        "telephone": None,  # Add if available
    },
    "address": {
        "addressCountry": "IN",  # Update with actual country code
        "addressLocality": None,  # Add if available
        "addressRegion": None,  # Add if available
        "postalCode": None,  # Add if available
        "streetAddress": None,  # Add if available
    },
    "sameAs": [
        f"https://twitter.com/{SOCIAL_MEDIA['twitter'].lstrip('@')}" if SOCIAL_MEDIA.get("twitter") else None,
        f"https://www.facebook.com/{SOCIAL_MEDIA['facebook']}" if SOCIAL_MEDIA.get("facebook") else None,
        f"https://www.linkedin.com/{SOCIAL_MEDIA['linkedin']}" if SOCIAL_MEDIA.get("linkedin") else None,
    ],
}

# Remove None values from sameAs
ORGANIZATION["sameAs"] = [url for url in ORGANIZATION["sameAs"] if url]

# Default Meta Tags
DEFAULT_META = {
    "title": f"{SITE_NAME} - {SITE_TAGLINE}",
    "description": SITE_DESCRIPTION,
    "keywords": "government tenders, public procurement, tender management, bidding platform, government contracts, procurement software, tender intelligence, BidSuite",
    "author": SITE_NAME,
    "robots": "index, follow",
    "language": "en",
    "locale": "en_US",
}

# Open Graph Defaults
OG_DEFAULTS = {
    "type": "website",
    "site_name": SITE_NAME,
    "locale": "en_US",
    "image": DEFAULT_OG_IMAGE,
    "image_width": 1200,
    "image_height": 630,
    "image_alt": f"{SITE_NAME} - {SITE_TAGLINE}",
}

# Twitter Card Defaults
TWITTER_DEFAULTS = {
    "card": "summary_large_image",
    "site": SOCIAL_MEDIA.get("twitter", ""),
    "creator": SOCIAL_MEDIA.get("twitter", ""),
    "image": DEFAULT_OG_IMAGE,
    "image_alt": f"{SITE_NAME} - {SITE_TAGLINE}",
}

# Sitemap Configuration
SITEMAP_CONFIG = {
    "changefreq": {
        "home": "daily",
        "tenders": "hourly",
        "projects": "daily",
        "static": "weekly",
    },
    "priority": {
        "home": 1.0,
        "tenders": 0.9,
        "tender_detail": 0.8,
        "projects": 0.7,
        "project_detail": 0.7,
        "static": 0.5,
    },
}

# Pages that should not be indexed
NOINDEX_PAGES = [
    "/login",
    "/employee/login",
    "/expert/login",
    "/dashboard",
    "/employee/dashboard",
    "/expert/dashboard",
    "/profile",
    "/expert/profile",
    "/api/",
]

def get_site_url() -> str:
    """Get the base site URL."""
    return SITE_URL

def get_organization_schema() -> Dict:
    """Get the organization schema for structured data."""
    return ORGANIZATION.copy()

def get_default_meta() -> Dict:
    """Get default meta tags."""
    return DEFAULT_META.copy()

def get_og_defaults() -> Dict:
    """Get default Open Graph tags."""
    return OG_DEFAULTS.copy()

def get_twitter_defaults() -> Dict:
    """Get default Twitter Card tags."""
    return TWITTER_DEFAULTS.copy()

def should_index_page(path: str) -> bool:
    """Check if a page should be indexed."""
    return not any(path.startswith(noindex_path) for noindex_path in NOINDEX_PAGES)

def get_canonical_url(path: str, query_params: Optional[Dict] = None) -> str:
    """Generate canonical URL for a given path."""
    # Remove tracking parameters
    excluded_params = ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "fbclid", "gclid"]
    
    if query_params:
        # Filter out excluded params
        clean_params = {k: v for k, v in query_params.items() if k not in excluded_params}
        if clean_params:
            query_string = "&".join(f"{k}={v}" for k, v in clean_params.items())
            return f"{SITE_URL}{path}?{query_string}"
    
    return f"{SITE_URL}{path}"

