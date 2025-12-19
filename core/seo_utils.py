"""
SEO Utility Functions for BidSuite

This module provides utility functions for generating meta tags, structured data,
canonical URLs, and other SEO-related content.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

from core.seo_config import (
    SITE_URL,
    SITE_NAME,
    DEFAULT_META,
    OG_DEFAULTS,
    TWITTER_DEFAULTS,
    ORGANIZATION,
    get_canonical_url,
    should_index_page,
)


def generate_meta_tags(
    title: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[str] = None,
    image: Optional[str] = None,
    url: Optional[str] = None,
    noindex: bool = False,
    author: Optional[str] = None,
    published_time: Optional[str] = None,
    modified_time: Optional[str] = None,
    og_type: Optional[str] = None,
    twitter_card: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate comprehensive meta tags for a page.
    
    Args:
        title: Page title (defaults to site default)
        description: Page description (defaults to site default)
        keywords: Page keywords
        image: OG/Twitter image URL
        url: Page URL (for canonical and OG)
        noindex: Whether to add noindex tag
        author: Page author
        published_time: ISO 8601 published time
        modified_time: ISO 8601 modified time
        og_type: Open Graph type (defaults to 'website')
        twitter_card: Twitter card type (defaults to 'summary_large_image')
    
    Returns:
        Dictionary of meta tag name-value pairs
    """
    meta = {}
    
    # Basic meta tags
    meta["title"] = title or DEFAULT_META["title"]
    meta["description"] = description or DEFAULT_META["description"]
    if keywords:
        meta["keywords"] = keywords
    if author:
        meta["author"] = author
    else:
        meta["author"] = DEFAULT_META["author"]
    
    # Robots meta
    if noindex:
        meta["robots"] = "noindex, nofollow"
    else:
        meta["robots"] = DEFAULT_META["robots"]
    
    # Language and locale
    meta["language"] = DEFAULT_META["language"]
    meta["locale"] = DEFAULT_META["locale"]
    
    # Open Graph tags
    og_image = image or OG_DEFAULTS["image"]
    og_url = url or SITE_URL
    og_type = og_type or OG_DEFAULTS["type"]
    
    meta["og:title"] = meta["title"]
    meta["og:description"] = meta["description"]
    meta["og:image"] = og_image
    meta["og:url"] = og_url
    meta["og:type"] = og_type
    meta["og:site_name"] = OG_DEFAULTS["site_name"]
    meta["og:locale"] = OG_DEFAULTS["locale"]
    meta["og:image:width"] = str(OG_DEFAULTS["image_width"])
    meta["og:image:height"] = str(OG_DEFAULTS["image_height"])
    meta["og:image:alt"] = OG_DEFAULTS["image_alt"]
    
    if published_time:
        meta["article:published_time"] = published_time
    if modified_time:
        meta["article:modified_time"] = modified_time
    
    # Twitter Card tags
    twitter_card = twitter_card or TWITTER_DEFAULTS["card"]
    meta["twitter:card"] = twitter_card
    meta["twitter:title"] = meta["title"]
    meta["twitter:description"] = meta["description"]
    meta["twitter:image"] = og_image
    meta["twitter:image:alt"] = OG_DEFAULTS["image_alt"]
    
    if TWITTER_DEFAULTS.get("site"):
        meta["twitter:site"] = TWITTER_DEFAULTS["site"]
    if TWITTER_DEFAULTS.get("creator"):
        meta["twitter:creator"] = TWITTER_DEFAULTS["creator"]
    
    return meta


def generate_organization_schema() -> Dict[str, Any]:
    """Generate Organization schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": ORGANIZATION["name"],
        "url": ORGANIZATION["url"],
        "logo": ORGANIZATION["logo"],
        "description": ORGANIZATION["description"],
    }
    
    if ORGANIZATION.get("legalName"):
        schema["legalName"] = ORGANIZATION["legalName"]
    
    if ORGANIZATION.get("foundingDate"):
        schema["foundingDate"] = ORGANIZATION["foundingDate"]
    
    if ORGANIZATION.get("contactPoint"):
        contact = ORGANIZATION["contactPoint"]
        schema["contactPoint"] = {
            "@type": "ContactPoint",
            "contactType": contact.get("contactType", "Customer Service"),
        }
        if contact.get("email"):
            schema["contactPoint"]["email"] = contact["email"]
        if contact.get("telephone"):
            schema["contactPoint"]["telephone"] = contact["telephone"]
    
    if ORGANIZATION.get("address"):
        address = ORGANIZATION["address"]
        address_schema = {"@type": "PostalAddress"}
        if address.get("addressCountry"):
            address_schema["addressCountry"] = address["addressCountry"]
        if address.get("addressLocality"):
            address_schema["addressLocality"] = address["addressLocality"]
        if address.get("addressRegion"):
            address_schema["addressRegion"] = address["addressRegion"]
        if address.get("postalCode"):
            address_schema["postalCode"] = address["postalCode"]
        if address.get("streetAddress"):
            address_schema["streetAddress"] = address["streetAddress"]
        
        if len(address_schema) > 1:  # More than just @type
            schema["address"] = address_schema
    
    if ORGANIZATION.get("sameAs"):
        schema["sameAs"] = ORGANIZATION["sameAs"]
    
    return schema


def generate_breadcrumb_schema(items: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Generate BreadcrumbList schema.org JSON-LD.
    
    Args:
        items: List of breadcrumb items, each with 'name' and 'url' keys
    
    Returns:
        BreadcrumbList schema dictionary
    """
    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [],
    }
    
    for idx, item in enumerate(items, start=1):
        schema["itemListElement"].append({
            "@type": "ListItem",
            "position": idx,
            "name": item["name"],
            "item": urljoin(SITE_URL, item["url"]),
        })
    
    return schema


def generate_webpage_schema(
    name: str,
    description: str,
    url: str,
    breadcrumb: Optional[Dict] = None,
    main_entity: Optional[Dict] = None,
    date_published: Optional[str] = None,
    date_modified: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate WebPage schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": name,
        "description": description,
        "url": urljoin(SITE_URL, url),
    }
    
    if date_published:
        schema["datePublished"] = date_published
    if date_modified:
        schema["dateModified"] = date_modified
    
    if breadcrumb:
        schema["breadcrumb"] = breadcrumb
    
    if main_entity:
        schema["mainEntity"] = main_entity
    
    return schema


def generate_itemlist_schema(
    name: str,
    description: str,
    items: List[Dict[str, Any]],
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate ItemList schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "ItemList",
        "name": name,
        "description": description,
        "itemListElement": [],
    }
    
    if url:
        schema["url"] = urljoin(SITE_URL, url)
    
    for idx, item in enumerate(items, start=1):
        list_item = {
            "@type": "ListItem",
            "position": idx,
        }
        
        if isinstance(item, dict):
            if "name" in item:
                list_item["name"] = item["name"]
            if "url" in item:
                list_item["item"] = urljoin(SITE_URL, item["url"])
            if "description" in item:
                list_item["description"] = item["description"]
        else:
            list_item["name"] = str(item)
        
        schema["itemListElement"].append(list_item)
    
    return schema


def generate_event_schema(
    name: str,
    start_date: str,
    end_date: Optional[str] = None,
    location: Optional[Dict] = None,
    description: Optional[str] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate Event schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Event",
        "name": name,
        "startDate": start_date,
    }
    
    if end_date:
        schema["endDate"] = end_date
    if description:
        schema["description"] = description
    if url:
        schema["url"] = urljoin(SITE_URL, url)
    if location:
        schema["location"] = location
    
    return schema


def generate_project_schema(
    name: str,
    description: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    status: Optional[str] = None,
    value: Optional[float] = None,
    organization: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Generate Project schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Project",
        "name": name,
        "description": description,
    }
    
    if start_date:
        schema["startDate"] = start_date
    if end_date:
        schema["endDate"] = end_date
    if status:
        schema["projectStatus"] = status
    if value:
        schema["value"] = {
            "@type": "MonetaryAmount",
            "value": value,
            "currency": "INR",  # Update based on actual currency
        }
    if organization:
        schema["sponsor"] = organization
    
    return schema


def generate_service_schema(
    name: str,
    description: str,
    provider: Optional[Dict] = None,
    area_served: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate Service schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "Service",
        "name": name,
        "description": description,
    }
    
    if provider:
        schema["provider"] = provider
    if area_served:
        schema["areaServed"] = area_served
    
    return schema


def generate_software_application_schema(
    name: str,
    description: str,
    application_category: str = "BusinessApplication",
    operating_system: str = "Web",
    offers: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Generate SoftwareApplication schema.org JSON-LD."""
    schema = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": name,
        "description": description,
        "applicationCategory": application_category,
        "operatingSystem": operating_system,
    }
    
    if offers:
        schema["offers"] = offers
    
    return schema


def generate_faq_schema(faq_items: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Generate FAQPage schema.org JSON-LD.
    
    Args:
        faq_items: List of FAQ items, each with 'question' and 'answer' keys
    
    Returns:
        FAQPage schema dictionary
    """
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [],
    }
    
    for item in faq_items:
        schema["mainEntity"].append({
            "@type": "Question",
            "name": item["question"],
            "acceptedAnswer": {
                "@type": "Answer",
                "text": item["answer"],
            },
        })
    
    return schema


def generate_howto_schema(
    name: str,
    description: str,
    steps: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Generate HowTo schema.org JSON-LD.
    
    Args:
        name: Name of the how-to
        description: Description of the how-to
        steps: List of steps, each with 'name' and 'text' keys
    
    Returns:
        HowTo schema dictionary
    """
    schema = {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": name,
        "description": description,
        "step": [],
    }
    
    for idx, step in enumerate(steps, start=1):
        schema["step"].append({
            "@type": "HowToStep",
            "position": idx,
            "name": step.get("name", f"Step {idx}"),
            "text": step.get("text", ""),
        })
    
    return schema


def render_structured_data(schema: Dict[str, Any]) -> str:
    """Convert schema dictionary to JSON-LD script tag HTML."""
    return f'<script type="application/ld+json">\n{json.dumps(schema, indent=2)}\n</script>'


def clean_url_for_canonical(path: str, query_string: Optional[str] = None) -> str:
    """Clean URL and generate canonical URL."""
    # Parse query string
    query_params = {}
    if query_string:
        query_params = parse_qs(query_string, keep_blank_values=True)
        # Flatten single-item lists
        query_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
    
    return get_canonical_url(path, query_params if query_params else None)


def truncate_text(text: str, max_length: int = 155) -> str:
    """Truncate text to max_length, ensuring it doesn't break words."""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length].rsplit(" ", 1)[0]
    if len(truncated) < len(text):
        truncated += "..."
    
    return truncated


def format_date_for_schema(date_value: Any) -> str:
    """Format date/datetime for schema.org (ISO 8601 format)."""
    if isinstance(date_value, str):
        return date_value
    elif isinstance(date_value, datetime):
        return date_value.isoformat()
    elif hasattr(date_value, "isoformat"):
        return date_value.isoformat()
    else:
        return str(date_value)

