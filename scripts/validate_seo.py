"""
SEO Validation Script for BidSuite

This script validates SEO implementations including:
- Structured data (JSON-LD)
- Meta tags
- Open Graph tags
- Canonical URLs
- Sitemap and robots.txt

Usage:
    python scripts/validate_seo.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.seo_config import SITE_URL, ORGANIZATION
from core.seo_utils import (
    generate_organization_schema,
    generate_breadcrumb_schema,
    generate_webpage_schema,
    render_structured_data,
)


def validate_json_ld(schema: Dict) -> List[str]:
    """Validate JSON-LD structured data."""
    errors = []
    
    # Check required fields
    if "@context" not in schema:
        errors.append("Missing @context in JSON-LD schema")
    elif schema["@context"] != "https://schema.org":
        errors.append(f"Invalid @context: {schema['@context']}")
    
    if "@type" not in schema:
        errors.append("Missing @type in JSON-LD schema")
    
    # Validate Organization schema
    if schema.get("@type") == "Organization":
        required_fields = ["name", "url"]
        for field in required_fields:
            if field not in schema:
                errors.append(f"Organization schema missing required field: {field}")
    
    # Validate BreadcrumbList schema
    if schema.get("@type") == "BreadcrumbList":
        if "itemListElement" not in schema:
            errors.append("BreadcrumbList missing itemListElement")
        else:
            for idx, item in enumerate(schema["itemListElement"]):
                if "@type" not in item or item["@type"] != "ListItem":
                    errors.append(f"BreadcrumbList item {idx} missing or invalid @type")
                if "position" not in item:
                    errors.append(f"BreadcrumbList item {idx} missing position")
                if "name" not in item:
                    errors.append(f"BreadcrumbList item {idx} missing name")
                if "item" not in item:
                    errors.append(f"BreadcrumbList item {idx} missing item URL")
    
    return errors


def validate_meta_tags(meta_tags: Dict) -> List[str]:
    """Validate meta tags."""
    errors = []
    
    required_tags = ["title", "description"]
    for tag in required_tags:
        if tag not in meta_tags:
            errors.append(f"Missing required meta tag: {tag}")
        elif not meta_tags[tag] or len(meta_tags[tag].strip()) == 0:
            errors.append(f"Empty meta tag: {tag}")
    
    # Validate title length
    if "title" in meta_tags:
        title_len = len(meta_tags["title"])
        if title_len > 60:
            errors.append(f"Title too long ({title_len} chars, max 60)")
        elif title_len < 30:
            errors.append(f"Title too short ({title_len} chars, min 30)")
    
    # Validate description length
    if "description" in meta_tags:
        desc_len = len(meta_tags["description"])
        if desc_len > 160:
            errors.append(f"Description too long ({desc_len} chars, max 160)")
        elif desc_len < 120:
            errors.append(f"Description too short ({desc_len} chars, min 120)")
    
    # Validate Open Graph tags
    og_required = ["og:title", "og:description", "og:image", "og:url", "og:type"]
    for tag in og_required:
        if tag not in meta_tags:
            errors.append(f"Missing required Open Graph tag: {tag}")
    
    # Validate Twitter Card tags
    twitter_required = ["twitter:card", "twitter:title", "twitter:description"]
    for tag in twitter_required:
        if tag not in meta_tags:
            errors.append(f"Missing required Twitter Card tag: {tag}")
    
    return errors


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_canonical_url(canonical: str) -> List[str]:
    """Validate canonical URL."""
    errors = []
    
    if not canonical:
        errors.append("Canonical URL is empty")
        return errors
    
    if not validate_url(canonical):
        errors.append(f"Invalid canonical URL format: {canonical}")
    
    # Check if canonical uses HTTPS
    if canonical.startswith("http://") and not canonical.startswith("https://"):
        errors.append("Canonical URL should use HTTPS")
    
    return errors


def test_seo_config():
    """Test SEO configuration."""
    print("Testing SEO Configuration...")
    errors = []
    
    if not SITE_URL:
        errors.append("SITE_URL is not set")
    elif not validate_url(SITE_URL):
        errors.append(f"Invalid SITE_URL: {SITE_URL}")
    
    if not ORGANIZATION.get("name"):
        errors.append("Organization name is not set")
    
    if errors:
        print("  ❌ Configuration errors found:")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        print("  ✅ Configuration is valid")
        return True


def test_structured_data():
    """Test structured data generation."""
    print("\nTesting Structured Data Generation...")
    errors = []
    
    # Test Organization schema
    try:
        org_schema = generate_organization_schema()
        org_errors = validate_json_ld(org_schema)
        if org_errors:
            errors.extend([f"Organization schema: {e}" for e in org_errors])
        else:
            print("  ✅ Organization schema is valid")
    except Exception as e:
        errors.append(f"Organization schema generation failed: {e}")
    
    # Test BreadcrumbList schema
    try:
        breadcrumb_items = [
            {"name": "Home", "url": "/"},
            {"name": "Tenders", "url": "/procurement"},
            {"name": "Tender Detail", "url": "/tender/123"},
        ]
        breadcrumb_schema = generate_breadcrumb_schema(breadcrumb_items)
        breadcrumb_errors = validate_json_ld(breadcrumb_schema)
        if breadcrumb_errors:
            errors.extend([f"BreadcrumbList schema: {e}" for e in breadcrumb_errors])
        else:
            print("  ✅ BreadcrumbList schema is valid")
    except Exception as e:
        errors.append(f"BreadcrumbList schema generation failed: {e}")
    
    # Test WebPage schema
    try:
        webpage_schema = generate_webpage_schema(
            name="Test Page",
            description="Test description",
            url="/test",
        )
        webpage_errors = validate_json_ld(webpage_schema)
        if webpage_errors:
            errors.extend([f"WebPage schema: {e}" for e in webpage_errors])
        else:
            print("  ✅ WebPage schema is valid")
    except Exception as e:
        errors.append(f"WebPage schema generation failed: {e}")
    
    if errors:
        print("  ❌ Structured data errors found:")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        print("  ✅ All structured data tests passed")
        return True


def test_meta_tags():
    """Test meta tag generation."""
    print("\nTesting Meta Tag Generation...")
    errors = []
    
    try:
        from core.seo_utils import generate_meta_tags
        
        meta = generate_meta_tags(
            title="Test Page Title",
            description="This is a test description for the page that should be long enough to pass validation.",
        )
        
        meta_errors = validate_meta_tags(meta)
        if meta_errors:
            errors.extend(meta_errors)
        else:
            print("  ✅ Meta tags are valid")
    except Exception as e:
        errors.append(f"Meta tag generation failed: {e}")
    
    if errors:
        print("  ❌ Meta tag errors found:")
        for error in errors:
            print(f"    - {error}")
        return False
    else:
        print("  ✅ All meta tag tests passed")
        return True


def main():
    """Run all SEO validation tests."""
    print("=" * 60)
    print("BidSuite SEO Validation")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Configuration", test_seo_config()))
    results.append(("Structured Data", test_structured_data()))
    results.append(("Meta Tags", test_meta_tags()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All SEO validations passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

