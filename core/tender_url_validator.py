"""
Tender URL Validator

Validates that tender URLs belong to allowed government e-procurement portals.
"""

# List of allowed URL prefixes for tender links
ALLOWED_TENDER_URL_PREFIXES = [
    "https://eprocure.gov.in/eprocure/app",
    "https://defproc.gov.in/nicgep/app",
    "https://pmgsytenders.gov.in/nicgep/app",
    "https://etenders.gov.in/eprocure/app",
    "https://coalindiatenders.nic.in/nicgep/app",
    "https://iocletenders.nic.in/nicgep/app",
    "https://cpcletenders.nic.in/nicgep/app",
    "https://eprocurebel.co.in/nicgep/app",
    "https://eprocurentpc.nic.in/nicgep/app",
    "https://eprocuregsl.nic.in/nicgep/app",
    "https://eprocurehsl.nic.in/nicgep/app",
    "https://eprocuremdl.nic.in/nicgep/app",
    "https://www.eprocuremidhani.nic.in/nicgep/app",
    "https://eprocuregrse.co.in/nicgep/app",
    "https://eprocurebhel.co.in/nicgep/app",
    "https://arunachaltenders.gov.in/nicgep/app",
    "https://eprocure.andamannicobar.gov.in/nicgep/app",
    "https://assamtenders.gov.in/nicgep/app",
    "https://etenders.chd.nic.in/nicgep/app",
    "https://dnhtenders.gov.in/nicgep/app",
    "https://ddtenders.gov.in/nicgep/app",
    "https://govtprocurement.delhi.gov.in/nicgep/app",
    "https://eprocure.goa.gov.in/nicgep/app",
    "https://etenders.hry.nic.in/nicgep/app",
    "https://hptenders.gov.in/nicgep/app",
    "https://jktenders.gov.in/nicgep/app",
    "https://jharkhandtenders.gov.in/nicgep/app",
    "https://etenders.kerala.gov.in/nicgep/app",
    "https://tenders.ladakh.gov.in/nicgep/app",
    "https://tendersutl.gov.in/nicgep/app",
    "https://mahatenders.gov.in/nicgep/app",
    "https://mptenders.gov.in/nicgep/app",
    "https://manipurtenders.gov.in/nicgep/app",
    "https://meghalayatenders.gov.in/nicgep/app",
    "https://mizoramtenders.gov.in/nicgep/app",
    "https://nagalandtenders.gov.in/nicgep/app",
    "https://tendersodisha.gov.in/nicgep/app",
    "https://pudutenders.gov.in/nicgep/app",
    "https://eproc.punjab.gov.in/nicgep/app",
    "https://eproc.rajasthan.gov.in/nicgep/app",
    "https://sikkimtender.gov.in/nicgep/app",
    "https://tntenders.gov.in/nicgep/app",
    "https://tripuratenders.gov.in/nicgep/app",
    "https://uktenders.gov.in/nicgep/app",
    "https://etender.up.nic.in/nicgep/app",
    "https://wbtenders.gov.in/nicgep/app",
    "https://eprocure.gov.in/epublish/app",
]


def is_valid_tender_url(url: str) -> bool:
    """
    Validate if a URL starts with any of the allowed tender URL prefixes.
    
    Args:
        url: The URL string to validate
        
    Returns:
        True if the URL starts with any allowed prefix, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Normalize URL: strip whitespace and convert to lowercase for comparison
    url = url.strip()
    
    # Check if URL starts with any allowed prefix
    for allowed_prefix in ALLOWED_TENDER_URL_PREFIXES:
        if url.startswith(allowed_prefix):
            return True
    
    return False


def validate_tender_url(url: str) -> tuple[bool, str]:
    """
    Validate a tender URL and return validation result with message.
    
    Args:
        url: The URL string to validate
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required and must be a string"
    
    url = url.strip()
    
    if not url:
        return False, "URL cannot be empty"
    
    if is_valid_tender_url(url):
        return True, "URL is valid"
    else:
        return False, "URL must start with one of the allowed government e-procurement portal URLs. Please enter a valid tender link."
