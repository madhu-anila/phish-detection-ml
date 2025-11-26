"""
Brand Detection from Email Text
Extracts mentioned brands using regex patterns
"""

import re

# Comprehensive brand patterns for phishing detection
BRAND_PATTERNS = {
    # Financial Services
    'paypal': [r'\bpaypal\b', r'\bpay\s*pal\b'],
    'amazon': [r'\bamazon\b', r'\bamzn\b'],
    'ebay': [r'\bebay\b', r'\be\s*bay\b'],
    'visa': [r'\bvisa\b'],
    'mastercard': [r'\bmastercard\b', r'\bmaster\s*card\b'],
    'american_express': [r'\bamerican\s*express\b', r'\bamex\b'],
    'discover': [r'\bdiscover\b'],
    
    # Banks
    'chase': [r'\bchase\b', r'\bjp\s*morgan\b'],
    'wells_fargo': [r'\bwells\s*fargo\b'],
    'bank_of_america': [r'\bbank\s*of\s*america\b', r'\bbofa\b'],
    'citibank': [r'\bcitibank\b', r'\bciti\b'],
    'us_bank': [r'\bus\s*bank\b'],
    'capital_one': [r'\bcapital\s*one\b'],
    'hsbc': [r'\bhsbc\b'],
    'barclays': [r'\bbarclays\b'],
    
    # Tech Companies
    'apple': [r'\bapple\b', r'\bicloud\b', r'\bitunes\b'],
    'microsoft': [r'\bmicrosoft\b', r'\boffice\s*365\b', r'\boutlook\b'],
    'google': [r'\bgoogle\b', r'\bgmail\b'],
    'facebook': [r'\bfacebook\b', r'\bmeta\b'],
    'twitter': [r'\btwitter\b', r'\bx\. com\b'],
    'linkedin': [r'\blinkedin\b'],
    'instagram': [r'\binstagram\b'],
    'netflix': [r'\bnetflix\b'],
    'spotify': [r'\bspotify\b'],
    'adobe': [r'\badobe\b'],
    
    # Shipping/Logistics
    'ups': [r'\bups\b'],
    'fedex': [r'\bfedex\b', r'\bfed\s*ex\b'],
    'dhl': [r'\bdhl\b'],
    'usps': [r'\busps\b', r'\bus\s*postal\b'],
    
    # E-commerce/Retail
    'walmart': [r'\bwalmart\b'],
    'target': [r'\btarget\b'],
    'bestbuy': [r'\bbest\s*buy\b'],
    
    # Telecom
    'att': [r'\bat&t\b', r'\batt\b'],
    'verizon': [r'\bverizon\b'],
    'tmobile': [r'\bt\s*mobile\b'],
    'sprint': [r'\bsprint\b'],
    
    # Others
    'irs': [r'\birs\b', r'\binternal\s*revenue\b'],
    'ssa': [r'\bsocial\s*security\b', r'\bssa\b'],
    'bank': [r'\bbank\b', r'\bbanking\b'],  # generic
}


def extract_brands_from_text(text: str) -> list:
    """
    Extract brand mentions from email text
    
    Args:
        text: Email subject + body combined
    
    Returns:
        List of detected brand keys (e.g., ['paypal', 'ups'])
    """
    if not text:
        return []
    
    text = text. lower()
    found_brands = []
    
    for brand_key, patterns in BRAND_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re. IGNORECASE):
                found_brands.append(brand_key)
                break  # Only count each brand once
    
    # Return unique brands
    return list(set(found_brands))


def has_brand_mention(text: str) -> bool:
    """Check if text mentions any known brand"""
    return len(extract_brands_from_text(text)) > 0


# Test function
if __name__ == "__main__":
    test_cases = [
        "Dear PayPal user, verify your account",
        "Your Amazon order has shipped via UPS",
        "Microsoft Office 365 subscription expired",
        "Hello, this is a generic email",
        "Bank of America account suspended - Chase bank alert"
    ]
    
    print("="*60)
    print("Brand Extraction Test")
    print("="*60)
    for text in test_cases:
        brands = extract_brands_from_text(text)
        print(f"\nText: {text}")
        print(f"Brands: {brands}")
    print("="*60)
