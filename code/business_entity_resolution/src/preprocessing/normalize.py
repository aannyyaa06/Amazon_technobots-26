import re
import unicodedata

# Common legal suffix patterns to standardize
LEGAL_SUFFIXES = [
    (r'\b(private\s+limited|pvt\s+ltd|pvt\.\s*ltd\.?|private\s+ltd\.?)\b', 'pvt ltd'),
    (r'\b(limited|ltd\.?)\b', 'ltd'),
    (r'\b(incorporated|inc\.?)\b', 'inc'),
    (r'\b(corporation|corp\.?)\b', 'corp'),
    (r'\b(limited\s+liability\s+partnership|llp\.?)\b', 'llp'),
    (r'\b(company|co\.?)\b', 'co'),
    (r'\b(llc\.?)\b', 'llc'),
    (r'\b(gmbh\.?)\b', 'gmbh'),
    (r'\b(s\.?a\.?r\.?l\.?|sarl)\b', 'sarl'),
    (r'\b(s\.?a\.?s\.?|sas)\b', 'sas'),
]

# Common address abbreviations
ADDRESS_ABBR = [
    (r'\b(road|rd\.?)\b', 'rd'),
    (r'\b(street|st\.?)\b', 'st'),
    (r'\b(avenue|ave\.?)\b', 'ave'),
    (r'\b(boulevard|blvd\.?)\b', 'blvd'),
    (r'\b(drive|dr\.?)\b', 'dr'),
    (r'\b(lane|ln\.?)\b', 'ln'),
    (r'\b(highway|hwy\.?)\b', 'hwy'),
    (r'\b(floor|fl\.?)\b', 'fl'),
    (r'\b(suite|ste\.?)\b', 'ste'),
    (r'\b(apartment|apt\.?)\b', 'apt'),
    (r'\b(building|bldg\.?)\b', 'bldg'),
    (r'\b(nagar|ngr\.?)\b', 'nagar'),
    (r'\b(cross|x\.?)\b', 'cross'),
    (r'\b(sector|sec\.?)\b', 'sec'),
]


def unicode_clean(text: str) -> str:
    """Normalize unicode characters (NFKD) and convert to ASCII-compatible lowercase."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text.lower()


def normalize_name(name: str) -> str:
    """
    Conservative business name normalization:
    - Lowercase & Unicode clean
    - Replace '&' with 'and'
    - Standardize common legal suffixes
    - Strip punctuation and collapse whitespace
    """
    if not name or not isinstance(name, str):
        return ""
    
    text = unicode_clean(name)
    text = text.replace("&", " and ")
    
    for pattern, repl in LEGAL_SUFFIXES:
        text = re.sub(pattern, repl, text)
        
    # Replace non-alphanumeric with spaces, preserving digits
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(address: str) -> str:
    """
    Conservative address normalization:
    - Lowercase & Unicode clean
    - Preserve numbers (house/unit/shop numbers are critical)
    - Standardize common street/road abbreviations
    - Strip punctuation and collapse whitespace
    """
    if not address or not isinstance(address, str):
        return ""
    
    text = unicode_clean(address)
    text = text.replace("&", " and ")
    
    for pattern, repl in ADDRESS_ABBR:
        text = re.sub(pattern, repl, text)
        
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_numbers(text: str) -> set:
    """Extract all numeric tokens from text."""
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))
