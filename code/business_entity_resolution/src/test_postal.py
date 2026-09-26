import re

def extract_postal_code(addr: str, country: str) -> str:
    if not addr: return ""
    if country == 'US':
        m = re.findall(r'\b\d{5}\b', addr)
        return m[-1] if m else ""
    elif country == 'India':
        m = re.findall(r'\b[1-9]\d{5}\b', addr)
        return m[-1] if m else ""
    elif country == 'France':
        m = re.findall(r'\b\d{5}\b', addr)
        return m[0] if m else ""
    return ""

samples = [
    ('1004 7th st minneapolis mn 55415', 'US'),
    ('301 campus 31 marathahalli orr bangalore 560103 karnataka', 'India'),
    ('75008 paris 12 rue de la paix', 'France')
]

for a, c in samples:
    print(f"{c}: '{a}' -> Postal Code: '{extract_postal_code(a, c)}'")
