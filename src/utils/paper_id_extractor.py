import re
from typing import Optional

class PaperIdExtractor:
    _PATTERNS = [
        r'paper\s*(?:id|ID|Id)?\s*[:#]?\s*(\d{4}\.\d{4})',
        r'id\s*[:#]?\s*(\d{4}\.\d{4})',
        r'(?<!\d)(\d{4}\.\d{4})(?!\d)',
        r'paper\s*[\s:#]?\s*(\d{4}\.\d{4})',
        r'paper\s*number\s*[:#]?\s*(\d{4}\.\d{4})',
        r'paper\s*#\s*(\d{4}\.\d{4})'
    ]

    @classmethod
    def extract(cls, text: str) -> Optional[str]:
        """Extract paper ID from text using various patterns."""
        for pattern in cls._PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None