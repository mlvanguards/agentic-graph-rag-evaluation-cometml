from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class ConversationMessage:
    content: str
    type: str
    timestamp: datetime
    query: Optional[str] = None
    paper_id: Optional[str] = None
    response: Optional[str] = None
