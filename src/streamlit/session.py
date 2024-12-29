from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SessionState:
    messages: List[Any] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)
    show_predefined: bool = True
    session_active: bool = True
