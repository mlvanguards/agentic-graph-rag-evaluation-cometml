from abc import ABC, abstractmethod
from src.core.state import ConversationState
from typing import Dict, Any

class BaseAgent(ABC):
    @abstractmethod
    def process_message(self, state: ConversationState) -> Dict[str, Any]:
        """Process a message and return updated state."""
        pass

    @abstractmethod
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors during message processing."""
        pass