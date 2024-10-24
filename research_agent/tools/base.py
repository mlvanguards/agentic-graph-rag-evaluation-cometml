from abc import ABC, abstractmethod
from typing import Any

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool's primary function"""
        pass