from dataclasses import dataclass
from typing import List, Callable

@dataclass
class PredefinedQuestion:
    text: str
    callback: Callable[[str], None]

class PredefinedQuestionsManager:
    def __init__(self):
        self.questions = [
            "Explain the concept of transformer models in NLP.",
            "What is the significance of paper ID 0704.2002?",
            "How does reinforcement learning apply to robotics?",
            "Summarize recent advances in computer vision."
        ]

    def get_questions(self) -> List[str]:
        return self.questions
