from langchain.schema import AIMessage, HumanMessage

class Message:
    def __init__(self, content: str, is_human: bool = True):
        self.content = content
        self.message = HumanMessage(content=content) if is_human else AIMessage(content=content)

    def format_for_display(self) -> str:
        prefix = "You" if isinstance(self.message, HumanMessage) else "Assistant"
        return f"**{prefix}:** {self.content}"
