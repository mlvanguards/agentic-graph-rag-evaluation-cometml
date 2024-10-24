import streamlit as st
from typing import List, Callable
from research_agent.streamlit.message import Message
from langchain.schema import HumanMessage, AIMessage

class ChatDisplay:
    @staticmethod
    def display_messages(messages: List):
        for message in messages:
            if hasattr(message, 'format_for_display'):
                formatted_message = message.format_for_display()
            elif isinstance(message, HumanMessage):
                formatted_message = f"**You:** {message.content}"
            elif isinstance(message, AIMessage):
                formatted_message = f"**Assistant:** {message.content}"
            else:
                formatted_message = f"**Unknown:** {message.content}"

            st.write(formatted_message)
        st.markdown("<br>" * 2, unsafe_allow_html=True)


class InputArea:
    def __init__(self, on_submit: Callable[[str], None]):
        self.on_submit = on_submit

    def render(self):
        st.text_input(
            "Enter your question here:",
            key="user_input",
            on_change=self._handle_input
        )

    def _handle_input(self):
        if st.session_state.user_input:
            self.on_submit(st.session_state.user_input)
            st.session_state.user_input = ""

class SessionControls:
    def __init__(self, on_end_session: Callable[[], None]):
        self.on_end_session = on_end_session

    def render(self):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("End Session"):
                self.on_end_session()