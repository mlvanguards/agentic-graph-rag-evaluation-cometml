import streamlit as st
from research_agent.orchestrator.coordinator import Coordinator
from research_agent.streamlit.layout import ResearchAssistantUI

def main():
    if "app" not in st.session_state:
        st.session_state.app = Coordinator()

    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "show_predefined" not in st.session_state:
        st.session_state.show_predefined = True
    if "session_active" not in st.session_state:
        st.session_state.session_active = True

    ui = ResearchAssistantUI(st.session_state.app, st.session_state)
    ui.render()