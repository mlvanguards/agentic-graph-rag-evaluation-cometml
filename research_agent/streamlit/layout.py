import streamlit as st
from research_agent.streamlit.ui_component import ChatDisplay, InputArea, SessionControls
from research_agent.streamlit.predefined_questions import PredefinedQuestionsManager


class ResearchAssistantUI:
    def __init__(self, coordinator, session_state):
        self.coordinator = coordinator
        self.session_state = session_state
        self.chat_display = ChatDisplay()
        self.input_area = InputArea(self._handle_user_input)
        self.session_controls = SessionControls(self._handle_session_end)
        self.predefined_questions = PredefinedQuestionsManager()

    def render(self):
        st.set_page_config(page_title="Research Paper Assistant", layout="wide")
        st.title("Research Paper Assistant")

        if not self.session_state.session_active:
            st.info("Session has ended. Please refresh the page to start a new session.")
            return

        chat_container = st.container()
        input_container = st.container()

        with chat_container:
            self.chat_display.display_messages(self.session_state.messages)

        with input_container:
            self.session_controls.render()
            self._render_predefined_questions()
            self.input_area.render()

    def _render_predefined_questions(self):
        if self.session_state.show_predefined:
            st.subheader("Predefined Questions")
            questions = self.predefined_questions.get_questions()
            col1, col2 = st.columns(2)

            for i, question in enumerate(questions):
                with col1 if i < len(questions) // 2 else col2:
                    st.button(
                        question,
                        key=f"pred_q_{i}",
                        on_click=self._handle_predefined_question,
                        args=(question,)
                    )

    def _handle_user_input(self, user_input: str):
        self.session_state.show_predefined = False
        self.coordinator.process_message(user_input, self.session_state)

    def _handle_predefined_question(self, question: str):
        self.session_state.show_predefined = False
        self.coordinator.process_message(question, self.session_state)

    def _handle_session_end(self):
        self.coordinator.cleanup()
        self.session_state.session_active = False
        st.success("Session has been ended. Thank you for using the Research Paper Assistant!")
        st.stop()
