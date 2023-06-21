from dataclasses import dataclass
from typing import Literal
import streamlit as st
import streamlit.components.v1 as components
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

class ChatApplication:
    def __init__(self):
        load_dotenv()
        st.title("Lucidate Robo Advisor:")
        st.title("Powered by LangChain 🦜🔗")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The following is an informative conversation between a human and an AI financial adviser. The financial adviser will ask lots of questions. The financial adviser will attempt to answer any question asked and will probe for the human's risk appetite by asking questions of its own. If the human's risk appetite is low it will offer conservative financial advice, if the risk appetite of the human is higher it will offer more aggressive advice "
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        self.load_css()
        self.initialize_session_state()

        self.chat_placeholder = st.container()
        self.prompt_placeholder = st.form("chat-form")
        self.log_placeholder = st.empty()

    def load_css(self):
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

    def initialize_session_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "token_count" not in st.session_state:
            st.session_state.token_count = 0
        if "conversation" not in st.session_state:
            llm = ChatOpenAI(
                temperature=0,
                openai_api_key=self.openai_api_key,
            )
            st.session_state.conversation = ConversationChain(
                llm=llm,
                memory=ConversationBufferMemory(return_messages=True),
                prompt=self.prompt
            )

    def on_click_callback(self):
        with get_openai_callback() as cb:
            human_prompt = st.session_state.human_prompt
            history = st.session_state.history
            llm_response = st.session_state.conversation.predict(
                input=human_prompt
            )
            st.session_state.history.append(
                Message("human", human_prompt)
            )
            st.session_state.history.append(
                Message("ai", llm_response)
            )
            st.session_state.token_count += cb.total_tokens

    def run(self):
        with self.chat_placeholder:
            for chat in st.session_state.history:
                div = f"""
        <div class="chat-row 
            {'' if chat.origin == 'ai' else 'row-reverse'}">
            <img class="chat-icon" src="app/static/{
                'ai_icon.png' if chat.origin == 'ai'
                else 'user_icon.png'}"
                width=32 height=32>
            <div class="chat-bubble
            {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
                """
                st.markdown(div, unsafe_allow_html=True)

            for _ in range(3):
                st.markdown("")

        with self.prompt_placeholder:
            st.markdown("**Chat**")
            cols = st.columns((6, 1))
            cols[0].text_input(
                "Chat",
                value="Hi Lucidate FinBot!",
                label_visibility="collapsed",
                key="human_prompt",
            )
            cols[1].form_submit_button(
                "Submit",
                type="primary",
                on_click=self.on_click_callback,
            )

        self.log_placeholder.caption(f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain conversation: 
    {st.session_state.conversation.memory.buffer}
    """)

app = ChatApplication()
app.run()
