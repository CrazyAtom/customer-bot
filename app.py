import logging
import time

import streamlit as st

from agent.customer_service import setup_agent
from config.settings import GEMINI_MODEL
from models.chat_model import create_chat_model
from utils.loader import load_and_process_data

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.title("AI 고객 서비스 상담원")

# 데이터 로드 및 초기 설정
@st.cache_resource  # 캐싱을 통한 성능 개선
def initialize_agent():
    try:
        # retriever = load_and_process_data("https://conspara.kr/shop_view/?idx=139?scrollTo=custom", source_type="url")
        retriever = load_and_process_data("guide.ini", source_type="file")
        if not retriever:
            st.error("Failed to load data from the URL")
            return None

        llm = create_chat_model()
        return setup_agent(retriever, llm)
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

# 에이전트 초기화
agent_executor = initialize_agent()

if agent_executor is None:
    st.error("Failed to initialize the chat agent. Please refresh the page.")
    st.stop()

if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = GEMINI_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = agent_executor({"input": prompt})
        for chunk in result['output'].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})