from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.messages import SystemMessage

from config.settings import GEMINI_MODEL

def create_chat_model():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        convert_system_message_to_human=True
    )

def create_system_message():
    return SystemMessage(
        content=(
            "You are a nice customer service agent."
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
            "If you don't know the answer, just say you don't know. "
            "Don't try to make up an answer."
            "Make sure to answer in Korean"
        )
    )