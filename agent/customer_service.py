from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.schema.messages import SystemMessage
from langchain_core.agents import AgentAction, AgentFinish


def create_system_message():
    return SystemMessage(
        content=(
            "You are a nice customer service agent. "
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary. "
            "If you don't know the answer, just say you don't know. "
            "Don't try to make up an answer. "
            "Make sure to answer in Korean"
        )
    )

def setup_agent(retriever, llm):
    tool = create_retriever_tool(
        retriever,
        "Dalpha_customer_service_guide",
        "Searches and returns information regarding the customer service guide.",
    )
    tools = [tool]

    memory_key = "history"
    memory = ConversationTokenBufferMemory(
        memory_key=memory_key,
        llm=llm,
        return_messages=True
    )

    system_message = create_system_message()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message.content),
        MessagesPlaceholder(variable_name=memory_key),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )