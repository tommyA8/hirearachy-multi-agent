from typing import Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question

class GeneralAssistant:
    def __init__(self, model: ChatOllama):
        self.model = model
        self.prompt = (
            "Your name is ChatCM.\n"
            "You are a concise, polite, and professional help desk assistant for a Construction Management (CM) system.\n"
            "You're a friendly assistant and your goal is to answer general questions.\n"
            "If the user asks something out of scope, politely decline and remind them of the supported topics.\n"
            "Keep answers concise, clear, and polite.\n"
            "You must use the provided **CHAT HISTORY** to infer intent."
            "Question:\n{question}\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(MessagesState)
        g.add_node("general", self.general_assistant)

        g.add_edge(START, "general")
        g.add_edge("general", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()
    
    def general_assistant(self, state: MessagesState) -> MessagesState:
        prompt = self.prompt.format(question=get_latest_question(state))
        res = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])

        # Ensure output is an AIMessage or at least has 'content'
        ai_message = res if isinstance(res, AIMessage) else AIMessage(content=res['messages'][-1].content)
        return {"messages": [ai_message]}
    
