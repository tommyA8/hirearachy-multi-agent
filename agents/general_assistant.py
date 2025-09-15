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
            "You're a friendly assistant and your goal is to answer general questions.\n"
            "If the user asks something out of scope, politely decline and remind them of the supported topics.\n"
            "Keep answers concise, clear, and polite.\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(MessagesState)
        g.add_node("general", self.general_assistant)

        g.add_edge(START, "general")
        g.add_edge("general", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()
    
    def general_assistant(self, state: MessagesState):
        res = self.model.invoke([SystemMessage(content=self.prompt)] + state['messages'])

        # Ensure output is an AIMessage or at least has 'content'
        ai_message = res if isinstance(res, AIMessage) else AIMessage(content=res['messages'][-1].content)
        return {"messages": [ai_message]}
    
