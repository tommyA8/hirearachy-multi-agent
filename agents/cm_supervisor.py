from typing import Dict, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question
import enum

class RouterState(MessagesState):
    tool: str

class CMSupervisor:
    def __init__(self, model: ChatOllama):
        self.model = model
        self.prompt = (
            "You are a Construction Management (CM) domain expert. Your task is to classify the incoming QUESTIONs into the single most relevant CM Tools.\n"
            "Depending on your answer, question will be routed to the right team, so your task is crucial for our team.\n"
            "There are 3 possible CM Tools:\n"
            "- RFI - Formal clarification process with workflow, deadlines, and status tracking.\n"
            "- SUBMITTAL - Digital review/approval process for materials, shop drawings, and product data.\n"
            "- INSPECTION - Field inspections logged digitally with photos, comments, and corrective actions.\n"
            "You must use the provided **CHAT HISTORY** to infer intent.\n"
            "QUESTION:\n{question}\n"
            "Return in the output only one word (RFI, SUBMITTAL or INSPECTION).\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(RouterState)
        g.add_node("cm_tool_router_node", self.cm_tool_router)

        g.add_edge(START, "cm_tool_router_node")
        g.add_edge("cm_tool_router_node", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_tool_router(self, state: RouterState) -> RouterState:
        prompt = self.prompt.format(question=get_latest_question(state))
        res = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
    
        return {"tool": res.content}

