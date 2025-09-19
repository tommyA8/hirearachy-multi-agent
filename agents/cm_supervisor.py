from typing import Dict, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from utils.get_latest_question import get_latest_question
import enum
from utils.tools import get_current_weather, search_tool

class Tools(BaseModel):
    tool: Literal["RFI", "SUBMITTAL", "INSPECTION", "UNKNOWN"]

class RouterState(MessagesState):
    tool: str

class CMSupervisor:
    def __init__(self, model: ChatOllama):
        self.model = model
        self.structured_model = self.model.with_structured_output(Tools)
        self.prompt = (
            "You are a Construction Management (CM) domain expert. Your task is to classify the incoming query into the single most relevant CM Tools.\n"
            "Available Tools (Choose one):\n"
            "- RFI - Formal clarification process with workflow, deadlines, and status tracking.\n"
            "- SUBMITTAL - Digital review/approval process for materials, shop drawings, and product data.\n"
            "- INSPECTION - Field inspections logged digitally with photos, comments, and corrective actions.\n"
            "- UNKNOWN - No relevant tool identified.\n\n"
            "You must use the provided **CHAT HISTORY** to infer intent.\n"
            "QUERY:\n{query}\n"
            "Return in the output only one word (RFI, SUBMITTAL or INSPECTION).\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(RouterState)
        g.add_node("cm_tool_router_node", self.cm_tool_router)
        g.add_node("non_cm_tool_node", self.non_cm_tool_answer)

        g.add_edge(START, "cm_tool_router_node")
        g.add_conditional_edges(
            "cm_tool_router_node",
            lambda s: s['tool'],
            {"UNKNOWN": "non_cm_tool_node", "RFI": END, "SUBMITTAL": END, "INSPECTION": END}
        )
        g.add_edge("non_cm_tool_node", END)

        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_tool_router(self, state: RouterState) -> RouterState:
        prompt = self.prompt.format(query=get_latest_question(state))
        tool_res = self.structured_model.invoke([SystemMessage(content=prompt)] + state['messages'])
    
        return {"tool": tool_res.tool if tool_res is not None else "UNKNOWN"}
    
    def non_cm_tool_answer(self, state: RouterState) -> RouterState:
        prompt = (
            "You are a Construction Management (CM) domain expert and SitearoundCM SaaS consultant in Thailand. Your task is to answer the incoming query.\n"
            "You can use the provided **CHAT HISTORY**  and **search_tool** to infer intent.\n"
            "DO NOT answer questions that are not related to construction management.\n\n"
            "QUERY:\n{query}\n"
        )
        prompt = prompt.format(query=get_latest_question(state))
        agent = create_react_agent(self.model, tools=[search_tool], prompt=prompt)

        res = agent.invoke({"messages": state['messages']})
        return {"messages": res['messages'][-1]}

