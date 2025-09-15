from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from utils.is_valid_tool_permission import is_valid_tool_permission
from utils.get_latest_question import get_latest_question
from model.state_model import *
# from agent_hub import RouterTeams, ConversationTeams, DatabaseTeams, ResearchTeams
from agents.is_cm_related import CMRelated
from agents.general_assistant import GeneralAssistant

from typing import Dict, List, Optional
import enum


class PermissionLevel(enum.Enum):
    Not_Allowed = 0
    View_Only = 1
    General = 2
    Admin = 3

class CMTools(enum.Enum):
    RFI = 0
    Submittal=  1
    Inspection = 2

class UserContext(BaseModel):
    class Permission(BaseModel):
        level: PermissionLevel
        tool: str

    user_id: int
    company_id: int
    project_id: int
    permission_tools: Optional[List[Permission]]

class MainState(MessagesState):
    user: UserContext
    question_type: str

class ChatCM:
    def __init__(self):
        self._cm_related = None
        self._general_assistant = None

    @property
    def cm_related_agent(self):
        return self._cm_related
    
    @property
    def general_assistant_agent(self):
        return self._general_assistant
    
    @cm_related_agent.setter
    def cm_related_agent(self, graph: CMRelated):
        self._cm_related = graph.build()

    @general_assistant_agent.setter
    def general_assistant_agent(self, graph: GeneralAssistant):
        self._general_assistant = graph.build()

    def is_related_to_cm_agent(self, state: MainState):
        question = get_latest_question(state)
        res = self._cm_related.invoke({
            "question": [HumanMessage(content=question)]
        })
        return {
            "question_type": res["question_type"]
        }
        
    
    def general_agent(self, state: MainState):
        res = self._general_assistant.invoke({
            "messages": state["messages"]
        })
        return {
            "messages": res["messages"]
        }

    def tool_permission(self, state: MainState):
        if not state['user'].permission_tools:
            res = is_valid_tool_permission(user_id=state["user"].user_id,
                                           project_id=state['user'].project_id,
                                           company_id=state['user'].company_id)

            # Filter only RFI, Submittal, Inspection
            tools = [tool.name for tool in CMTools]
            state['user'].permission_tools = [
                UserContext.Permission(level=PermissionLevel(tool[0]), tool=tool[1])
                for tool in res if tool[1] in tools
            ]
        
        return {"user": state['user']}
    
    def cm_supervisor_agent(self, state: MainState):
        return {
            "messages": AIMessage(content="Passing to supervisor...")
        }

    def build(self, checkpointer, save_graph=False):
        g = StateGraph(MainState)
        g.add_node("is_cm_related_node", self.is_related_to_cm_agent)
        g.add_node("general_assistant_node", self.general_agent)
        g.add_node("permission_node", self.tool_permission)
        g.add_node("cm_supervisor_node", self.cm_supervisor_agent)

        g.add_edge(START, "is_cm_related_node")
        g.add_conditional_edges(
            "is_cm_related_node",
            lambda s: s['question_type'],
            {"CM": "permission_node", "GENERAL": "general_assistant_node"}
        )
        g.add_edge("permission_node", "cm_supervisor_node")

        self.agent = g.compile(checkpointer=checkpointer)

        if save_graph:
            self.save_graph()

        return self.agent

    def save_graph(self):
        try:
            png_bytes = self.agent.get_graph().draw_mermaid_png()
            with open("chatcm_agentic_graph.png", "wb") as f:
                f.write(png_bytes)
            print("Saved graph diagram to graph.png")
        except Exception as e:
            print(f"Could not render graph diagram: {e}")

