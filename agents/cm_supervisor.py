import re
import json
from typing import Literal
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from utils.get_latest_question import get_latest_question
from utils.tools import search_tool
from model.user import UserContext

class Tools(BaseModel):
    """Pydantic schema for structured tool classification output."""
    tool: Literal["RFI", "SUBMITTAL", "INSPECTION", "UNKNOWN"]

class RouterState(MessagesState):
    tool: str
    user: UserContext
    

class CMSupervisor:
    """Routes a CM-related user query to the appropriate CM tool or fallback.

    Flow:
      1. Classify latest user question into one of (RFI, SUBMITTAL, INSPECTION, UNKNOWN).
      2. Check user permission; if insufficient return NO_VALID sentinel.
      3. For UNKNOWN -> fallback ReAct QA with search tool restricted to CM domain.
    """

    # Canonical tool labels and sentinels
    RFI = "RFI"
    SUBMITTAL = "SUBMITTAL"
    INSPECTION = "INSPECTION"
    UNKNOWN = "UNKNOWN"
    NO_VALID = "NO_VALID"
    VALID_TOOLS = {RFI, SUBMITTAL, INSPECTION, UNKNOWN}

    def __init__(self, model: ChatOllama):
        self.model = model
        self.structured_model = self.model.with_structured_output(Tools)
        # Prompt explicitly includes UNKNOWN and mandated single token answer
        self.prompt = (
            "You are a Construction Management (CM) domain expert. Classify the latest user query into EXACTLY one CM tool label.\n"
            "Allowed labels: RFI, SUBMITTAL, INSPECTION, UNKNOWN.\n"
            "Definitions:\n"
            "- RFI: Formal clarification workflow (deadlines, tracking).\n"
            "- SUBMITTAL: Review/approval of materials, shop drawings, product data.\n"
            "- INSPECTION: Field inspections, photos, comments, corrective actions.\n"
            "- UNKNOWN: None of the above.\n\n"
            "Use ONLY the chat history for context.\n\n"
            "LATEST QUERY:\n{query}\n"
            "You must respond ONLY in the following strict JSON format:\n"
            "```json\n"
            "{{\n"
            "  \"tool\": \"RFI\" or \"SUBMITTAL\" or \"INSPECTION\" or \"UNKNOWN\",\n"
            "}}\n"
            "```\n"
            "Do not include any text outside the JSON.\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(RouterState)
        g.add_node("cm_tool_router_node", self.cm_tool_router)
        g.add_node("check_permission_node", self.check_permission)
        g.add_node("answer_no_permission_node", self.answer_no_permission)
        g.add_node("answer_non_cm_tool", self.answer_non_cm_tool)

        g.add_edge(START, "cm_tool_router_node")
        g.add_edge("cm_tool_router_node", "check_permission_node")
        g.add_conditional_edges(
            "check_permission_node",
            lambda s: s['tool'],
            {
                "NO_VALID": "answer_no_permission_node",
                "UNKNOWN": "answer_non_cm_tool", 
                "RFI": END, "SUBMITTAL": END, "INSPECTION": END
            }
        )
        g.add_edge("answer_no_permission_node", END)
        g.add_edge("answer_non_cm_tool", END)

        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_tool_router(self, state: RouterState) -> RouterState:
        """Invoke the LLM to classify the latest question.

        Returns only the chosen tool label. Falls back to UNKNOWN on any failure
        or invalid model output.
        """
        prompt = self.prompt.format(query=get_latest_question(state))
        try:
            resp = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
            tool_res = Tools(tool=self.parse_model_output(resp.content))
            chosen = getattr(tool_res, "tool", None)
        except Exception:
            chosen = None

        if chosen not in self.VALID_TOOLS:
            chosen = self.UNKNOWN

        return {"tool": chosen}
    
    def check_permission(self, state: RouterState) -> RouterState:
        """Map disallowed tool selection to NO_VALID sentinel.

        A tool is disallowed if user's permission level < 1. UNKNOWN bypasses permission.
        """
        if state.get("tool") == self.UNKNOWN:
            return {"tool": self.UNKNOWN}

        not_valid_tools = [pm.tool for pm in state["user"].tool_permissions if pm.level < 1]
        if state["tool"] in not_valid_tools:
            return {"tool": self.NO_VALID}
        return {"tool": state["tool"]}
    
    def answer_no_permission(self, state: RouterState) -> RouterState:
        """Return explanatory denial message including the requested tool."""
        requested = state.get("tool", "")
        return {"messages": AIMessage(content=f"You do not have sufficient permission to access {requested} data. Please contact your administrator.")}
    
    def answer_non_cm_tool(self, state: RouterState) -> RouterState:
        prompt = (
            "You are a Construction Management (CM) domain expert and SitearoundCM SaaS consultant in Thailand. Your task is to answer the incoming query.\n"
            "You can use the provided **CHAT HISTORY** and **search_tool** to infer intent.\n"
            "DO NOT answer questions that are not related to construction management.\n\n"
            "QUERY:\n{query}\n"
        )
        prompt = prompt.format(query=get_latest_question(state))
        agent = create_react_agent(self.model, tools=[search_tool], prompt=prompt)

        res = agent.invoke({"messages": state['messages']})
        return {"messages": res['messages'][-1]}

    @staticmethod
    def parse_model_output(text: str) -> str:
        if not text:
            return None
        # find first JSON object occurrence
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if not match:
            return None
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            candidate = obj.get('tool')
            if isinstance(candidate, str):
                return candidate
        except Exception:
            print("Failed to parse JSON")
            return None
        return None