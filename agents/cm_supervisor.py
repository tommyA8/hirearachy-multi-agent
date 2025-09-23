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
    tool: Literal["RFI", "SUBMITTAL", "INSPECTION", "NEED_MORE_CNTX", "NON_CM_TOOL"]

class RouterState(MessagesState):
    tool: str
    user: UserContext
    

class CMSupervisor:
    RFI = "RFI"
    SUBMITTAL = "SUBMITTAL"
    INSPECTION = "INSPECTION"
    NEED_MORE_CNTX = "NEED_MORE_CNTX"
    NON_CM_TOOL = "NON_CM_TOOL"
    NO_VALID = "NO_VALID"
    VALID_TOOLS = {RFI, SUBMITTAL, INSPECTION, NEED_MORE_CNTX, NON_CM_TOOL}

    def __init__(self, model: ChatOllama):
        self.model = model
        self.prompt = (
            "You are a Construction Management (CM) domain expert. Classify the user query into EXACTLY one CM tool label.\n"
            "You can use the provided **CHAT HISTORY** to infer intent.\n"
            "Allowed labels: RFI, SUBMITTAL, INSPECTION, NEED_MORE_CNTX, NON_CM_TOOL.\n"
            "Definitions:\n"
            "- RFI: Formal clarification workflow (deadlines, tracking).\n"
            "- SUBMITTAL: Review/approval of materials, shop drawings, product data.\n"
            "- INSPECTION: Field inspections, photos, comments, corrective actions.\n"
            "- NEED_MORE_CNTX: The user's latest query seems ambiguous or lacks sufficient context to classify.\n"
            "- NON_CM_TOOL: None of the above.\n\n"
            "Use ONLY the chat history for context.\n\n"
            "LATEST QUERY:\n{query}\n"
            "You must respond ONLY in the following strict JSON format:\n"
            "```json\n"
            "{{\n"
            "  \"tool\": \"RFI\" or \"SUBMITTAL\" or \"INSPECTION\" or \"NEED_MORE_CNTX\" or \"NON_CM_TOOL\",\n"
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
        g.add_node("ask_for_more_context_node", self.ask_for_more_context)

        g.add_edge(START, "cm_tool_router_node")
        g.add_conditional_edges(
            "cm_tool_router_node",
            lambda s: s['tool'],
            {
                self.RFI: "check_permission_node",
                self.SUBMITTAL: "check_permission_node",
                self.INSPECTION: "check_permission_node",
                self.NON_CM_TOOL: "answer_non_cm_tool",
                self.NEED_MORE_CNTX: "ask_for_more_context_node",
            }
        )
        g.add_edge("check_permission_node", END)
        g.add_edge("answer_no_permission_node", END)
        g.add_edge("answer_non_cm_tool", END)
        g.add_edge("ask_for_more_context_node", END)

        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_tool_router(self, state: RouterState) -> RouterState:
        prompt = self.prompt.format(query=get_latest_question(state))
        try:
            resp = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
            tool_res = Tools(tool=self.parse_model_output(resp.content, self.VALID_TOOLS))
            chosen = getattr(tool_res, "tool", None)
        except Exception:
            chosen = None

        if chosen not in self.VALID_TOOLS:
            chosen = self.NON_CM_TOOL

        return {"tool": chosen}
    
    def check_permission(self, state: RouterState) -> RouterState:
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
        res = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])

        return {"messages": res.content, "tool": state["tool"]}
    
    def ask_for_more_context(self, state: RouterState) -> RouterState:
        prompt = (
            "The user's latest query seems ambiguous or lacks sufficient context to classify.\n"
            "Please ask a clarifying question to gather more details about their intent.\n\n"
            "LATEST QUERY:\n{query}\n\n"
            "Your clarifying question should be concise and directly related to understanding whether the query is about construction management (CM) or general topics.\n"
        ).format(query=get_latest_question(state))
        
        resp = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
                
        return {"messages": resp.content, "tool": self.NEED_MORE_CNTX}
    
    @staticmethod
    def parse_model_output(text: str, allowed: set[str]) -> str | None:
        if not text:
            return None

        text = text.strip()

        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        upper_only = text.upper()
        if upper_only in allowed:
            return upper_only

        candidates = [m.group(0) for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL)]
        if not candidates:
            broad = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
            candidates.extend(broad[:3])

        def try_parse(snippet: str):
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                repaired = re.sub(r"'", '"', snippet)
                try:
                    return json.loads(repaired)
                except Exception:
                    return None
            except Exception:
                return None

        for snip in candidates:
            if 'tool' not in snip.lower():
                continue
            obj = try_parse(snip)
            if isinstance(obj, dict):
                val = obj.get('tool') or obj.get('Tool') or obj.get('TOOL')
                if isinstance(val, str):
                    val_up = val.strip().upper()
                    if val_up in allowed:
                        return val_up

        # Regex heuristic "tool": <label>
        m = re.search(r'"?tool"?\s*[:=]\s*"?(RFI|SUBMITTAL|INSPECTION|NEED_MORE_CNTX|NON_CM_TOOL)"?', text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # Standalone token heuristic
        token = re.search(r"\b(RFI|SUBMITTAL|INSPECTION|NEED_MORE_CNTX|NON_CM_TOOL)\b", text, flags=re.IGNORECASE)
        if token:
            return token.group(1).upper()

        return None