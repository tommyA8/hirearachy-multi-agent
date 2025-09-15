import os
import re
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from utils.qdrant_helper import QdrantVector
from qdrant_client import QdrantClient, models
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# Model
from model.state_model import RouterState, RoutingDecision
# Utils
from utils.get_latest_question import get_latest_question
from model.state_model import Tools

cm_tools = [tool.tool for tool in Tools]
tool_descriptions = [tool.description for tool in Tools]
CM_TOOLS = "\n".join([f"- {tool}: {desc}" for tool, desc in zip(cm_tools, tool_descriptions)])

class RouterTeams:
    def __init__(self, model: ChatOllama):
        self.model = model
        self.structured_model = ChatOllama(model="qwen3:0.6b", temperature=0).with_structured_output(RoutingDecision)
        self.relavant_cntx = ""
        self.prompt = (
            "You are a Construction Management (CM) domain expert. "
            "Classify the user’s question into the single most relevant CM Feature, or 'Unknown' if none apply.\n\n"
            "Reasoning steps:\n"
            "1. Identify the user’s core intent.\n"
            "2. Check if it relates to Construction Management.\n"
            "3. If yes, choose the ONE best-fitting feature from the list. If none, return 'Unknown'.\n\n"
            "Available CM Features:\n{cm_tools}\n\n"
            "Respond ONLY in a valid JSON object with this format:\n"
            "```json\n"
            "{{\n"
            "  \"tool\": \"<CM Feature>\",\n"
            "  \"tool_selected_reason\": \"<Reason>\",\n"
            "}}\n"
            "```"
        )
        
    def build(self):
        g = StateGraph(RouterState)
        g.add_node("tool_router_node", self.tool_classification)
        g.add_edge(START, "tool_router_node")
        g.add_edge("tool_router_node", END)
        return g.compile()

    @staticmethod
    def extract_tool_and_reason(text: str):
        # grab JSON inside ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S|re.I)
        json_str = m.group(1) if m else re.search(r"(\{.*\})", text, flags=re.S).group(1)
        data = json.loads(json_str)
        return data["tool"], data["tool_selected_reason"]
    #TODO: ทำให้มันอ่าน chat history แทนที่จะดูจากคำถามล่าสุดอย่างเดียว ไม่งั้น ถ้า user ถามต่อจากคำถามเดิม มันจะเข้าใจว่าไม่เกี่ยวกับ tool ไหนเลย
    # TODO: ให้มันวิเคราหะ์ว่า คำถามปัจจุบัน ต้องการอะไร และเกี่ยวกับ Tool ไหน โดยวิเคราะห์ history chat
    def tool_classification(self, state: RouterState):
        # Get latest human question
        question = get_latest_question(state)

        # Create prompt with available tools
        prompt_str = self.prompt.format(cm_tools=CM_TOOLS)

        # Find best tool and reason
        res = self.model.invoke(question + [SystemMessage(content=prompt_str)])
        
        # Then answer with structed output
        response = self.structured_model.invoke([res] + [SystemMessage(content=prompt_str)])

        # Extract info
        cm_tool = response.tool
        reason = response.tool_selected_reason

        return {
            "tool": cm_tool,
            "tool_selected_reason": reason,
        }
