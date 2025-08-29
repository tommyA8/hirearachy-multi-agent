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

# @tool
# def semantic_search_cm_tool(human_question):
#     # """
#     # COSTLY. Only call if you truly cannot classify from your own knowledge.
#     # Returns at most 2 short snippets from the CM User Manual to unblock you.
#     # """
#     """
#     COSTLY. Only call if you truly cannot classify from your own knowledge.
#     Returns at most 2 short snippets from the CM User Manual to unblock you.
#     """

#     collection_name = "CM-User-Manual"
#     embedder_name = "bge-m3:latest"
#     qdrant = QdrantVector(qdrant_url=os.getenv("QDRANT_URL"), 
#                           collection_name=collection_name,
#                           model_name=embedder_name)

#     response = qdrant.client.query_points(
#         collection_name=collection_name,
#         query=qdrant.embedder.embed_query(human_question),
#         search_params=models.SearchParams(hnsw_ef=128, exact=False),
#         limit=2,
#     )

#     relavant_cntx = [pt.payload for pt in response.points]
#     return relavant_cntx

class RouterTeams:
    def __init__(self, model: ChatOllama):
        self.model = model
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

    def tool_classification(self, state: RouterState):
        # Get latest human question
        question = get_latest_question(state)

        # Create prompt with available tools
        prompt_str = self.prompt.format(cm_tools=CM_TOOLS)

        response = self.model.invoke(question + [SystemMessage(content=prompt_str)])

        # Parse two-line format
        text = response.content.strip()
        cm_tool, reason = self.extract_tool_and_reason(text)

        return {
            "tool": cm_tool,
            "tool_selected_reason": reason,
        }


    # def generate_structured_response(self, state: RouterState):
    #     # Create prompt
    #     prompt_str = self.prompt.format(cm_tools=CM_TOOLS)

    #     # Then answer with structed output
    #     structed_model = self.model.with_structured_output(RoutingDecision)
    #     response = structed_model.invoke([state['messages'][-1]] + [SystemMessage(content=prompt_str)])

    #     # Extract info
    #     cm_tool = response.tool
    #     tool_selected_reason = response.tool_selected_reason
    #     ai_msg = AIMessage(content=f"Tool: {cm_tool}\nReason: {tool_selected_reason}")

    #     return {
    #         "messages": [ai_msg],
    #         "tool": cm_tool,
    #         "tool_selected_reason": tool_selected_reason,
    #     }
    