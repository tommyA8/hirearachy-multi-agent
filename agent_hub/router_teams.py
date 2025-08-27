import os
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
            "Your task is to carefully analyze a user's question and map it to the most relevant CM Feature. "
            "If no feature clearly applies, select 'Unknown'.\n\n"

            "Follow this reasoning process step by step:\n"
            "1. Analyze the user’s question to identify the core intent (what they really want or need).\n"
            "2. Check if the intent is related to Construction Management.\n"
            "3. If related, select the single CM Feature that best fits the intent.\n"
            "   - If multiple could apply, choose the one most directly responsible for solving the problem.\n"
            "   - If no feature clearly applies, return 'Unknown'.\n\n"

            "Available CM Features:\n{cm_tools}\n\n"

            "Your response must be a PLAIN TEXT with the following fields:\n"
            "- features: The chosen feature name from the list above (or 'Unknown').\n"
            "- features_selected_reason: A concise explanation (2–3 sentences) that:\n"
            "   • Identifies the user’s core intent\n"
            "   • Explains why this feature addresses that intent based on CM business logic\n"
            "   • Justifies why other features are less relevant\n"
        )

    def build(self):
        g = StateGraph(RouterState)
        g.add_node("tool_router_node", self.tool_classification)
        g.add_node("generate_structured_response", self.generate_structured_response)

        g.add_edge(START, "tool_router_node")
        g.add_edge("tool_router_node", "generate_structured_response")
        g.add_edge("generate_structured_response", END)
        return g.compile()

    def tool_classification(self, state: RouterState):
        # Get latest human question
        question = get_latest_question(state)

        # Create prompt
        prompt_str = self.prompt.format(cm_tools=CM_TOOLS)

        # Let model thinking first
        response = self.model.invoke(question + [SystemMessage(content=prompt_str)])

        return {"messages": response}

    def generate_structured_response(self, state: RouterState):
        # Create prompt
        prompt_str = self.prompt.format(cm_tools=CM_TOOLS)

        # Then answer with structed output
        structed_model = self.model.with_structured_output(RoutingDecision)
        response = structed_model.invoke(state['messages'] + [SystemMessage(content=prompt_str)])

        # Extract info
        cm_tool = response.tool
        tool_selected_reason = response.tool_selected_reason
        ai_msg = AIMessage(content=f"Tool: {cm_tool}\nReason: {tool_selected_reason}")

        return {
            "messages": [ai_msg],
            "tool": cm_tool,
            "tool_selected_reason": tool_selected_reason,
        }
    