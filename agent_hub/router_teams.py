from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import RouterState, RoutingDecision
# Utils
from utils.get_latest_question import get_latest_question
from model.state_model import Tools

tools = [tool.value for tool in Tools]

class RouterTeams:
    def __init__(self, model: ChatOllama):
        self.model = model

    def build(self):
        g = StateGraph(RouterState)
        g.add_node("tool_router_node", self.tool_classification)
        g.add_edge(START, "tool_router_node")
        g.add_edge("tool_router_node", END)
        return g.compile()

    def tool_classification(self, state: RouterState):
        prompt = (
        "You are a Construction Management (CM) domain expert. "
        "You are given a user question and must decide which CM tool (if any) is most relevant. "
        "Follow these rules step by step:\n"
        "1. Identify what the user is asking for (the core intent).\n"
        "2. Determine if the question is related to Construction Management.\n"
        "3. If related, decide which CM tool best fits the question.\n"
        "   - If no tool clearly applies, respond with 'Unknown'.\n\n"
        "Available Tools: {tools}\n"
        "User Question: {question}\n\n"
        "Your response must include:\n"
        "- The chosen tool from the list or 'Unknown'.\n"
        "- A short reasoning.\n"
        "- Justification based on Construction Management business logic.\n"
        )
        
        agent = self.model.with_structured_output(RoutingDecision)

        question = get_latest_question(state)
        question = question[-1].content

        decision = agent.invoke(state['messages'] + [SystemMessage(content=prompt.format(question=question, tools=tools))])

        ai_msg = AIMessage(content=f"{decision.tool.value} Tool is most related to user's question.")
        return {
            "messages": [ai_msg],
            "tool": decision.tool.value,
            "tool_selected_reason": decision.tool_selected_reason,
        }

