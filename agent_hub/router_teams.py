from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import RouterState, RoutingDecision
# Utils
from utils.get_latest_question import get_latest_question

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
        prompt = """
        You are a Costruction Management expert. You know end to end Construction Management process.
        Analyze the user query below and determine its Available Tools with deeply reason.

        question:
        {question}
        """

        agent = self.model.with_structured_output(RoutingDecision)

        question = get_latest_question(state)
        decision = agent.invoke(prompt.format(question=question[-1].content))

        return {
            "messages": [AIMessage(content=f"LLM Router decided: {decision.tool.value}")],
            "tool": decision.tool,
            "tool_selected_reason": decision.tool_selected_reason,
        }

