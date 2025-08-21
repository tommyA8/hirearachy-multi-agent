from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import RouterState, RoutingDecision

class RouterTeams:
    def __init__(self, model: ChatOllama):
        self.model = model

    def build(self):
        g = StateGraph(RouterState)
        g.add_node("tool_router_node", self.tool_router)
        g.add_edge(START, "tool_router_node")
        g.add_edge("tool_router_node", END)
        return g.compile()

    def tool_router(self, state: RouterState):
        prompt = """
        You are a Costruction Management expert. You know End to End Construction Management process.
        Analyze the user query below and determine its Available Tools with deeply reason.

        Available tools (Choose one):
            - Document: For questions about user's documents related to the project that they uploaded to the system. It's not related to attachmented submittal files or other.
            - Submittal: For questions about construction submittals.
            - RFI: For questions about construction requests for information (RFI).
            - Inspection: For questions about construction inspections.
            - Work Order: For questions about construction work orders.
            - Unknown: If the user's query is not related to any of the above features. Try to answer the user's question as best you can and take the user to the next step of the process.

        Return a JSON object with fields: question, tool, selected_reason.

        Query: {question}
        """

        agent = self.model.with_structured_output(RoutingDecision)

        decision = agent.invoke(prompt.format(question=state["messages"][-1].content))

        return {
            "messages": [AIMessage(content=f"LLM Router decided: {decision.tool.value}")],
            "tool": decision.tool,
            "selected_reason": decision.selected_reason,
        }
