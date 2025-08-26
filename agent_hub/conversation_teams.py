from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import MainState
from utils.get_latest_question import get_latest_question

class ConversationTeams:
    def __init__(self, model: ChatOllama):
        self.model = model

    def build(self):
        g = StateGraph(MainState)
        g.add_node("help_desk_node", self.help_desk)
        g.add_edge(START, "help_desk_node")
        g.add_edge("help_desk_node", END)
        return g.compile()
    
    def help_desk(self, state: MainState):
        response = self.model.invoke(state["messages"])

        return {
            "messages": [AIMessage(content=response.content)],
        }
