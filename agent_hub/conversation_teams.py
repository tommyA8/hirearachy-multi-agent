from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import MainState

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
        system = SystemMessage(content=(
            "Your name is ChatCM.\n"
            "You are a helpful, concise, and polite help desk for a CM system.\n"
            "You can only answer questions related to Construction Management (CM), greetings and about yourself.\n"
            "Always use prior conversation context if helpful. Keep answers SHORT."
            "DO NOT Answer with prompts."
        ))
        
        # invoke the chat model correctly with a list of messages
        response = self.model.invoke(state["messages"] + [system])

        ai_msg = AIMessage(content=response.content)
        # ai_msg = AIMessage(content=response.content.split("</think>\n")[-1])
        return {
            "messages": [ai_msg],
        }
