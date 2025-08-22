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

        question = get_latest_question(state)

        system = SystemMessage(content=(
                "Your name is ChatCM."
                "You are a helpful, concise, and polite help desk for a CM system."
                "You are given a question from a user.\n"
                # "Question: {question}\n"
                "You can only to answer questions only related to Construction Management (CM), greetings and about yourself. Follwing instructions:\n"
                "- Always use prior conversation context if helpful. Keep answers SHORT.\n"
                "- You can answer questions in table format if needed.\n"
                "- You can use database tables and columns as context.\n"
                "- DO NOT Answer with prompts.\n"
                "- DO NOT Answer with SQLQuery.\n"
            ))
        
        try:
            db_cntx = SystemMessage(content=f"Database Context:\n{state['relavant_context']}")
            response = self.model.invoke(state["messages"] + [system] + [db_cntx])

        except:
            # invoke the chat model correctly with a list of messages
            response = self.model.invoke(state["messages"] + [system])

        ai_msg = AIMessage(content=response.content)
        # ai_msg = AIMessage(content=response.content.split("</think>\n")[-1])
        return {
            "messages": [ai_msg],
        }
