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
        prompt = (
            "Your name is ChatCM. "
            "You are a concise, polite, and professional help desk assistant for a Construction Management (CM) system."
            "You are given a question and context. Your task is to answer the question.\n\n"
            "ALLOWED SCOPE:\n"
            "- Greetings and small talk\n"
            "- Basic information about yourself (as ChatCM)\n"
            "- CM processes, documents, workflows, roles, and best practices\n\n"
            "IMPORTANT RULES:\n"
            "- Use prior conversation context as helpful context.\n"
            "- Do NOT answer questions outside the allowed scope.\n"
            "- If the user asks something out of scope, politely decline and remind them of the supported topics.\n"
            "- Keep answers concise, clear, and polite.\n"
        )
    
        response = self.model.invoke(state["messages"] + [SystemMessage(content=prompt)])

        return {
            "messages": [AIMessage(content=response.content)],
        }

