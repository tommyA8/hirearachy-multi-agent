from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import MessagesState


def get_latest_question(state: MessagesState) -> str:
    return [next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)))][-1].content
