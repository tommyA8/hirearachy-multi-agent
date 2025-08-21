from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from model.state_model import MainState


def get_latest_question(state: MainState) -> HumanMessage:
    return [next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)))]
