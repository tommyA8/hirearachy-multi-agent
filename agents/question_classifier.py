from typing import Dict, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import  HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question

class QuestionType(BaseModel):
    type: Literal['CM', 'GENERAL']
    reason: str

class QuestionTypeState(MessagesState):
    question_type: str

class QuestionClassifier:
    def __init__(self, model: ChatOllama):
        self.model = model
        self.prompt = (
            "You are a domain expert in Construction Management (CM). "
            "Your task is to classify the user's latest query based on its intent.\n\n"
            "Classification Types (choose exactly one):\n"
            "- CM: query directly related to construction management topics.\n"
            "- GENERAL: All other questions not related to construction management.\n\n"
            "Use the provided **CHAT HISTORY** and the latest user message to determine the correct type. "
            "QUERY:\n{query}\n"
            "Respond with **only one word**: either 'CM' or 'GENERAL' — no extra text, punctuation, or explanation."
        )

    def build(self, checkpointer=None):
        g = StateGraph(QuestionTypeState)
        g.add_node("cm_classifier", self.cm_classifier)

        g.add_edge(START, "cm_classifier")
        g.add_edge("cm_classifier", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_classifier(self, state: QuestionTypeState) -> QuestionTypeState:
        prompt = self.prompt.format(query=get_latest_question(state))
        res = self.model.with_structured_output(QuestionType).invoke([SystemMessage(content=prompt)] + state['messages'])

        question_type = "CM" if "CM" in res.reason.upper() else "GENERAL"
        return {"question_type": question_type}
    