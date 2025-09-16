from typing import Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import  HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question

# class QuestionType(BaseModel):
#     CM: bool = Field(description="True if question is related to Construction Management (CM), False otherwise")
#     GENERAL: bool = Field(description="True if question is related to general conversation, False otherwise")

class QuestionTypeState(TypedDict):
    question: str
    question_type: str

class QuestionClassifier:
    def __init__(self, model: ChatOllama):
        self.model = model
        self.prompt = (
            "You are a Construction Management (CM) domain expert. Your task is to classify the incoming questions.\n"
            "Depending on your answer, question will be routed to the right team, so your task is crucial for our team.\n"
            "There are 2 possible question types:\n"
            "- CM - questions related to construction management.\n"
            "- GENERAL - general questions.\n"
            "Return in the output only one word (CM or GENERAL).\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(QuestionTypeState)
        g.add_node("cm_classifier", self.cm_classifier)

        g.add_edge(START, "cm_classifier")
        g.add_edge("cm_classifier", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_classifier(self, state: QuestionTypeState) -> QuestionTypeState:
        # question = get_latest_question(state)
        question_type = self._classify_question(state['question'][-1].content)
        return {"question_type": question_type}
    
    def _classify_question(self, question: str) -> str:
        messages = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=question)
        ]
        res = self.model.invoke(messages)
        return res.content
