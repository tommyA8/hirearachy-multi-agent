from typing import Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import  HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question

class QuestionType(BaseModel):
    CM: bool = Field(description="True if question is related to Construction Management (CM), False otherwise")
    GENERAL: bool = Field(description="True if question is related to general conversation, False otherwise")

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
        g.add_node("is_cm_related_node", self.is_cm_related)

        g.add_edge(START, "is_cm_related_node")
        g.add_edge("is_cm_related_node", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def is_cm_related(self, state: QuestionTypeState) -> Dict[str, str]:
        # question = get_latest_question(state)
        question_type = self._classify_question(state['question'][-1].content)
        return {"question_type": question_type}
    
    def _classify_question(self, question: str) -> str:
        messages = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=question)
        ]
        structured_llm = self.model.with_structured_output(QuestionType)
        res = structured_llm.invoke(messages)
        if res.CM:
            return "CM"
        else:
            return "GENERAL"
