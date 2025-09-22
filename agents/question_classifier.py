import json
import re
from typing import Literal
from pydantic import BaseModel, ValidationError
from langchain_ollama import ChatOllama
from langchain_core.messages import  SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question

class QuestionType(BaseModel):
    type: Literal['CM', 'GENERAL', None]

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
            "Use the provided CHAT HISTORY and the latest user message to determine the correct type.\n\n"
            "QUERY:\n{query}\n\n"
            "You must respond ONLY in the following strict JSON format:\n"
            "```json\n"
            "{{\n"  # double braces
            "  \"type\": \"CM\" or \"GENERAL\",\n"
            # "  \"reason\": \"brief explanation of why the query was classified as such\"\n"
            "}}\n"  # double braces
            "```\n"
            "Do not include any text outside the JSON.\n"
        )

    def build(self, checkpointer=None):
        g = StateGraph(QuestionTypeState)
        g.add_node("cm_classifier", self.cm_classifier)

        g.add_edge(START, "cm_classifier")
        g.add_edge("cm_classifier", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_classifier(self, state: QuestionTypeState) -> QuestionTypeState:
        prompt = self.prompt.format(query=get_latest_question(state))
        resp = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])

        question_type = QuestionType(type=self.parse_model_output(resp.content))
        if question_type.type is None:
            return {"question_type": "GENERAL"}
        return {"question_type": question_type.type}
    
    @staticmethod
    def parse_model_output(text: str) -> str:
        if not text:
            return None
        # find first JSON object occurrence
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if not match:
            return None
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            candidate = obj.get('type')
            if isinstance(candidate, str):
                return candidate
        except Exception:
            print("Failed to parse JSON")
            return None
        return None
    
