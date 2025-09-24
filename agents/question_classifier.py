import json
import re
from typing import Literal, Optional
from pydantic import BaseModel, ValidationError
from langchain_ollama import ChatOllama
from langchain_core.messages import  SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from utils.get_latest_question import get_latest_question

class QuestionType(BaseModel):
    type: Optional[Literal['CM', 'GENERAL', 'NEED_MORE_CNTX']]

class QuestionTypeState(MessagesState):
    question_type: str

class QuestionClassifier:
    CM = "CM"
    GENERAL = "GENERAL"
    NEED_MORE_CNTX = "NEED_MORE_CNTX"
    VALID_TYPES = {CM, GENERAL, NEED_MORE_CNTX}

    def __init__(self, model: ChatOllama):
        self.model = model
        self.prompt = (
            "You are a domain expert in Construction Management (CM).\n"
            "Your task is to classify the user's latest query based on its intent.\n\n"
            "Classification Types (choose exactly one):\n"
            "- CM: query directly related to construction management topics such as projects, documents, schedules, RFIs, submittals, and related workflows.\n"
            "- GENERAL: All other questions not related to construction management.\n\n"
            "- NEED_MORE_CNTX: The user's latest query seems ambiguous or lacks sufficient context to classify.\n\n"
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
        g.add_node("general_assistant", self.general_assistant)
        g.add_node("ask_for_more_context_node", self.ask_for_more_context)

        g.add_edge(START, "cm_classifier")
        g.add_conditional_edges(
            "cm_classifier",
            lambda s: s['question_type'],
            {
                self.CM: END,
                self.GENERAL: "general_assistant",
                self.NEED_MORE_CNTX: "ask_for_more_context_node",
            }
        )
        g.add_edge("general_assistant", END)
        g.add_edge("ask_for_more_context_node", END)
        return g.compile(checkpointer) if checkpointer is not None else g.compile()

    def cm_classifier(self, state: QuestionTypeState) -> QuestionTypeState:
        prompt = self.prompt.format(query=get_latest_question(state))

        try:
            resp = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
            question_type = QuestionType(type=self.parse_model_output(resp.content, self.VALID_TYPES))
            chosen = getattr(question_type, "type", None)
        except ValidationError:
            chosen = None

        if chosen not in self.VALID_TYPES:
            chosen = self.NEED_MORE_CNTX

        return {"question_type": chosen}
    
    def general_assistant(self, state: QuestionTypeState) -> QuestionTypeState:
        return {
            "messages":  AIMessage(content="I'm sorry, I can only provide assistance with Construction Management topics such as RFIs, submittals, and related workflows."), 
            "question_type": self.GENERAL
        }
    
    def ask_for_more_context(self, state: QuestionTypeState) -> QuestionTypeState:
        prompt = (
            "The user's latest query seems ambiguous or lacks sufficient context to classify.\n"
            "Please ask a clarifying question to gather more details about their intent.\n\n"
            "LATEST QUERY:\n{query}\n\n"
            "Your clarifying question should be concise and directly related to understanding whether the query is about construction management (CM) or general topics.\n"
        ).format(query=get_latest_question(state))
        
        resp = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
                
        return {"messages": resp.content, "question_type": self.NEED_MORE_CNTX}
    
    @staticmethod
    def parse_model_output(text: str, allowed: set[str]) -> str | None:
        if not text:
            return None

        text = text.strip()

        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        upper_clean = text.upper()
        if upper_clean in allowed:
            return upper_clean

        candidates = [m.group(0) for m in re.finditer(r"\{[^{}]*\}" , text, flags=re.DOTALL)]

        if not candidates:
            broad = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
            candidates.extend(broad[:3])  # limit attempts

        def try_parse_json(snippet: str):
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                simple = re.sub(r"'", '"', snippet)
                try:
                    return json.loads(simple)
                except Exception:
                    return None
            except Exception:
                return None

        for snippet in candidates:
            if 'type' not in snippet:
                continue
            obj = try_parse_json(snippet)
            if not isinstance(obj, dict):
                continue
            val = obj.get('type') or obj.get('Type') or obj.get('TYPE')
            if isinstance(val, str):
                val_up = val.strip().upper()
                if val_up in allowed:
                    return val_up

        # 5. Heuristic: search for "type": <label>
        m = re.search(r'"?type"?\s*[:=]\s*"?(CM|GENERAL)"?', text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 6. Final fallback: look for standalone allowed tokens
        token_match = re.search(r"\b(CM|GENERAL)\b", text, flags=re.IGNORECASE)
        if token_match:
            return token_match.group(1).upper()

        return None
    
