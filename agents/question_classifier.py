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
    def parse_model_output(text: str) -> str | None:
        """Extract the classification type (CM|GENERAL) from an LLM response.

        The model SHOULD return a JSON block like:
            {"type": "CM"}

        However, in practice the response may contain:
        - Extra prose before/after JSON
        - Markdown code fences ```json ... ```
        - Multiple JSON objects
        - Minor JSON issues like single quotes
        - A bare label like CM or GENERAL

        This function attempts to robustly extract and normalize the value.
        Returns the upper‑cased label (CM|GENERAL) or None if not found.
        """
        if not text:
            return None

        allowed = {"CM", "GENERAL"}

        # 1. Strip leading/trailing whitespace
        text = text.strip()

        # 2. Remove surrounding markdown code fences if present
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        # 3. Quick path: if response is just the label
        upper_clean = text.upper()
        if upper_clean in allowed:
            return upper_clean

        # 4. Find all lightweight JSON object snippets and try to parse those containing 'type'
        # Use a non-greedy brace match; this is imperfect but usually sufficient for small structured outputs
        candidates = [m.group(0) for m in re.finditer(r"\{[^{}]*\}" , text, flags=re.DOTALL)]

        # Fallback: also include larger matches if nothing small found (first balanced-ish block)
        if not candidates:
            broad = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
            candidates.extend(broad[:3])  # limit attempts

        def try_parse_json(snippet: str):
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                # Attempt a minimal repair: replace single quotes with double IF it looks like simple JSON
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
    
