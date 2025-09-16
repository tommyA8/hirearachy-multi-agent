from __future__ import annotations
import os
import uuid
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from agents import *
from workflow.chat_cm import ChatCM, UserContext

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY")

# ----- FastAPI models -----
class ChatSession(UserContext):
    question: str
    thread_id: Optional[str] = None  # allow client to pin a thread explicitly

class ChatResponse(BaseModel):
    answer: str
    latency_ms: int
    request_id: str
    meta: Dict[str, Any] = Field(default_factory=dict)

# ----- App setup -----
app = FastAPI(title="ChatCM - API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_agent():
    graph = ChatCM()
    graph.question_classifier = QuestionClassifier(
        # model= ChatOllama(model="qwen3:0.6b", temperature=0.1),
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
       )
    graph.general_assistant_team = GeneralAssistant(
    #    model= ChatOllama(model="qwen3:0.6b", temperature=0.1),
       model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0.2, api_key=NVIDIA_LLM_API_KEY),
       )
    graph.supervisor_team = CMSupervisor(
        # model= ChatOllama(model="qwen3:0.6b", temperature=0.1),
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
    )
    graph.rfi_team = RFIAgent(
        # model= ChatOllama(model="qwen3:0.6b", temperature=0.1),
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        yaml_path="docs/cm_db_knowledge.yaml",
        db_uri=POSTGRES_URI
    )
    # Building the agent
    memory = MemorySaver()
    return graph.build(checkpointer=memory)

def make_thread_id(s: ChatSession) -> str:
    # Deterministic per user/company/project unless overridden
    return s.thread_id if s.thread_id else str(uuid.uuid4())

agent = init_agent()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/v1/agent/chat", response_model=ChatResponse)
def chat(session: ChatSession, request: Request):
    if not session.question or not session.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    thread_id = make_thread_id(session)

    start = time.perf_counter()
    try:
        out = agent.invoke(
            {
                "messages": [HumanMessage(content=session.question)],
                "user": UserContext(
                    user_id=session.user_id,
                    company_id=session.company_id,
                    project_id=session.project_id,
                    tool_permissions=session.tool_permissions
                ),
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        # You can surface graph traces here if you store them
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    latency_ms = int((time.perf_counter() - start) * 1000)
    # LangGraph prebuilt agents usually return a dict with "messages"
    try:
        answer = out["messages"][-1].content
    except Exception:
        answer = str(out)

    return ChatResponse(
        answer=answer,
        latency_ms=latency_ms,
        request_id=request_id,
        meta={"thread_id": thread_id},
    )

# Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000