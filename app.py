from __future__ import annotations
import os
import uuid
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ----- your agent bits (unchanged) -----
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from agent_hub import *
from main import HierarchicalAgent
from model.state_model import UserContext

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")

# ----- FastAPI models -----
class ChatSession(BaseModel):
    question: str
    user_id: int
    company_id: int
    project_id: int
    thread_id: Optional[str] = None  # allow client to pin a thread explicitly

class ChatResponse(BaseModel):
    answer: str
    latency_ms: int
    request_id: str
    meta: Dict[str, Any] = Field(default_factory=dict)

# ----- App setup -----
app = FastAPI(title="Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_agent():
    graph = HierarchicalAgent()
    # Setting up the nodes
    graph.router = RouterTeams(
        model=ChatOllama(model="qwen3:1.7b", temperature=0.1)
    )
    graph.help_desk = ConversationTeams(
        model=ChatOllama(model="qwen3:1.7b", temperature=0.1)
    )
    graph.database = DatabaseTeams(
        model=ChatOllama(model="qwen3:1.7b", temperature=0.1), 
        db_uri=POSTGRES_URI
    )
    graph.research = ResearchTeams(
        model=ChatOllama(model="sqlcoder:7b", temperature=0.1), 
        qdrant_url=QDRANT_URL, 
        collection_name=QDRANT_COLLECTION_NAME, 
        embeded_model_nam=EMBEDED_MODEL_NAME
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