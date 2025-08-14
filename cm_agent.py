import os
import json
import warnings
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv(override=True)
warnings.filterwarnings("ignore")

# LangChain / LangGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# SQL
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Qdrant
from qdrant_client import QdrantClient, models

# Optional search tool (kept but unused in graph)
from langchain_community.tools import BraveSearch

# ────────────────────────────────────────────────────────────────────────────────
# ENV
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# ────────────────────────────────────────────────────────────────────────────────
# LLMs & DB
llm = ChatOllama(model=os.getenv("LLM_MODEL"), temperature=0)
emb_model = OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text:latest"))
engine = create_engine(POSTGRES_URI)
db = SQLDatabase(engine=engine)

def get_sql_tools(db: SQLDatabase, llm: ChatOllama):
    """Create SQL tools for ReAct agent."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()

DB_TOOLS = get_sql_tools(db, llm)

# ────────────────────────────────────────────────────────────────────────────────
# Tools
@tool
def web_search(q: str) -> str:
    """Dummy web search via Brave (not used in graph)."""
    search_tool = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 3})
    res = search_tool.invoke({"query": q})
    return f"Search results for {q}: {res}"

@tool
def outline_tool(topic: str) -> List[str]:
    """Quick outline."""
    return [f"Intro to {topic}", "Key facts", "Implications", "Conclusion"]

@tool
def get_relavant_context(q: str) -> List[Dict[str, Any]]:
    """Retrieve top-5 relevant payloads from Qdrant for the user's question."""
    client = QdrantClient(url=QDRANT_URL)

    resp = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=emb_model.embed_query(q),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=5,
    )
    return [pt.payload for pt in resp.points]

# ────────────────────────────────────────────────────────────────────────────────
# Subgraphs

def build_retrieval_agent(model: ChatOllama):
    """ReAct agent that *must* call get_relavant_context and return it in state."""
    agent = create_react_agent(
        model,
        tools=[get_relavant_context],
        prompt=(
            "You are a data retrieval agent. Use the get_relavant_context tool to "
            "fetch the most relevant tables or snippets for the user's question."
        ),
    )

    class RTState(MessagesState):
        relavant_context: str  # JSON string for downstream prompts

    g = StateGraph(RTState)

    def retrieve_node(state: RTState):
        # Nudge the agent to call the tool explicitly
        msgs: List[BaseMessage] = state["messages"] + [
            HumanMessage(content=(
                "Call get_relavant_context with a concise query derived from our chat. "
                "Then return ONLY a compact JSON array of the top results."
            ))
        ]
        res = agent.invoke({"messages": msgs})
        out_msgs: List[BaseMessage] = res["messages"]
        # Try to capture the final assistant content as JSON string
        final_text = out_msgs[-1].content if out_msgs else "[]"
        # Ensure JSON serializable string
        try:
            _ = json.loads(final_text)
            ctx_json = final_text
        except Exception:
            # If the agent replied in prose, fallback to empty array
            ctx_json = "[]"
        return {"messages": out_msgs + [AIMessage(content="(Retrieval completed.)")],
                "relavant_context": ctx_json}

    g.add_node("retrieve", retrieve_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", END)
    return g.compile()

def build_db_query_agent(model: ChatOllama):
    """Agent that uses SQL tools. We inject table context via a SystemMessage each run."""
    system_message = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.

    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final Answer here

    Only use the following tables:
    {table_context}

    """
    # Create a base agent (prompt will be overridden per run by adding a SystemMessage)
    base_agent = create_react_agent(model, tools=DB_TOOLS)

    class DBState(MessagesState):
        table_context: str  # JSON string
        sql_results: str

    g = StateGraph(DBState)

    def database_node(state: DBState):
        rendered_system = system_message.format(dialect='postgresql', 
                                                top_k=5, 
                                                table_context=state.get("table_context", "[]"))

        msgs: List[BaseMessage] = [HumanMessage(content=rendered_system)] + state["messages"]
        res = base_agent.invoke({"messages": msgs})
        out_msgs: List[BaseMessage] = res["messages"]
        sql_results = out_msgs[-1].content if out_msgs else ""

        return {"messages": out_msgs, "sql_results": sql_results}

    g.add_node("db_query", database_node)
    g.add_edge(START, "db_query")
    g.add_edge("db_query", END)
    return g.compile()

def build_sql_evaluator_agent(model: ChatOllama):
    """Agent that critiques the previous SQL & answer.
    It reuses SQL tools to re-check if necessary."""
    agent = create_react_agent(
        model,
        tools=DB_TOOLS,
        prompt=(
            "You are a PostgreSQL Expert. Evaluate the proposed SQL and result.\n"
            "If the SQL/answer looks wrong, DO NOT fix the SQL and derive a 'Sorry, I can't do that. as Final Answer'"
        ),
    )

    class EvalState(MessagesState):
        sql_query_source: str  # text capturing 'Question / SQLQuery / SQLResult / Answer'
        draft_answer: str

    g = StateGraph(EvalState)

    def evaluator_node(state: EvalState):
        msgs: List[BaseMessage] = state["messages"] + [
            HumanMessage(content=(
                "If the SQL/answer looks wrong, DO NOT fix the SQL and derive a 'Sorry, I can't do that. as Final Answer'"
                "Evaluate the following block and reply with a short verdict" + state.get("sql_query_source", "")
            ))
        ]
        res = agent.invoke({"messages": msgs})
        out_msgs: List[BaseMessage] = res["messages"]
        draft = out_msgs[-1].content if out_msgs else ""
        return {"messages": out_msgs, "draft_answer": draft}

    g.add_node("query_evaluator", evaluator_node)
    g.add_edge(START, "query_evaluator")
    g.add_edge("query_evaluator", END)
    return g.compile()

def build_writting_answer(model: ChatOllama):
    """Agent that critiques the previous SQL & answer.
    It reuses SQL tools to re-check if necessary."""
    agent = create_react_agent(
        model,
        tools=DB_TOOLS,
    )

    class EvalState(MessagesState):
        query_source: str  # text capturing 'Question / SQLQuery / SQLResult / Answer'
        draft_answer: str

    g = StateGraph(EvalState)

    def evaluator_node(state: EvalState):
        msgs: List[BaseMessage] = state["messages"] + [
            HumanMessage(content=("Please answer the question based on the provided information with a short summary." + state.get("query_source", "")))
        ]
        res = agent.invoke({"messages": msgs})
        out_msgs: List[BaseMessage] = res["messages"]
        draft = out_msgs[-1].content if out_msgs else ""
        return {"messages": out_msgs, "draft_answer": draft}

    g.add_node("query_evaluator", evaluator_node)
    g.add_edge(START, "query_evaluator")
    g.add_edge("query_evaluator", END)
    return g.compile()


# ────────────────────────────────────────────────────────────────────────────────
# Top-level Supervisor
class TopState(MessagesState):
    relavant_context: str
    sql_query_source: str
    final_answer: str

def build_hierarchical_app(model: ChatOllama):
    retrieve_team = build_retrieval_agent(model)
    query_team = build_db_query_agent(model)
    eval_team = build_sql_evaluator_agent(model)

    g = StateGraph(TopState)

    def plan(state: TopState):
        msgs: List[BaseMessage] = state["messages"] + [
            AIMessage(content=(
                "Plan: retrieve relevant context → generate SQL & results → evaluate → answer."
            ))
        ]
        return {"messages": msgs}

    def call_retrieval(state: TopState):
        res = retrieve_team.invoke({"messages": state["messages"]})
        msgs = res["messages"]
        ctx = res.get("relavant_context", "[]")
        return {"messages": msgs, "relavant_context": ctx}


    def call_query(state: TopState):
        res = query_team.invoke({
            "messages": state["messages"],
            "table_context": state.get("relavant_context", "[]"),
        })
        msgs = res["messages"]
        sql_res = res.get("sql_results", "")
        return {"messages": msgs, "sql_query_source": sql_res}

    def call_query_evaluator(state: TopState):
        res = eval_team.invoke({
            "messages": state["messages"],
            "sql_query_source": state.get("sql_query_source", ""),
        })
        msgs = res["messages"]
        draft = res.get("draft_answer", "")
        return {"messages": msgs, "final_answer": draft}

    g.add_node("plan", plan)
    g.add_node("retrieval", call_retrieval)
    g.add_node("query", call_query)
    g.add_node("query_evaluator", call_query_evaluator)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieval")
    g.add_edge("retrieval", "query")
    g.add_edge("query", "query_evaluator")  # (FIX) Missing edge in original code
    g.add_edge("query_evaluator", END)

    # Optional memory/checkpointing if you plan to stream with resume support
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


# ────────────────────────────────────────────────────────────────────────────────
# Build app & export diagram
app = build_hierarchical_app(llm)

# Save Mermaid PNG correctly (API returns bytes; don't pass a path arg)
try:
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph diagram to graph.png")
except Exception as e:
    print(f"Could not render graph diagram: {e}")


if __name__ == "__main__":
    RUN_CONFIG = {"configurable": {"thread_id": "demo-thread-1"}}

    # Example streaming run
    for step in app.stream(
        {"messages": [{"role": "user", "content": "รายชื่อสมาชิกคนล่าสุด"
                    #    "What is first name and joined date of newest user?"
                       }]},
        stream_mode="values",
        config=RUN_CONFIG

    ):
        step["messages"][-1].pretty_print()
