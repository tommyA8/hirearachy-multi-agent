import os
import json
from typing import List
import warnings
from dotenv import load_dotenv
load_dotenv(override=True)
warnings.filterwarnings("ignore")

from langgraph.graph import MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

## SQL
from sqlalchemy import create_engine, MetaData, Table, select, bindparam, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# utils
from utils.qdrant_helper import QdrantVector
from model.agent_model import *

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")

engine = create_engine(os.getenv("POSTGRES_URI"))
metadata = MetaData()
db = SQLDatabase(engine=engine)

qdrant = QdrantVector(qdrant_url=QDRANT_URL, collection_name=QDRANT_COLLECTION_NAME, model_name=EMBEDED_MODEL_NAME)

# Define states
class RouterState(MessagesState):
    tool: Tools
    selected_reason: str

class PermissionsState(RouterState):
    user: UserContext
    permission: str

class RetrieveState(PermissionsState):
    relavant_context: str

class DBState(RetrieveState):
    generated_sql: str
    evaluated_sql: str
    sql_results: List

class MainState(DBState):
    final_answer: str

# Build sub agents/teams
def help_desk_team(model: ChatOllama):
    def help_desk_node(state: MainState):
        
        try:
            if state["permission"] == "valid":
                prompt = """
                Your name is ChatCM.
                You are a polite and helpful help desk of a construction management (CM) system.
                
                You are given the SQL results and a question.
                Your job is to answer with a short answer.

                REMEMBER DO NOT Explain why error occurs.

                question: {question}
                """

                prompt += f"SQL Results: \n{state["sql_results"]}"

            else:
                prompt = """
                Your name is ChatCM.
                You are a polite and helpful help desk of a construction management (CM) system.
                Your job is to answer with a short answer to a given questions.

                REMEMBER
                User does not have permission to ask question related to this feature.

                You MUST NOT answer the question and give a reason to the user.
                question: {question}
                """
   
        except:
            prompt = """
            Your name is ChatCM.
            You are a polite and helpful help desk of a construction management (CM) system.
            Your job is to answer with a short answer to a given questions following instructions.
            instructions:
            - Construction Management (CM) questions
            - Greetings
            - Introduce yourself
            
            DO NOT ANSWER questions that are not related to Construction Management (CM).
            question: {question}
            """

        response = model.invoke(prompt.format(question=state["messages"][0].content))
        ai_msg = AIMessage(content=response.content.split("</think>\n")[-1])
        return {
            "messages": [ai_msg],
        }
    
    g = StateGraph(MainState)
    g.add_node("help_desk", help_desk_node)
    g.add_edge(START, "help_desk")
    g.add_edge("help_desk", END)
    return g.compile()

def router_team(model: ChatOllama):
    prompt_router = """
    You are a Costruction Management expert. You know End to End Construction Management process.
    Analyze the user query below and determine its Available Tools with deeply reason.

    Available tools (Choose one):
        - Document: For questions about user's documents related to the project that they uploaded to the system. It's not related to attachmented submittal files or other.
        - Submittal: For questions about construction submittals.
        - RFI: For questions about construction requests for information (RFI).
        - Inspection: For questions about construction inspections.
        - Work Order: For questions about construction work orders.
        - Unknown: If the user's query is not related to any of the above features. Try to answer the user's question as best you can and take the user to the next step of the process.

    Return a JSON object with fields: question, tool, selected_reason.

    Query: {question}
    """
        
    agent = model.with_structured_output(RoutingDecision)

    def tool_router_node(state: RouterState):
        question = state["messages"][-1].content

        decision = agent.invoke(prompt_router.format(question=question))

        ai_msg = AIMessage(
            content=f"LLM Router decided: {decision.tool.value}"
        )
        return {
            "messages": [ai_msg],
            "tool": decision.tool,
            "selected_reason": decision.selected_reason,
        }
    
    g = StateGraph(RouterState)
    g.add_node("tool_router", tool_router_node)
    g.add_edge(START, "tool_router")
    g.add_edge("tool_router", END)
    return g.compile()

def retrieval_team():
    def retrieve_node(state: RetrieveState):
        payload = {
            "question": state["messages"][0].content,
            "tool": state["tool"].value,
            "selected_reason": state["selected_reason"]
        }
        text_to_embed = json.dumps(payload, ensure_ascii=False)

        relevant_cntx = qdrant.get_relavant_context(text_to_embed, limit=1)

        for table in relevant_cntx[0]['related_tables']:
            relationship = qdrant.filter_payload(key="table", value=table)
            relevant_cntx.append(relationship[0])

        sys_msg = SystemMessage(
            content="Retrieved relevant " 
            + state["tool"].value 
            + " tables: " 
            + ", ".join([tbl['table'] for tbl in relevant_cntx])
        )

        return {"messages": [sys_msg],
                "relavant_context": relevant_cntx}
    
    g = StateGraph(RetrieveState)
    g.add_node("retrieve", retrieve_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", END)
    return g.compile()

def database_team(model: ChatOllama):
    gen_agent = model.with_structured_output(SQLGenerator)
    eval_agent = model.with_structured_output(SQLEvaluator)
    
    def generate_sql(state: DBState):
        system_message = """
        You are a careful, read-only data analyst agent that interacts with a SQL database.

        GOAL
        - Given the user's question, produce a syntactically correct {dialect} query, execute it, and summarize the result succinctly.
        - Default to at most {top_k} rows unless the user asks for a different limit.

        SCOPE & SAFETY
        - READ-ONLY ONLY. Absolutely NO DML/DDL: INSERT, UPDATE, DELETE, MERGE, DROP, TRUNCATE, CREATE, ALTER, GRANT, REVOKE.
        - Use only the tables and columns provided in `TABLE CONTEXT`. If something is missing, say so rather than guessing.
        - Prefer parameterized predicates for sensitive values (e.g., :company_id, :project_id).
        - Avoid full scans when possible. Use WHERE, appropriate ORDER BY, and LIMIT.
        - Never SELECT *; only select columns needed to answer the question.
        - Assume timezone = {{"Asia/Bangkok"}} unless the question specifies otherwise. Be explicit about which timestamp you consider authoritative (e.g., created_at vs updated_at).

        DISAMBIGUATION HEURISTICS
        - "latest", "newest": order by the most relevant timestamp (prefer business-complete time > created_at > updated_at), DESC, LIMIT {top_k}.
        - "count", "how many": return an aggregate (COUNT(*)) and any requested breakdowns (GROUP BY).
        - Ranges like "last 7 days" are relative to the current time in the assumed timezone.
        - If the wording remains too ambiguous to write a safe query, ask one short clarifying question before proceeding.

        QUERY STYLE
        - Use fully-qualified table names if schemas are present.
        - Use clear column aliases and CTEs for readability when helpful.
        - When joining, use explicit JOIN ... ON with keys implied by names (e.g., *_id) or described in `TABLE CONTEXT`.
        - For text search, use functions supported by {dialect} only if listed in `TABLE CONTEXT` (avoid vendor-specific features unless certain).
        - When returning "top" records, specify deterministic ORDER BY.

        VALIDATION
        - Double-check syntax and table/column names against `TABLE CONTEXT` before execution.
        - If execution fails, correct the query and retry once with a brief explanation of the fix.

        OUTPUT FORMAT
        Return your work in exactly this structure:

        REMEMBER When generate SQLQuery, You have to validate the avaliable columns and tables from `TABLE CONTEXT`

        Question: <copy the user question>
        Assumptions: <only if you had to assume/clarify anything; keep to one short sentence>
        SQLQuery:
        <the final SQL to run (read-only, parameterized where appropriate)>
        SQLResult: <result rows or a compact aggregate preview (max {top_k} rows)>
        Answer: <one-sentence answer in plain language, citing key numbers and timeframes>

        USER CONTEXT
        Company ID: {company_id}
        Project ID: {project_id}

        TABLE CONTEXT 
        {table_context}

        Question:
        {question}
        """
        
        rendered_system = system_message.format(dialect='postgresql', 
                                                top_k=1,
                                                company_id=state["user"].company_id,
                                                project_id=state["user"].project_id,
                                                table_context=state['relavant_context'],
                                                question=state["messages"][0].content)
        res = gen_agent.invoke(rendered_system)

        ai_msg = AIMessage(content="SQL Query Generated")

        return {"messages": [ai_msg], 
                "generated_sql": res.query}
    
    def evaluate_sql(state: DBState):
        system_message = """
        You are given a SQL query and a list of relevant tables and their relationships.
        You MUST TO evaluate the provided SQL query, check for existence of columns and tables and correctness of the query.
        
        Finally, Generate a correct, clear, and readable SQL query.

        SQLQuery:
        {query}

        Relevant tables:
        {relevant_tables}

        Table relationships:
        {relationships}
        """

        rendered_system = system_message.format(
            query=state["generated_sql"],
            relevant_tables=state["relavant_context"],
            relationships=state["relavant_context"][0]['relationships']
        )

        res = eval_agent.invoke(rendered_system)

        ai_msg = AIMessage(content="SQL Query Evaluated:\n\n" + res.query)

        return {"messages": [ai_msg], 
                "evaluated_sql": res.query}
    
    def query(state: DBState):
        sql = state.get("evaluated_sql").strip().rstrip(";")
        if not sql.lower().startswith("select"):
            print(f"Refusing to execute non-SELECT SQL: {sql}")
            raise ValueError("Refusing to execute non-SELECT SQL.")
        if " limit " not in sql.lower():
            sql += " LIMIT 50"

        # Build toolkit (ideally do this once in __init__)
        llm = ChatOllama(model="qwen3:4b", temperature=0.1)
        # db = SQLDatabase.from_engine(self.engine)  # or SQLDatabase.from_uri("postgresql+psycopg://...")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        tools_list = toolkit.get_tools()
        tool_map = {t.name: t for t in tools_list}

        # Resolve the query tool across versions
        def get_query_tool(tool_map):
            preferred = ["query_sql_db", "sql_db_query", "sql_db_query_tool"]
            for name in preferred:
                if name in tool_map:
                    return tool_map[name]
            # fallback: fuzzy match
            for name in tool_map:
                if "query" in name and "sql_db" in name:
                    return tool_map[name]
            raise KeyError(f"No SQL query tool found. Available: {list(tool_map)}")

        query_tool = get_query_tool(tool_map)

        # Run the tool directly (no agent needed)
        tool_out = query_tool.invoke({"query": sql})  # returns a string with rows/preview

        ai_msg = AIMessage(content="SQL executed via SQLDatabaseToolkit.")
        return {"messages": [ai_msg], 
                "sql_results": tool_out}
        
    g = StateGraph(DBState)
    g.add_node("generate_node", generate_sql)
    g.add_node("evaluate_node", evaluate_sql)
    g.add_node("query_node", query)

    g.add_edge(START, "generate_node")
    g.add_edge("generate_node", "evaluate_node")
    g.add_edge("evaluate_node", "query_node")
    g.add_edge("query_node", END)
    return g.compile()