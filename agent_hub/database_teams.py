import enum
import re
from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

from model.state_model import DBState, SQLGenerator
from utils.get_latest_question import get_latest_question

# ----- Small helpers -----
_SELECT_REGEX = re.compile(r"^\s*select\b", re.IGNORECASE)
_LIMIT_REGEX = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)

def _ensure_limit(sql: str, default_limit: int) -> str:
    """Append LIMIT if not present; leaves existing LIMIT/OFFSET untouched."""
    if not _LIMIT_REGEX.search(sql):
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql} LIMIT {int(default_limit)}"
    return sql

def _is_select(sql: str) -> bool:
    return bool(_SELECT_REGEX.match(sql or ""))

class TOOLTYPE(enum.Enum):
    RFI = 0
    Submittal=  1
    Inspection = 2

class DatabaseTeams:
    def __init__(self,
        model: ChatOllama,
        db_uri: str,
        dialect: str = "postgresql",
        default_top_k: int = 5,
        default_limit: int = 10
    ):  
        self.model = model
        structured_model = ChatOllama(model="qwen3:0.6b", temperature=0)
        self.structured_model = structured_model.with_structured_output(SQLGenerator)

        self.dialect = dialect
        self.default_top_k = int(default_top_k)
        self.default_limit = int(default_limit)

        # DB + toolkit
        self._engine = create_engine(db_uri)
        self.db = SQLDatabase(engine=self._engine)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        self.db_tools = self.toolkit.get_tools()
        self.db_tool_map = {t.name: t for t in self.db_tools}

        # Validate required tools up-front (fail fast)
        missing = [k for k in ("sql_db_query", "sql_db_query_checker") if k not in self.db_tool_map]
        if missing:
            raise RuntimeError(f"Required SQL tools not available: {missing}")
        
        # Static prompt template (lean + explicit)
        self._prompt_tmpl = (
            "You are a {dialect} expert. Given the question, context, and table relationships, "
            "produce a syntactically correct {dialect} SQL query that answers the question.\n\n"

            "INTENT RULES:\n"
            "- If the question is about RFI, Submittal, or Inspection, treat it as a document-centric query using table aliases:\n"
            "  document_document AS d, project_project AS p, company_company AS c.\n"
            "- For RFI queries, you can use document_document (d) only.\n"
            "- For Submittal queries, you MUST JOIN document_submittal AS s ON s.document_id = d.id.\n"
            "- For Inspection queries, you MUST JOIN document_inspection AS i ON i.document_id = d.id.\n"
            "- Always JOIN project_project (p) and company_company (c):\n"
            "    JOIN project_project AS p ON p.id = d.project_id\n"
            "    JOIN company_company AS c ON c.id = p.company_id\n"
            "- The core tables available for these tools are: company_company, project_project, document_document, auth_user "
            "(join auth_user only if explicitly required by the question, e.g., filtering by creator/reviewer).\n"
            "- If question implies recency (e.g., 'latest', 'newest', 'most recent'), order by d.created_at DESC.\n"
            "- If question implies oldest, order by d.created_at ASC.\n"
            "- Always user WHERE clauses d.deleted IS NULL.\n\n"

            "FILTER RULES (MANDATORY for RFI/Submittal/Inspection queries):\n"
            "- Always add this WHERE clause:\n"
            "    WHERE d.type = {tool_type}\n"
            "      AND d.project_id = {project_id}\n"
            "      AND p.company_id = {company_id}\n"
            "      AND d.deleted IS NULL\n\n"

            "SELECTION & STYLE RULES:\n"
            "- Unless a specific number of rows is requested, limit to {top_k} rows.\n"
            "- Use only columns/tables shown below; ensure column-table correctness.\n"
            "- If the user does not specify fields, default to: d.code, d.title, d.created_at.\n"
            "- If the question implies recency (e.g., 'latest', 'newest', 'most recent'), order by d.created_at DESC.\n"
            "- Use date('now') to refer to the current date if the question involves 'today'.\n"
            "- Prefer DISTINCT when joining child tables (e.g., submittal/inspection) to avoid duplicate rows.\n\n"

            "FEW-SHOT EXAMPLES:\n"
            "Q: What are the latest 10 RFIs?\n"
            "SELECT d.code, d.title, d.created_at FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 0\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1\n"
            "ORDER BY d.created_at DESC LIMIT 10;\n"

            "Q: How many RFIs are there?\n"
            "SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 0\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1;\n\n"

            "Q: Get the process status of the latest RFI\n"
            "SELECT d.process AS FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 0\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1\n"
            "ORDER BY d.created_at DESC LIMIT 1;\n"
            
            "Q: How many RFIs are in process?\n"
            "SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 0\n"
            "AND d.process = 1\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1;\n\n"
            "AND d.status = 2 --(User ask 'In Process' so, this status is in 'status fields')\n"

            "Q: How many RFIs are closed?\n"
            "SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 0\n"
            "AND d.process = 2\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1;\n\n"
            "AND d.process = 4\n\n"
# --------------------------------------------------------------------------------------------------------------------------------
            "Q: What are the latest 10 Submittals?\n"
            "SELECT d.code, d.title, d.created_at FROM document_document AS d\n"
            "JOIN document_submittal AS s ON s.document_id = d.id\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 1\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1\n"
            "ORDER BY d.created_at DESC LIMIT 10;\n"

            "Q: How many Submittals are there?\n"
            "SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN document_submittal AS s ON s.document_id = d.id\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 1\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1;\n"

            "Q: How many submittals are rejected? in this month?\n"
            "SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN document_submittal AS s ON s.document_id = d.id\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 1\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1;\n"
            "AND d.status = 5\n"
            "AND date_trunc('month', d.created_at) = date_trunc('month', CURRENT_DATE);\n"

            "Q: I'm barely able to remember the document code; it ends with 61. I'd like to know the status and code of the submittals."
            "SELECT d.code, d.status FROM document_document AS d\n"
            "JOIN document_submittal AS s ON s.document_id = d.id\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 1\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1;\n"
            "AND d.code LIKE '%61'\n"
            "ORDER BY d.created_at DESC;\n\n"
# --------------------------------------------------------------------------------------------------------------------------------
            "Q: What are the latest 10 Inspections?\n"
            "SELECT i.* FROM document_document AS d\n"
            "JOIN document_inspection AS i ON i.document_id = d.id\n"
            "JOIN document_submittal AS s ON s.document_id = d.id\n"
            "JOIN project_project AS p ON p.id = d.project_id JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL\n"
            "AND d.type = 2\n"
            "AND d.project_id = 1\n"
            "AND p.company_id = 1\n"
            "ORDER BY d.created_at DESC LIMIT 10;\n\n"
# --------------------------------------------------------------------------------------------------------------------------------

            "Only use the following tables:\n{tables}\n\n"
            "Context:\n{tool_context}\n\n"
            "Question: {question}\n\n"

            "Generate SQL Query in PLAIN TEXT:\n"
        )
        
    def build(self):
        g = StateGraph(DBState)
        g.add_node("generate_node", self.generate_sql)
        g.add_node("execute_node", self.execute_sql)

        g.add_edge(START, "generate_node")
        g.add_edge("generate_node", "execute_node")
        g.add_edge("execute_node", END)
        return g.compile()
    
    def generate_sql(self, state: DBState) -> Dict[str, Any]:
        question = get_latest_question(state)
        project_id = state["user"].project_id
        company_id = state["user"].company_id
        tool_type = TOOLTYPE[f"{state['tool']}"].value
        tool_context = str(state.get("tool_selected_reason", "")).strip() or "None."
        table_contexts = "\n".join([f"Table:{tbl['name']}\n Fields: {tbl['fields']}" for tbl in state["relevant_tables"]])
        
        rendered = self._prompt_tmpl.format(
            dialect=self.dialect,
            top_k=self.default_top_k,
            tables=table_contexts,
            tool_context=tool_context,
            question=question[-1].content,
            tool_type=tool_type,
            project_id=project_id,
            company_id=company_id,
        )

        try:
            # Generate SQL
            res = self.model.invoke([SystemMessage(content=rendered)])
            # Structured output
            #BUG: It failed to structured output
            response = self.structured_model.invoke(res.content)
            generated_sql = (response.query or "").strip()

        except Exception as e:
            # Fail closed with a clear message in state
            return {"generated_sql": f"Failed to generate SQL: {e}"}

        return {"generated_sql": generated_sql}
    
    def execute_sql(self, state: DBState) -> Dict[str, Any]:
        sql = (state.get("generated_sql") or "").strip()
        if not sql:
            return {
                "messages": [AIMessage(content="No SQL to execute.")]
            }

        # Enforce SELECT-only
        if not _is_select(sql):
            return {
                "messages": [AIMessage(content="Refusing to execute non-SELECT SQL.")]
            }

        # Ensure LIMIT if absent
        sql = _ensure_limit(sql, self.default_limit)

        try:
            question = get_latest_question(state)
            table_contexts = "\n".join([f"Table:{tbl['name']}\n Fields: {tbl['fields']}" for tbl in state["relevant_tables"]])

            prompt=(
                "You are a SQL executor. You will be given a SQL query, a question and a table context.\n"
                "SQL query: {sql}\n"
                "Question: {question}\n"
                "Table context: {table_contexts}\n"
                "Answer: "
            ).format(
                sql=sql,
                question=question[-1].content,
                table_contexts=table_contexts
            )
            # Execute tools
            runner = self.db_tool_map["sql_db_query"]

            # Execute LLM
            model = ChatOllama(model="llama3.2", temperature=0)

            # React Agent
            exec_agent = create_react_agent(model, tools=[runner], prompt=prompt)
            res = exec_agent.invoke({"messages": [HumanMessage(content=question[-1].content)]})
            res_txt = res['messages'][-1].content
            
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"SQL execution crashed: {e}")]
            }

        return {
            "messages": [AIMessage(content=res_txt)],
        }