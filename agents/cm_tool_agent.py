import yaml
import re
import json
import ast
from abc import ABC
from typing import Dict, List, Optional
from sqlalchemy import create_engine

from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit

from utils import get_latest_question, QdrantVector
from constants.constants import *
from model.user import UserContext

class SQLState(MessagesState):
    """State passed between SQL tool agent nodes.

    Fields:
        generated_sql: Raw SQL text produced by LLM (expected single SELECT).
        results: Text serialization of execution result OR error message.
    messages: Inherited conversation message list from MessagesState.
    """
    user: UserContext
    generated_sql: str
    results: str

class BaseToolAgent(ABC):
    def __init__(self, 
                 model: ChatOllama,
                 db_uri: str,
                 db_docs_path: str,
                 sql_prompt: str = None,
                 dialect: str = "postgresql",
                 default_top_k: int = 3,
                 default_limit: int = 5,
                 default_tables: Optional[List[str]] = None):
        self.model = model
        self.db_docs_path = db_docs_path
        self.dialect = dialect
        self.db_toolkits = self.get_db_toolkit(db_uri, model)

        self.default_top_k = int(default_top_k)
        self.default_limit = int(default_limit)
        self.default_tables = default_tables or ['document_document', "company_company", "project_project"]
        self.default_table_info = self.get_db_info(self.db_docs_path, self.default_tables)

        self._sql_prompt: str = sql_prompt
        self._answer_prompt = (
            "You are a friendly and professional Help Desk Assistant for a Construction Management (CM) system.\n\n"
            "You will be given:\n"
            "  1. A natural-language QUESTION.\n"
            "  2. The executed SQL QUERY.\n"
            "  3. TABLE INFORMATION describing the schema.\n"
            "  4. The RESULT of the SQL query.\n\n"
            "Your task: read the SQL and RESULT, understand the context, and produce a clear, concise, factual answer to the QUESTION in markdown tabular format.\n\n"
            "SQL QUERY: {sql}\n\n"
            "enums:\n"
            "- type (Document Code): 0 = RFI, 1 = Submittal, 2 = Inspection\n"
            "- process (Process Status): 0 = Draft, 1 = Review, 2 = Open, 3 = Waiting for Manager, 4 = Closed\n"
            "- status (Task Decision): -1 = Invalid, 0 = Pending, 1 = In Process, 2 = Approved, 3 = Approved as Note, 4 = Rejected, 5 = Rejected as Note, 6 = Deleted\n\n"
            "RESULT: {result}\n\n"
            "STRICT FORMAT RULES:\n"
            "- Never mention or reveal SQL queries, database structures, or technical details to the user.\n"
            "- Respond in natural, plain English with a friendly, approachable tone.\n"
            "- If RESULT is empty or null, politely say you’re unable to find the requested information.\n"
            "- If the QUESTION asks for a list or table, use a Markdown table or a numbered list for clarity.\n"
            "- Do not repeat or refer to the SQL query text.\n"
            "- Do not hallucinate columns or values; only use what is present in the RESULT.\n"
            "- Summarize multiple rows in a helpful way if needed.\n\n"
            "QUESTION:\n{question}\n\n"
            "FINAL ANSWER:"
        )

    @property
    def sql_prompt(self):
        return self._sql_prompt
    
    @sql_prompt.setter
    def sql_prompt(self, prompt):
        self._sql_prompt = prompt

    @property
    def answer_prompt(self):
        return self._answer_prompt
    
    @answer_prompt.setter
    def answer_prompt(self, prompt):
        self._answer_prompt = prompt

    @staticmethod
    def get_db_toolkit(db_uri, model: ChatOllama) -> Dict[str, BaseTool]:
        engine = create_engine(db_uri)
        db = SQLDatabase(engine=engine)

        toolkit = SQLDatabaseToolkit(db=db, llm=model)
        db_tools = toolkit.get_tools()
        return {t.name: t for t in db_tools}

    @staticmethod
    def get_db_info(yaml_path: str, table_names: List[str]) -> str:
        """Extract structured documentation for the requested tables.

        Builds a lookup dictionary first for efficiency, then assembles a
        concatenated textual description (used inside prompts).
        """
        with open(yaml_path, "r") as f:
            db_docs = yaml.safe_load(f)

        tables = db_docs.get('database_docs', {}).get('tables', {})
        index = {}
        for tbl in tables.values():
            name = tbl.get('name')
            if name:
                index[name] = tbl

        parts: List[str] = []
        for table_name in table_names:
            tbl = index.get(table_name)
            if not tbl:
                continue
            desc = tbl.get('description', '')
            fields = tbl.get('fields', [])
            relations = tbl.get('relations', [])
            enums = tbl.get('enums', [])
            if enums:
                enum_type = [e.get('type') for e in enums if 'type' in e]
                enum_process = [e.get('process') for e in enums if 'process' in e]
                enum_status = [e.get('status') for e in enums if 'status' in e]
                enums = ""
                if enum_type:
                    enums += f"\n  - type: {enum_type[0]}"
                if enum_process:
                    enums += f"\n  - process: {enum_process[0]}"
                if enum_status:
                    enums += f"\n  - status: {enum_status[0]}"

            fields_str = "".join(
                f"{k}: {v}\n" for f in fields for k, v in f.items()
            )
            relations_str = "\n".join(relations)
            parts.append(
                f"Table: {table_name}\nDescription: {desc}\nFields:\n{fields_str}\nRelations:\n{relations_str}\n\nEnums:{enums}\n\n"
            )
        return "".join(parts)

    def build(self) -> CompiledStateGraph:
        g = StateGraph(SQLState)
        g.add_node("generate_sql_node", self.generate_sql_query)
        g.add_node("execute_sql_node", self.execute_query)
        g.add_node("generate_answer_node", self.generate_answer)

        g.add_edge(START, "generate_sql_node")
        g.add_edge("generate_sql_node", "execute_sql_node")
        g.add_edge("execute_sql_node", "generate_answer_node")
        g.add_edge("generate_answer_node", END)

        return g.compile()

    def generate_sql_query(self, state: SQLState) -> SQLState:
        """Generate a single read-only SELECT statement answering the latest question.

        Any failure is recorded verbosely in generated_sql for later explanation.
        """
        if self._sql_prompt is None:
            return {"generated_sql": "ERROR: SQL prompt not configured."}

        base_prompt = self._sql_prompt.format(
            dialect=self.dialect,
            top_k=self.default_top_k,
            question=get_latest_question(state),
            table_info=self.default_table_info,
            project_id=state['user'].project_id,
            company_id=state["user"].company_id
        )
        try:
            res = self.model.invoke([SystemMessage(content=base_prompt)] + state['messages'])
            raw_text = getattr(res, 'content', '') or ''
        except Exception as e:
            return {"generated_sql": f"ERROR: Failed to generate SQL: {e}"}

        # Attempt JSON parse first
        sql = self._parse_json_sql(raw_text)
        if not sql:
            # Fallback: heuristic extraction
            sql = self._extract_sql(raw_text)

        # check LIMIT presence and not too high; append default if missing
        sql = self._check_limit(sql, limit=self.default_limit)

        sql = (sql or '').strip()
        return {"generated_sql": sql}

    def execute_query(self, state: SQLState) -> SQLState:
        """Execute generated SQL if valid; restrict to simple SELECT statements.

        Stores textual results or error description in 'results'.
        """
        sql = (state.get("generated_sql") or "").strip()
        if not sql or sql.startswith("ERROR:"):
            return {"results": f"ERROR: No executable SQL. Source: {sql[:120]}"}

        if not self._is_select_statement(sql):
            return {"results": "ERROR: Rejected non-SELECT or unsafe SQL statement."}

        try:
            runner = self.db_toolkits.get("sql_db_query")
            if runner is None:
                return {"results": "ERROR: sql_db_query tool unavailable."}
            res = runner.invoke({"query": sql})
            return {"results": res}
        except Exception as e:
            return {"results": f"ERROR: SQL execution failed: {e}"}
    
    def generate_answer(self, state: SQLState) -> SQLState:
        """Produce natural language answer from SQL & results.

        Always returns an AIMessage—errors in previous steps surface transparently
        in the RESULT section so the model can politely explain.
        """
        if self._answer_prompt is None:
            return {"messages": AIMessage(content="Answer prompt not configured.")}

        # Process SQL Result into table
        results = (state.get("results") or "").strip()
        if not results.startswith("ERROR:") and results.strip():
            # Find column names in SELECT statement
            sql = state.get("generated_sql") or ""
            col_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql, flags=re.S | re.I)
            columns = []
            if col_match:
                cols_str = col_match.group(1)
                raw_columns = [col.strip().split(" AS ")[0] for col in cols_str.split(",")]
                raw_columns = [re.sub(r"\b[dsp]\.", "", col) for col in raw_columns]
                columns = [col.strip().split(" AS ")[-1] for col in cols_str.split(",")]
        
            # Extract data rows from results
            lines = [line for line in results.splitlines() if line.strip()][0]
            results = ""
            results = "Column names: " + ", ".join(columns) + "\n" + lines

            #TODO: Retrieve context for columns of SELECTed tables
            # data = self.qdrant.filter_payload(key="name", value='document_document')
            # print(data)

        try:
            system_prompt = self._answer_prompt.format(
                sql=(state.get("generated_sql") or "").strip(),
                result=results,
                question=get_latest_question(state),
            )
            res = self.model.invoke([SystemMessage(content=system_prompt)] + state["messages"])
            ai_message = res if isinstance(res, AIMessage) else AIMessage(content=getattr(res, "content", ""))
        except Exception as e:
            ai_message = AIMessage(content=f"Failed to generate answer. Please try again.")

        return {"messages": ai_message}

    @staticmethod
    def _check_limit(sql: str, limit: int = 10) -> str:
        limit_match = re.search(r"LIMIT\s+(\d+)", sql, flags=re.I)
        if not limit_match:
            sql = sql.rstrip().rstrip(";") + f" LIMIT {limit};"
        else:
            try:
                limit_val = int(limit_match.group(1))
                if limit_val > limit:
                    sql = re.sub(r"LIMIT\s+(\d+)", f"LIMIT {limit}", sql, flags=re.I)
            except Exception:
                # ignore parse error, leave as-is
                pass

        return sql

    @staticmethod
    def _parse_json_sql(text: str) -> str:
        if not text:
            return None
        # find first JSON object occurrence
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if not match:
            return None
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            candidate = obj.get('sql')
            if isinstance(candidate, str):
                return candidate
        except Exception:
            return None
        return None
    
    @staticmethod
    def _extract_sql(text: str) -> str:
        """Attempt to isolate a single SQL statement from model output.

        Preference order:
          1. Fenced ```sql blocks
          2. Generic fenced blocks
          3. First SELECT ... pattern
        Returns empty string if no plausible SQL fragment found.
        """
        if not text:
            return ""
        # Fenced with language
        code_block = re.search(r"```sql\s+(.*?)```", text, flags=re.S | re.I)
        if code_block:
            return code_block.group(1).strip()
        # Generic fenced
        code_block = re.search(r"```\s*(.*?)```", text, flags=re.S | re.I)
        if code_block:
            return code_block.group(1).strip()
        # First SELECT ... ; try to capture up to semicolon
        sql_like = re.search(r"SELECT\s.+?(;|$)", text, flags=re.S | re.I)
        if sql_like:
            return sql_like.group(0).strip()
        return ""
    
    @staticmethod
    def _is_select_statement(sql: str) -> bool:
        if not sql:
            return False
        sql_stripped = sql.strip()
        # Normalize whitespace for scanning
        lowered = re.sub(r"\s+", " ", sql_stripped).lower()
        if not lowered.startswith("select"):
            return False
        forbidden = [" update ", " delete ", " insert ", " drop ", " alter ", " create ", " truncate "]
        if any(f in lowered for f in forbidden):
            return False
        # Multi-statement detection: more than one semicolon or trailing content after first semicolon
        semicolons = sql_stripped.count(";")
        if semicolons > 1:
            return False
        if semicolons == 1 and not re.match(r"^.*;\s*$", sql_stripped):
            # Content after semicolon
            return False
        return True


class ToolAgentFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def inner(agent_cls):
            cls._registry[name] = agent_cls
            return agent_cls
        return inner

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseToolAgent:
        agent_cls = cls._registry.get(name)
        if not agent_cls:
            raise ValueError(f"No agent registered under '{name}'")
        return agent_cls(**kwargs)

@ToolAgentFactory.register("rfi")
class RFIAgent(BaseToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@ToolAgentFactory.register("submittal")
class SubmittalAgent(BaseToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@ToolAgentFactory.register("inspection")
class InspectionAgent(BaseToolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


