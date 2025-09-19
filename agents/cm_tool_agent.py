import yaml
import re
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

from utils.get_latest_question import get_latest_question

class SQLState(MessagesState):
    """State passed between SQL tool agent nodes.

    Fields:
        generated_sql: Raw SQL text produced by LLM (expected single SELECT).
        results: Text serialization of execution result OR error message.
    messages: Inherited conversation message list from MessagesState.
    """
    generated_sql: str
    results: str

class BaseToolAgent(ABC):
    def __init__(self, 
                 model: ChatOllama,
                 db_uri: str,
                 db_docs_path: str,
                 sql_prompt: str = None,
                 dialect: str = "postgresql",
                 default_top_k: int = 5,
                 default_limit: int = 10,
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
            "You are an Help Desk Assistant for a Construction Management (CM) system.\n"
            "You will be given:\n"
            "  1. A natural-language QUESTION.\n"
            "  2. The executed SQL QUERY.\n"
            "  3. TABLE INFORMATION describing the schema.\n"
            "  4. The RESULT of the SQL query.\n\n"
            "Your task: read the SQL and RESULT, understand the context, and produce a clear, concise factual answer to the QUESTION.\n\n"
            "SQL QUERY: {sql}\n"
            "TABLE INFORMATION: {table_info}\n"
            "RESULT: {result}\n\n"
            "Instructions:\n"
            "- Answer in plain English, concise and short.\n"
            "- If RESULT is empty, respond that you are unable to answer.\n"
            "- Do not repeat the SQL query text.\n"
            "- Do not hallucinate columns or values.\n"
            "- Do not tell the user aout the SQL.\n"
            "- Summarize multiple rows succinctly.\n\n"
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
            fields_str = "".join(
                f"{k}: {v}\n" for f in fields for k, v in f.items()
            )
            relations_str = "\n".join(relations)
            parts.append(
                f"Table: {table_name}\nDescription: {desc}\nFields:\n{fields_str}\nRelations:\n{relations_str}\nEnums:{enums}\n"
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
        )
        # Harden instructions a bit (idempotent if user already added similar)
        if "SELECT" not in base_prompt.upper():
            base_prompt += "\nReturn only a single valid SELECT statement."
        try:
            res = self.model.invoke([SystemMessage(content=base_prompt)] + state["messages"])
            sql_text = getattr(res, "content", "")
        except Exception as e:
            return {"generated_sql": f"ERROR: Failed to generate SQL: {e}"}

        sql_text = (sql_text or "").strip().strip('`')
        return {"generated_sql": sql_text}

    def execute_query(self, state: SQLState) -> SQLState:
        """Execute generated SQL if valid; restrict to simple SELECT statements.

        Stores textual results or error description in 'results'.
        """
        sql = (state.get("generated_sql") or "").strip()
        if not sql or sql.startswith("ERROR:"):
            return {"results": f"No executable SQL. Source: {sql[:120]}"}

        if not self._is_select_statement(sql):
            return {"results": "Rejected non-SELECT or unsafe SQL statement."}

        try:
            runner = self.db_toolkits.get("sql_db_query")
            if runner is None:
                return {"results": "ERROR: sql_db_query tool unavailable."}
            res = runner.invoke({"query": sql})
            return {"results": str(res)}
        except Exception as e:
            return {"results": f"ERROR: SQL execution failed: {e}"}
    
    def generate_answer(self, state: SQLState) -> SQLState:
        """Produce natural language answer from SQL & results.

        Always returns an AIMessage—errors in previous steps surface transparently
        in the RESULT section so the model can politely explain.
        """
        if self._answer_prompt is None:
            return {"messages": AIMessage(content="Answer prompt not configured.")}

        system_prompt = self._answer_prompt.format(
            sql=(state.get("generated_sql") or "").strip(),
            table_info=self.default_table_info,
            result=(state.get("results") or "").strip(),
            question=get_latest_question(state),
        )
        try:
            res = self.model.invoke([SystemMessage(content=system_prompt)] + state["messages"])
            ai_message = res if isinstance(res, AIMessage) else AIMessage(content=getattr(res, "content", ""))
        except Exception as e:
            ai_message = AIMessage(content=f"Failed to generate answer: {e}")
        return {"messages": ai_message}

    @staticmethod
    def _is_select_statement(sql: str) -> bool:
        """Very lightweight validation ensuring the statement is a single SELECT.

        Rejects statements containing semicolons (multiple queries) or modifying verbs.
        This is heuristic, not a full SQL parser.
        """
        sql_clean = re.sub(r"\s+", " ", sql).strip().lower()

        if not sql_clean.startswith("select"):
            return False
        forbidden = [" update ", " delete ", " insert ", " drop ", " alter ", " create "]
        return not any(tok in sql_clean for tok in forbidden)


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


