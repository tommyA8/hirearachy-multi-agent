from __future__ import annotations

import re
from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit


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

class DatabaseTeams:
    def __init__(self,
        model: ChatOllama,
        db_uri: str,
        dialect: str = "postgresql",
        default_top_k: int = 5,
        default_limit: int = 50
    ):  
        self.model = model
        self.sql_model = model.with_structured_output(SQLGenerator)

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
            "produce a syntactically correct {dialect} SQL query that answers the question.\n"
            "Rules:\n"
            "- Unless a specific number of rows is requested, limit to {top_k} rows.\n"
            "- Never select *; only include necessary columns. Wrap each column name in double quotes.\n"
            "- Use only columns/tables shown below; ensure column-table correctness.\n"
            "- Use date('now') to refer to the current date if the question involves 'today'.\n\n"
            "Use this format:\n"
            "Question: <question>\n"
            "SQLQuery: <query>\n"
            "Answer: <short natural-language answer to expect>\n\n"
            "Only use the following tables:\n{tables}\n\n"
            "Table relationships:\n{relationships}\n\n"
            "Context:\n{tool_context}\n\n"
            "Question: {question}\n"
        )
        
    def build(self):
        g = StateGraph(DBState)
        g.add_node("generate_node", self.generate_sql)
        g.add_node("evaluate_node", self.evaluate_sql)
        g.add_node("execute_node", self.execute_sql)

        g.add_edge(START, "generate_node")
        g.add_edge("generate_node", "evaluate_node")
        g.add_edge("evaluate_node", "execute_node")
        # g.add_conditional_edges(
        #     "execute_node",
        #     lambda s: s["sql_error"],
        #     {"error": "generate_node", "not_error": END},
        # )
        g.add_edge("execute_node", END)
        return g.compile()

    def generate_sql(self, state: DBState) -> Dict[str, Any]:
        question = get_latest_question(state)
        tool_context = str(state.get("tool_selected_reason", "")).strip() or "None."
        tables_block = "\n".join([f"Table:{tbl['table']}\n Fields: {tbl['fields']}" for tbl in state["relavant_context"]]) 
        relationships_block = state["relavant_context"][0]['relationships']
        
        rendered = self._prompt_tmpl.format(
            dialect=self.dialect,
            top_k=self.default_top_k,
            tables=tables_block,
            relationships=relationships_block,
            tool_context=tool_context,
            question=question[-1].content,
        )

        try:
            rendered = [SystemMessage(content=rendered)]
            res = self.sql_model.invoke(rendered)
            generated_sql = (res.query or "").strip()

        except Exception as e:
            # Fail closed with a clear message in state
            return {
                "messages": [AIMessage(content=f"Failed to generate SQL: {e}")],
                "generated_sql": "",
                "sql_error": "error",
            }

        return {
            "messages": [AIMessage(content="SQL Query Generated.")],
            "generated_sql": generated_sql,
        }
    
    def evaluate_sql(self, state: DBState) -> Dict[str, Any]:
        query = (state.get("generated_sql") or "").strip()
        if not query:
            return {
                "messages": [AIMessage(content="No SQL to evaluate.")],
                "evaluated_sql": "",
                "sql_error": "error",
            }

        try:
            checker = self.db_tool_map["sql_db_query_checker"]
            checked_sql = checker.invoke({"query": query})

        except Exception as e:
            return {
                "messages": [AIMessage(content=f"SQL evaluation failed: {e}")],
                "evaluated_sql": "",
                "sql_error": "error",
            }

        return {
            "messages": [AIMessage(content="SQL Query Evaluated.")],
            "evaluated_sql": checked_sql,
        }

    def execute_sql(self, state: DBState) -> Dict[str, Any]:
        sql = (state.get("evaluated_sql") or "").strip()
        if not sql:
            return {
                "messages": [AIMessage(content="No SQL to execute.")],
                "sql_results": "",
                "sql_error": "error",
            }

        # Enforce SELECT-only
        if not _is_select(sql):
            return {
                "messages": [AIMessage(content="Refusing to execute non-SELECT SQL.")],
                "sql_results": "",
                "sql_error": "error",
            }

        # Ensure LIMIT if absent
        sql = _ensure_limit(sql, self.default_limit)

        try:
            runner = self.db_tool_map["sql_db_query"]
            res = runner.invoke({"query": sql})
            res_txt = str(res)

            if "error" in res_txt.lower():
                return {
                    "messages": [AIMessage(content="SQL execution failed.")],
                    "sql_results": res_txt,
                    "sql_error": "error",
                }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"SQL execution crashed: {e}")],
                "sql_results": "",
                "sql_error": "error",
            }

        return {
            "messages": [AIMessage(content="SQL executed.")],
            "evaluated_sql": sql,
            "sql_results": res_txt,
            "sql_error": "not_error",
        }