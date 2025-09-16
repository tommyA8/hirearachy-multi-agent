import enum
import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
import yaml

from model.state_model import DBState, SQLGenerator
from utils.get_latest_question import get_latest_question

# Utils
from utils.get_latest_question import get_latest_question
from utils.qdrant_helper import QdrantVector

from abc import ABC, abstractmethod

# class AbstractToolAgent(ABC):
#     @abstractmethod
#     def build(self):
#         pass

#     @abstractmethod
#     def retrieve_context(self):
#         pass

#     @abstractmethod
#     def generate_sql_query(self, state):
#         pass

#     @abstractmethod
#     def execute_query_and_respond(self):
#         pass

class SQLState(MessagesState):
    generated_sql: str
    results: str

class RFIAgent:
    def __init__(self, 
                 model: ChatOllama,
                 db_uri: str,
                 yaml_path: str,
                 dialect: str = "postgresql",
                 default_top_k: int = 5,
                 default_limit: int = 10
                 ):
        """
        **DEFAULT Tables**
        MUST HAVE Tables: document_document, company_company, project_project
        """
        self.model = model
        self.yaml_path = yaml_path
        self.dialect = dialect
        self.default_top_k = int(default_top_k)
        self.default_limit = int(default_limit)
        self.default_table_info = self.get_db_info(self.yaml_path, ['document_document', "company_company", "project_project"])
        self._engine = create_engine(db_uri)
        self.db = SQLDatabase(engine=self._engine)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        self.db_tools = self.toolkit.get_tools()
        self.db_tool_map = {t.name: t for t in self.db_tools}

        self._prompt_tmpl = (
            "You are an expert in {dialect} SQL and a domain specialist in Construction Management (CM).\n"
            "Your task is to generate a **syntactically correct {dialect} SQL query** that answers the user's QUESTION.\n"
            "You must use the provided **CHAT HISTORY** and **DATABASE INFORMATION** to infer intent.\n\n"

            "INTENT & RULES\n"
            "- The table `document_document` (alias `d`) always represents RFIs.\n"
            "- Always JOIN `project_project` (alias `p`) and `company_company` (alias `c`) as follows:\n"
            "    JOIN project_project AS p ON p.id = d.project_id\n"
            "    JOIN company_company AS c ON c.id = p.company_id\n"
            "- Always include WHERE clauses:\n"
            "    d.deleted IS NULL\n"
            "    d.type = 0\n"
            "- If the question implies recency (latest, newest, most recent), order by d.created_at DESC.\n"
            "- If the question implies oldest, order by d.created_at ASC.\n"
            "- Use date('now') to represent the current date when the question involves 'today'.\n"
            "- Unless a specific number of rows is requested, apply: LIMIT {top_k}.\n"
            "- Only use columns/tables listed under DATABASE INFORMATION; ensure column-table correctness.\n"
            "- If no fields are specified, default to: d.code, d.title, d.created_at.\n\n"

            "FEW-SHOT EXAMPLES\n"
            "Q: What are the latest 10 RFIs?\n"
            "A: SELECT d.code, d.title, d.created_at FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id\n"
            "JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL AND d.type = 0 AND d.project_id = 1 AND p.company_id = 1\n"
            "ORDER BY d.created_at DESC LIMIT 10;\n\n"

            "Q: How many RFIs are there?\n"
            "A: SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id\n"
            "JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL AND d.type = 0 AND d.project_id = 1 AND p.company_id = 1;\n\n"

            "Q: Get the process status of the latest RFI\n"
            "A: SELECT d.process FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id\n"
            "JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL AND d.type = 0 AND d.project_id = 1 AND p.company_id = 1\n"
            "ORDER BY d.created_at DESC LIMIT 1;\n\n"

            "Q: How many RFIs are in process?\n"
            "A: SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id\n"
            "JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL AND d.type = 0 AND d.process = 1 AND d.project_id = 1 AND p.company_id = 1;\n\n"

            "Q: How many RFIs are closed?\n"
            "A: SELECT COUNT(*) FROM document_document AS d\n"
            "JOIN project_project AS p ON p.id = d.project_id\n"
            "JOIN company_company AS c ON c.id = p.company_id\n"
            "WHERE d.deleted IS NULL AND d.type = 0 AND d.process = 4 AND d.project_id = 1 AND p.company_id = 1;\n\n"

            "DATABASE INFORMATION:\nUse only the following tables and columns:\n{table_info}\n\n"
            "QUESTION:\n{question}\n\n"
            "RESPOND FORMAT:\n"
            "Provide only the SQL query as plain text (no explanation, no markdown):\n"
        )

    def build(self):
        g = StateGraph(SQLState)
        g.add_node("generate_sql_node", self.generate_sql_query)
        g.add_node("execute_sql_node", self.execute_query)
        g.add_node("generate_answer_node", self.generate_answer)

        g.add_edge(START, "generate_sql_node")
        g.add_edge("generate_sql_node", "execute_sql_node")
        g.add_edge("execute_sql_node", "generate_answer_node")
        g.add_edge("generate_answer_node", END)
        return g.compile()

    @staticmethod
    def get_db_info(yaml_path: str, table_names: List[str]):
        with open(yaml_path, "r") as f:
            db_docs = yaml.safe_load(f)
        
        content = ""
        for table_name in table_names:
            for tbl in db_docs['database_docs']['tables'].values():
                tbl_name = tbl['name']
                if tbl_name != table_name: continue

                desc = tbl['description']
                fields = tbl['fields']
                relations = tbl['relations']
                enums = tbl['enums']

                fields_str = ""
                for f in fields:
                    for k, v in f.items():
                        fields_str += f"{k}: {v}\n"

                relations_str = "\n".join(relations)
                
                content += f"Table: {tbl_name}\nDescription: {desc}\nFields:\n{fields_str}\nRelations:\n{relations_str}\nEnums:{enums}\n"

        return content
    
    def generate_sql_query(self, state: SQLState) -> SQLState:
        #NOTE: ที่ต้องแยก  generate SQL กับ Execute เพราะบางโมเดล ไม่สามารถใช้ Tools ได้
        """Generate an SQL query based on the RFI input."""
        system_prompt = self._prompt_tmpl.format(
            dialect=self.dialect,
            top_k=self.default_top_k,
            question=get_latest_question(state),
            table_info=self.default_table_info
        )
        try:
            res = self.model.invoke(
                [SystemMessage(content=system_prompt)] + state['messages']
            )

        except Exception as e:
            return {"generated_sql": f"Failed to generate SQL: {e}"}

        return {"generated_sql": res.content}

    def execute_query(self, state: SQLState) -> SQLState:
        sql = (state.get("generated_sql") or "").strip()
        if not sql: return {"messages": [AIMessage(content="No SQL to execute.")]}

        try:
            runner = self.db_tool_map["sql_db_query"]
            res = runner.invoke({"query": sql})
            res_txt = str(res)

        except Exception as e:
            return {
                "results": f"SQL execution crashed: {e}"
            }
        
        return {"results": res_txt}
    
    def generate_answer(self, state: SQLState) -> SQLState:
        prompt = (
            "You are an expert SQL result interpreter.\n"
            "You will be given:\n"
            "  1. A natural-language QUESTION.\n"
            "  2. The executed SQL QUERY.\n"
            "  3. The TABLE INFORMATION describing the schema.\n"
            "  4. The RESULT of the SQL query.\n\n"
            "Your task is to read the SQL query and its result, understand the context, "
            "and produce a **clear and concise natural-language summary** that directly answers the QUESTION.\n\n"
            "SQL QUERY: {sql}\n"
            "TABLE INFORMATION: {table_info}\n"
            "RESULT: {result}\n\n"
            "Instructions:\n"
            "- Answer in plain English.\n"
            "- Be brief and factual, but cover the key point from the RESULT.\n"
            "- If the RESULT is empty, say so explicitly.\n"
            "- Do not repeat the SQL query.\n"
            "- Do not invent data not present in RESULT.\n\n"
            "QUESTION:\n{question}\n\n"
            "FINAL ANSWER (summary):"
        ).format(
            sql=(state.get("generated_sql") or "").strip(),
            table_info=self.default_table_info,
            result=(state.get("results") or "").strip(),
            question=get_latest_question(state)
        )
        res = self.model.invoke([SystemMessage(content=prompt)] + state['messages'])
        ai_message = res if isinstance(res, AIMessage) else AIMessage(content=res.content)
        return {"messages": ai_message}




# class SubmittalAgent(AbstractToolAgent):
#     def __init__(self):
#         pass

#     def build(self):
#         pass

#     def retrieve_context(self):
#         """Retrieve the necessary context or data for the submittal."""
#         pass

#     def generate_sql_query(self):
#         """Generate an SQL query based on the submittal input."""
#         pass

#     def execute_query_and_respond(self):
#         """Execute the generated SQL query and return the answer."""
#         pass

# class InspectionAgent(AbstractToolAgent):
#     def __init__(self):
#         pass

#     def build(self):
#         pass

#     def retrieve_context(self):
#         """Retrieve the necessary context or data for the inspection."""
#         pass

#     def generate_sql_query(self):
#         """Generate an SQL query based on the inspection input."""
#         pass

#     def execute_query_and_respond(self):
#         """Execute the generated SQL query and return the answer."""
#         pass

