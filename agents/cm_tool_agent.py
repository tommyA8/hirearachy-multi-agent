import yaml
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
                 default_tables: Optional[List[str]] = None
                 ):
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

    def generate_sql_query(self, state: SQLState) -> SQLState: #NOTE: ที่ต้องแยก  generate SQL กับ Execute เพราะบางโมเดล ไม่สามารถใช้ Tools ได้
        if self._sql_prompt is None:
            raise ValueError("SQL prompt is not set.")
        
        system_prompt = self._sql_prompt.format(dialect=self.dialect,
                                                    top_k=self.default_top_k,
                                                    question=get_latest_question(state),
                                                    table_info=self.default_table_info)
        try:
            res = self.model.invoke([SystemMessage(content=system_prompt)] + state['messages'])

        except Exception as e:
            return {"generated_sql": f"Failed to generate SQL: {e}"}

        return {"generated_sql": res.content}

    def execute_query(self, state: SQLState) -> SQLState:
        sql = (state.get("generated_sql") or "").strip()
        if not sql: return {"messages": [AIMessage(content="No SQL to execute.")]}

        try:
            runner = self.db_toolkits["sql_db_query"]
            res = runner.invoke({"query": sql})

        except Exception as e:
            return {
                "results": f"SQL execution crashed: {e}"
            }
        
        return {"results": str(res)}
    
    def generate_answer(self, state: SQLState) -> SQLState:
        if self._answer_prompt is None:
            raise ValueError("Answer prompt is not set.")
        
        system_prompt = self._answer_prompt.format(sql=(state.get("generated_sql") or "").strip(),
                                                  table_info=self.default_table_info,
                                                  result=(state.get("results") or "").strip(),
                                                  question=get_latest_question(state))
        
        res = self.model.invoke([SystemMessage(content=system_prompt)] + state['messages'])
        ai_message = res if isinstance(res, AIMessage) else AIMessage(content=res.content)
        return {"messages": ai_message}


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


