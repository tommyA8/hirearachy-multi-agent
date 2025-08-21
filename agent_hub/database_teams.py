from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import DBState, SQLGenerator

# Utils
from utils.get_latest_question import get_latest_question

## SQL
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit


class DatabaseTeams:
    def __init__(self, model: ChatOllama, db_uri: str):
        self.model = model
        self.sql_model = model.with_structured_output(SQLGenerator)

        self._engine = create_engine(db_uri)
        self.db = SQLDatabase(engine=self._engine) #SQLDatabase.from_engine(self.engine)  # or SQLDatabase.from_uri("postgresql+psycopg://...")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)

    def build(self):
        g = StateGraph(DBState)
        g.add_node("generate_node", self.generate_sql)
        g.add_node("evaluate_node", self.evaluate_sql)
        g.add_node("execute_node", self.execute_sql)

        g.add_edge(START, "generate_node")
        g.add_edge("generate_node", "evaluate_node")
        g.add_edge("evaluate_node", "execute_node")
        g.add_edge("execute_node", END)
        return g.compile()

    def generate_sql(self, state: DBState):
        prompt = """
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
        
        human_question = get_latest_question(state)

        rendered_system = prompt.format(dialect='postgresql', 
                                        top_k=5,
                                        company_id=state["user"].company_id,
                                        project_id=state["user"].project_id,
                                        table_context=state['relavant_context'],
                                        question=human_question[-1].content)
        
        res = self.sql_model.invoke(rendered_system)

        return {
            "messages": [AIMessage(content="SQL Query Generated.")], 
            "generated_sql": res.query
        }

    def evaluate_sql(self, state: DBState):
        prompt = """
        You are given a SQL query and a list of relevant tables and their relationships.
        You MUST TO evaluate the provided SQL query, check for existence of columns and tables and correctness of the query.
        
        Finally, Generate a correct, clear, and readable SQL query.
        
        Evaluated SQLQuery: <Evaluated SQLQuery>

        SQLQuery:
        {query}

        Relevant tables:
        {relevant_tables}

        Table relationships:
        {relationships}
        """

        rendered_system = prompt.format(query=state["generated_sql"],
                                        relevant_tables=state["relavant_context"],
                                        relationships=state["relavant_context"][0]['relationships'])

        res = self.sql_model.invoke(rendered_system)

        return {
            "messages": [AIMessage(content="SQL Query Evaluated.")], 
            "evaluated_sql": res.query
        }

    def execute_sql(self, state: DBState):
        sql = state["evaluated_sql"] # sql = state.get("evaluated_sql").strip().rstrip(";")

        if not sql.lower().startswith("select"):
            print(f"Refusing to execute non-SELECT SQL: {sql}")
            raise ValueError("Refusing to execute non-SELECT SQL.")
        
        if " limit " not in sql.lower():
            sql += " LIMIT 50"

        tools_list = self.toolkit.get_tools()
        tool_map = {t.name: t for t in tools_list}

        query_tool = self._get_query_tool(tool_map)

        tool_out = query_tool.invoke({"query": sql})  # returns a string with rows/preview

        return {
            "messages": [AIMessage(content="SQL executed via SQLDatabaseToolkit.")], 
            "sql_results": tool_out
        }
    
    def _get_query_tool(self, tool_map):
        preferred = ["query_sql_db", "sql_db_query", "sql_db_query_tool"]
        for name in preferred:
            if name in tool_map:
                return tool_map[name]
        # fallback: fuzzy match
        for name in tool_map:
            if "query" in name and "sql_db" in name:
                return tool_map[name]
        raise KeyError(f"No SQL query tool found. Available: {list(tool_map)}")
