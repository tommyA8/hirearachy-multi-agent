import os
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agents import *
from workflows.chat_cm import ChatCM, UserContext
from agents.cm_tool_agent import ToolAgentFactory

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY")

def test_classifier():
    graph = QuestionClassifier(
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
    )
    agent = graph.build()
    
    while True:
        question = input("#> ")
        res = agent.invoke(input={"question": [HumanMessage(content=question)]})
        print(res)

def test_general_assistant():
    graph = GeneralAssistant(
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
    )
    agent = graph.build()
    
    while True:
        question = input("#> ")
        res = agent.invoke({"messages": [HumanMessage(content=question)]})
        print(res)

def test_cm_supervisor():
    graph = CMSupervisor(
        model=ChatOllama(model="qwen3:0.6b", temperature=0)
        # model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
    )
    agent = graph.build()
    
    while True:
        question = input("#> ")
        res = agent.invoke({"messages": [HumanMessage(content=question)]})
        print(res)

def test_rfi_agent():
    # create an RFI agent
    graph = ToolAgentFactory.create(
        "rfi",
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path="docs/cm_db_knowledge.yaml",
        db_uri=POSTGRES_URI,
        default_tables = ['document_document', "company_company", "project_project"]
    )

    # configure its answer prompt if needed
    graph.sql_prompt = (
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

            # "FEW-SHOT EXAMPLES\n"
            # "Q: What are the latest 10 RFIs?\n"
            # "A: SELECT d.code, d.title, d.created_at FROM document_document AS d\n"
            # "JOIN project_project AS p ON p.id = d.project_id\n"
            # "JOIN company_company AS c ON c.id = p.company_id\n"
            # "WHERE d.deleted IS NULL AND d.type = 0 AND d.project_id = 1 AND p.company_id = 1\n"
            # "ORDER BY d.created_at DESC LIMIT 10;\n\n"

            # "Q: How many RFIs are there?\n"
            # "A: SELECT COUNT(*) FROM document_document AS d\n"
            # "JOIN project_project AS p ON p.id = d.project_id\n"
            # "JOIN company_company AS c ON c.id = p.company_id\n"
            # "WHERE d.deleted IS NULL AND d.type = 0 AND d.project_id = 1 AND p.company_id = 1;\n\n"

            # "Q: Get the process status of the latest RFI\n"
            # "A: SELECT d.process FROM document_document AS d\n"
            # "JOIN project_project AS p ON p.id = d.project_id\n"
            # "JOIN company_company AS c ON c.id = p.company_id\n"
            # "WHERE d.deleted IS NULL AND d.type = 0 AND d.project_id = 1 AND p.company_id = 1\n"
            # "ORDER BY d.created_at DESC LIMIT 1;\n\n"

            # "Q: How many RFIs are in process?\n"
            # "A: SELECT COUNT(*) FROM document_document AS d\n"
            # "JOIN project_project AS p ON p.id = d.project_id\n"
            # "JOIN company_company AS c ON c.id = p.company_id\n"
            # "WHERE d.deleted IS NULL AND d.type = 0 AND d.process = 1 AND d.project_id = 1 AND p.company_id = 1;\n\n"

            # "Q: How many RFIs are closed?\n"
            # "A: SELECT COUNT(*) FROM document_document AS d\n"
            # "JOIN project_project AS p ON p.id = d.project_id\n"
            # "JOIN company_company AS c ON c.id = p.company_id\n"
            # "WHERE d.deleted IS NULL AND d.type = 0 AND d.process = 4 AND d.project_id = 1 AND p.company_id = 1;\n\n"

            "DATABASE INFORMATION:\nUse only the following tables and columns:\n{table_info}\n\n"
            "QUESTION:\n{question}\n\n"
            "RESPOND FORMAT:\n"
            "Provide only the SQL query as plain text (no explanation, no markdown):\n"
        )

    # build its graph
    agent = graph.build()

    # run graph
    initial_state = {"messages": [HumanMessage(content="What's 10 latest RFIs?")]}
    res = agent.invoke(initial_state)
    print(res['messages'][-1].content)
    print(res['generated_sql'])

def test_submittal_agent():
    # create an RFI agent
    graph = ToolAgentFactory.create(
        "submittal",
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path="docs/cm_db_knowledge.yaml",
        db_uri=POSTGRES_URI,
        default_tables = ['document_document', "company_company", "project_project", "document_submittal"]
    )

    # configure its answer prompt if needed
    graph.sql_prompt = (
            "You are an expert in {dialect} SQL and a domain specialist in Construction Management (CM).\n"
            "Your task is to generate a **syntactically correct {dialect} SQL query** that answers the user's QUESTION.\n"
            "You must use the provided **CHAT HISTORY** and **DATABASE INFORMATION** to infer intent.\n\n"

            "INTENT & RULES\n"
            "- The table `document_document` (alias `d`) always represents RFIs.\n"
            "- The table `document_submittal` (alias `s`), when JOINED with `document_document` (alias `d`), represents submittals.\n" #NOTE
            "- Always JOIN `project_project` (alias `p`) and `company_company` (alias `c`) as follows:\n"
            "    JOIN project_project AS p ON p.id = d.project_id\n"
            "    JOIN company_company AS c ON c.id = p.company_id\n"
            "- Always include WHERE clauses:\n"
            "    d.deleted IS NULL\n"
            "    d.type = 1 (Submittal)\n" #NOTE
            "- If the question implies recency (latest, newest, most recent), order by d.created_at DESC.\n"
            "- If the question implies oldest, order by d.created_at ASC.\n"
            "- Use date('now') to represent the current date when the question involves 'today'.\n"
            "- Unless a specific number of rows is requested, apply: LIMIT {top_k}.\n"
            "- Only use columns/tables listed under DATABASE INFORMATION; ensure column-table correctness.\n"
            "- If no fields are specified, default to: d.code, d.title, d.created_at.\n\n"

            "DATABASE INFORMATION:\nUse only the following tables and columns:\n{table_info}\n\n"
            "QUESTION:\n{question}\n\n"
            "RESPOND FORMAT:\n"
            "Provide only the SQL query as plain text (no explanation, no markdown):\n"
        )

    # build its graph
    agent = graph.build()

    # run graph
    initial_state = {"messages": [HumanMessage(content="What's 10 latest Submittals?")]}
    res = agent.invoke(initial_state)
    print(res['messages'][-1].content)
    print(res['generated_sql'])

def test_inspection_agent():
    # create an RFI agent
    graph = ToolAgentFactory.create(
        "submittal",
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path="docs/cm_db_knowledge.yaml",
        db_uri=POSTGRES_URI,
        default_tables = ['document_document', "company_company", "project_project", "document_inspection"]
    )

    # configure its answer prompt if needed
    graph.sql_prompt = (
            "You are an expert in {dialect} SQL and a domain specialist in Construction Management (CM).\n"
            "Your task is to generate a **syntactically correct {dialect} SQL query** that answers the user's QUESTION.\n"
            "You must use the provided **CHAT HISTORY** and **DATABASE INFORMATION** to infer intent.\n\n"

            "INTENT & RULES\n"
            "- The table `document_document` (alias `d`) always represents RFIs.\n"
            "- The table `document_inspection` (alias `i`), when JOINED with `document_document` (alias `d`), represents inspections.\n" #NOTE
            "- Always JOIN `project_project` (alias `p`) and `company_company` (alias `c`) as follows:\n"
            "    JOIN project_project AS p ON p.id = d.project_id\n"
            "    JOIN company_company AS c ON c.id = p.company_id\n"
            "- Always include WHERE clauses:\n"
            "    d.deleted IS NULL\n"
            "    d.type = 2 (Inspection)\n" #NOTE
            "- If the question implies recency (latest, newest, most recent), order by d.created_at DESC.\n"
            "- If the question implies oldest, order by d.created_at ASC.\n"
            "- Use date('now') to represent the current date when the question involves 'today'.\n"
            "- Unless a specific number of rows is requested, apply: LIMIT {top_k}.\n"
            "- Only use columns/tables listed under DATABASE INFORMATION; ensure column-table correctness.\n"
            "- If no fields are specified, default to: d.code, d.title, d.created_at.\n\n"

            "DATABASE INFORMATION:\nUse only the following tables and columns:\n{table_info}\n\n"
            "QUESTION:\n{question}\n\n"
            "RESPOND FORMAT:\n"
            "Provide only the SQL query as plain text (no explanation, no markdown):\n"
        )

    # build its graph
    agent = graph.build()

    # run graph
    initial_state = {"messages": [HumanMessage(content="What's 10 latest Inspections?")]}
    res = agent.invoke(initial_state)
    print(res['messages'][-1].content)
    print(res['generated_sql'])

#TODO: prompt template management
if __name__ == "__main__":
    # test_classifier()
    # test_general_assistant()
    # test_cm_supervisor()
    # test_rfi_agent()
    # test_submittal_agent()
    test_inspection_agent()