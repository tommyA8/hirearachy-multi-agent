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
from agents.cm_tool_agent import RFIAgent, SubmittalAgent, InspectionAgent
from constants.constants import *
from prompt_templates.prompts import RFI_SQL_PROMPT, SUBMITTAL_SQL_PROMPT, INSPECTION_SQL_PROMPT
from model.user import Permission, UserContext

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL")

def test_classifier():
    graph = QuestionClassifier(
        # model=ChatOllama(model="qwen3:4b", temperature=0, base_url=OLLAMA_URL)
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
    )
    agent = graph.build()
    
    while True:
        question = input("#> ")
        for step in agent.stream({"messages": [HumanMessage(content=question)]}, 
                                 stream_mode="values"):
            step["messages"][-1].pretty_print()

        step["question_type"] = step.get("question_type", "None")
        print("\nQuestion Type: ", step["question_type"])

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
        # model=ChatOllama(model="qwen3:1.7b", temperature=0),
        model=ChatNVIDIA(model="qwen/qwen2.5-7b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
    )
    agent = graph.build(checkpointer=MemorySaver())
    
    while True:
        question = input("#> ")
        for step in agent.stream({"messages": [HumanMessage(content=question)],
                                  "user": UserContext(user_id=1,
                                                      project_id=1,
                                                      company_id=1,
                                                      tool_permissions=[Permission(level=1, tool="RFI"),
                                                                        Permission(level=1, tool="SUBMITTAL"),
                                                                        Permission(level=1, tool="INSPECTION")])}, 
                                 stream_mode="values", 
                                 config={"configurable": {"thread_id": "test-cm-supervisor"}}):
            step["messages"][-1].pretty_print()
        
        print("Tool: ", step['tool'])

def test_rfi_agent():
    agent = RFIAgent(
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path=DB_DOCS,
        db_uri=POSTGRES_URI,
        sql_prompt=RFI_SQL_PROMPT,
        default_tables = ['document_document', "company_company", "project_project"]
    )
    agent = agent.build()

    # run graph
    initial_state = {"messages": [HumanMessage(content="List me most recent 350 RFIs for project 1 under company 1")], 
                     "user": UserContext(user_id=1, project_id=1, company_id=1, tool_permissions=[Permission(level=1, tool="RFI")])}

    for step in agent.stream(initial_state, stream_mode="values"):
        step['messages'][-1].pretty_print()

    print(step['generated_sql'])

def test_submittal_agent():
    agent = RFIAgent(
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path=DB_DOCS,
        db_uri=POSTGRES_URI,
        sql_prompt=SUBMITTAL_SQL_PROMPT,
        default_tables = ['document_document', "company_company", "project_project", "document_submittal"]
    )
    agent = agent.build()

    # run graph
    initial_state = {"messages": [HumanMessage(content="What's docs code of latest submittals?")], 
                     "user": UserContext(user_id=1, project_id=1, company_id=1, tool_permissions=[Permission(level=1, tool="RFI")])}

    for step in agent.stream(initial_state, stream_mode="values"):
        step['messages'][-1].pretty_print()

    print("\nSQL: \n")
    print(step['generated_sql'])

def test_inspection_agent():
    agent = RFIAgent(
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path=DB_DOCS,
        db_uri=POSTGRES_URI,
        sql_prompt=INSPECTION_SQL_PROMPT,
        default_tables = ['document_document', "company_company", "project_project", "document_inspection"]
    )
    agent = agent.build()

    # run graph
    initial_state = {"messages": [HumanMessage(content="What's docs code of latest submittals?")], 
                     "user": UserContext(user_id=1, project_id=1, company_id=1, tool_permissions=[Permission(level=1, tool="Inspection")])}

    for step in agent.stream(initial_state, stream_mode="values"):
        step['messages'][-1].pretty_print()

    print("\nSQL: \n")
    print(step['generated_sql'])

if __name__ == "__main__":
    # test_classifier()
    # test_general_assistant()
    # test_cm_supervisor()
    # test_rfi_agent()
    test_submittal_agent()
    # test_inspection_agent()