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


if __name__ == "__main__":
    # test_classifier()
    # test_general_assistant()
    test_cm_supervisor()