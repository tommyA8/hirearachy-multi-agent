import os
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from model.state_model import *
from agent_hub import ChatCM, RouterTeams, ConversationTeams, DatabaseTeams, ResearchTeams

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")


def chat_with_agent():
    graph = ChatCM()

    # Setting up the nodes
    graph.router = RouterTeams(
        # The mode should be a thinking model MoE
        model=ChatOllama(model="qwen3:0.6b", temperature=0)
    )
    graph.help_desk = ConversationTeams(
        # The mode should be a thinking model MoE
        model=ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
        # model=ChatNVIDIA(model="deepseek-ai/deepseek-r1", temperature=0.1, api_key="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"),
    )
    graph.database = DatabaseTeams(
        # SQL Expert
        # model=ChatOllama(model="llama3.2", temperature=0.1), 
        model=ChatNVIDIA(model="qwen/qwen2.5-coder-7b-instruct", temperature=0, api_key="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"),
        db_uri=POSTGRES_URI
    )
    graph.research = ResearchTeams(
        qdrant_url=QDRANT_URL, 
        collection_name=QDRANT_COLLECTION_NAME, 
        embeded_model_nam=EMBEDED_MODEL_NAME
    )

    # Building the agent
    memory = MemorySaver()
    agent = graph.build(checkpointer=memory, save_graph=False)

    while True:
        question = input("#> ")
        # out = agent.invoke({
        #     "messages": [HumanMessage(content=question)], 
        #     "user": UserContext(user_id=1, company_id=1, project_id=1)
        # }, 
        # config={"configurable": {"thread_id": "demo-user-001"}}
        # )
        for step in agent.stream({"messages": [HumanMessage(content=question)], "user": UserContext(user_id=1, company_id=1, project_id=1)}, 
                                 stream_mode="values", 
                                 config={"configurable": {"thread_id": "demo-user-001"}}):
            
            step["messages"][-1].pretty_print()


def test_router():
    router = RouterTeams(
        model=ChatOllama(model="qwen3:0.6b", temperature=0)
    )
    g = router.build()
    
    while True:
        query = input("Router#> ")

        for step in g.stream({"messages": [HumanMessage(content=query)], "user": UserContext(user_id=1, company_id=1, project_id=1)}, 
                            stream_mode="values", 
                            config={"configurable": {"thread_id": "router-test-001"},
                                    "recursion_limit": 100}):
            step["messages"][-1].pretty_print()


def test_research():
    research = ResearchTeams(
        qdrant_url=QDRANT_URL, 
        collection_name=QDRANT_COLLECTION_NAME, 
        embeded_model_nam=EMBEDED_MODEL_NAME
    )
    g = research.build()
    
    while True:
        query = input("Router#> ")

        for step in g.stream({"messages": [HumanMessage(content=query)], "user": UserContext(user_id=1, company_id=1, project_id=1)}, 
                            stream_mode="values", 
                            config={"configurable": {"thread_id": "research-node-test-001"},
                                    "recursion_limit": 100}):
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    chat_with_agent()
    # test_router()
    # test_research()