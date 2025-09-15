import os
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from model.state_model import *
# from agent_hub import RouterTeams, ConversationTeams, DatabaseTeams, ResearchTeams
from agents.is_cm_related import CMRelated
from agents.general_assistant import GeneralAssistant
from workflow.chat_cm import ChatCM, UserContext

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")

def inference():
    agent = ChatCM()
    agent.cm_related_agent = CMRelated(model= ChatOllama(model="qwen3:0.6b", temperature=0.1))
    agent.general_assistant_agent = GeneralAssistant(model= ChatOllama(model="qwen3:0.6b", temperature=0.1))
    agent = agent.build(checkpointer=MemorySaver(), save_graph=False)

    i = 0
    while True:
        question = input("#> ")

        if i == 0:
          for step in agent.stream({"messages": [HumanMessage(content=question)], 
                                    "user": UserContext(user_id=1, company_id=1, project_id=1, permission_tools=None),
                                    "permission_tools": None
                                    }, 
                                  stream_mode="values", 
                                  config={"configurable": {"thread_id": "demo-user-002"}}):
              
              step["messages"][-1].pretty_print()
        else:
          for step in agent.stream({"messages": [HumanMessage(content=question)]},
                                   stream_mode="values", 
                                   config={"configurable": {"thread_id": "demo-user-002"}}):
  
              step["messages"][-1].pretty_print()

        i += 1

if __name__ == "__main__":
    inference()