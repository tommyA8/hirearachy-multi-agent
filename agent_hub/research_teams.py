import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

# Model
from model.state_model import RetrieveState

# Utils
from utils.get_latest_question import get_latest_question
from utils.qdrant_helper import QdrantVector

class ResearchTeams:
    def __init__(self,
                 qdrant_url: str, 
                 collection_name: str, 
                 embeded_model_nam: str, 
                 model: ChatOllama = None
                 ):
        self.model = model
        self.qdrant = QdrantVector(qdrant_url=qdrant_url, 
                                   collection_name=collection_name,
                                   model_name=embeded_model_nam)

    def build(self):
        g = StateGraph(RetrieveState)
        g.add_node("semantic_search_node", self.semantic_search)
        g.add_edge(START, "semantic_search_node")
        g.add_edge("semantic_search_node", END)
        return g.compile()
    
    def semantic_search(self, state: RetrieveState):
        human_question = get_latest_question(state)

        # Get The Most Relevant Context 
        relevant_tables = self.qdrant.get_relevant_tables(q=human_question[-1].content, limit=2)

        # Always get company and project context
        company = self.qdrant.filter_payload(key="name", value="company_company")
        project = self.qdrant.filter_payload(key="name", value="project_project")
        relevant_tables = relevant_tables + company + project

        return {"relevant_tables": relevant_tables}
