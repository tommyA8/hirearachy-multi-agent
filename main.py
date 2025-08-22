import os
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from utils.is_valid_tool_permission import is_valid_tool_permission
from model.state_model import *
from agent_hub import RouterTeams, ConversationTeams, DatabaseTeams, ResearchTeams

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")

class HierarchicalAgent:
    def __init__(self):
        self._agent = None
        self._help_desk = None
        self._router = None
        self._database = None
        self._research = None

    @property
    def help_desk(self):
        return self._help_desk
    
    @property
    def router(self):
        return self._router
    
    @property
    def database(self):
        return self._database
    
    @property
    def research(self):
        return self._research
    
    @router.setter
    def router(self, graph: RouterTeams):
        self._router = graph.build()
    
    @help_desk.setter
    def help_desk(self, graph: ConversationTeams):
        self._help_desk = graph.build()
    
    @database.setter
    def database(self, graph: DatabaseTeams):
        self._database = graph.build()

    @research.setter
    def research(self, graph: ResearchTeams):
        self._research = graph.build()

    def router_node(self, state: MainState):
        res = self._router.invoke({"messages": state["messages"], 
                                   "user": state["user"]})
        return {
            "messages": res["messages"],
            "tool": res["tool"], 
            # "tool_selected_reason": res["tool_selected_reason"]
        }
    
    def help_desk_node(self, state: MainState):
        res = self._help_desk.invoke({
            "messages": state["messages"],
            })

        return {"messages": res["messages"]}

    def research_node(self, state: MainState):
        res = self._research.invoke({"messages": state["messages"], 
                                     "tool": state["tool"],
                                    #  "tool_selected_reason": state["tool_selected_reason"]
                                     })
        
        return {
            "messages": res["messages"],
            "relavant_context": res["relavant_context"]
        }

    def database_node(self, state: MainState):
        res = self._database.invoke({"messages": state["messages"],
                                     "user": state["user"],
                                     "relavant_context": state["relavant_context"]})

        return {
            "messages": res["messages"], 
            # "sql_results": res["sql_results"]
        }

    def _is_related_to_cm_node(self, state: MainState):
        if state['tool'] == Tools.UNKNOWN:
            return "not_related_to_cm"
        return "related_to_cm"

    def _permission_node(self, state: MainState):
        valid = is_valid_tool_permission(user_id=state["user"].user_id, 
                                               company_id=state["user"].company_id, 
                                               project_id=state["user"].project_id, 
                                               tool_title=state["tool"].value)
        ai_msg = SystemMessage(content=state["tool"].value + " Permission: " + ("Valid" if valid else "Not Valid"))
        return {
            "messages": [ai_msg],
            "permission": "valid" if valid else "not_valid"
        }
    
    def build(self, checkpointer, save_graph=False):
        g = StateGraph(MainState)
        g.add_node("tool_classification_node", self.router_node)
        g.add_node("check_permission_node", self._permission_node)
        g.add_node("help_desk_node", self.help_desk_node)
        g.add_node("research_node", self.research_node)
        g.add_node("database_node", self.database_node)
        
        g.add_edge(START, "tool_classification_node")
        g.add_conditional_edges(
            "tool_classification_node", 
            self._is_related_to_cm_node, 
            {"related_to_cm": "check_permission_node", "not_related_to_cm": "help_desk_node"})
        g.add_conditional_edges(
            "check_permission_node",
            lambda s: s["permission"],
            {"valid": "research_node", "not_valid": "help_desk_node"},
        )
        g.add_edge("research_node", "database_node")
        g.add_edge("database_node", "help_desk_node")
        g.add_edge("help_desk_node", END)
        self.agent = g.compile(checkpointer=checkpointer)

        if save_graph:
            self.save_graph()

        return self.agent

    def save_graph(self):
        try:
            png_bytes = self.agent.get_graph().draw_mermaid_png()
            with open("chatcm_agentic_graph.png", "wb") as f:
                f.write(png_bytes)
            print("Saved graph diagram to graph.png")
        except Exception as e:
            print(f"Could not render graph diagram: {e}")


def chat_with_agent():
    graph = HierarchicalAgent()

    # Setting up the nodes
    graph.router = RouterTeams(
        model=ChatOllama(model="qwen3:4b", temperature=0.1)
    )
    graph.help_desk = ConversationTeams(
        model=ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
    )
    graph.database = DatabaseTeams(
        model=ChatOllama(model="sqlcoder:7b", temperature=0), 
        db_uri=POSTGRES_URI
    )
    graph.research = ResearchTeams(
        qdrant_url=QDRANT_URL, 
        collection_name=QDRANT_COLLECTION_NAME, 
        embeded_model_nam=EMBEDED_MODEL_NAME
    )

    # Building the agent
    memory = MemorySaver()
    agent = graph.build(checkpointer=memory, save_graph=True)

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


if __name__ == "__main__":
    chat_with_agent()
