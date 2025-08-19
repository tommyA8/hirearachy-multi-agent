from abc import ABC, abstractmethod
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from utils.agents import *

class AgentNodes(ABC):

    @abstractmethod
    def router_node(self, ):
        pass

    @abstractmethod
    def help_desk_node(self, ):
        pass

    @abstractmethod
    def permission_node(self, ):
        pass

    @abstractmethod
    def is_related_to_cm_node(self, ):
        pass

    @abstractmethod
    def retrieval_node(self, ):
        pass

    @abstractmethod
    def database_node(self, ):
        pass

class HierarchicalAgent(AgentNodes):
    def __init__(self):
        self._agent = None
        self._help_desk = None
        self._router = None
        self._database = None
        self._retrieval = retrieval_team()

    @property
    def help_desk(self):
        return self._help_desk
    
    @property
    def router(self):
        return self._router
    
    @property
    def retrieval(self):
        return self._retrieval
    
    @property
    def database(self):
        return self._database
    
    @help_desk.setter
    def help_desk(self, model: ChatOllama):
        self._help_desk = help_desk_team(model)

    @router.setter
    def router(self, model: ChatOllama):
        self._router = router_team(model)
    
    @database.setter
    def database(self, model: ChatOllama):
        self._database = database_team(model)

    def help_desk_node(self, state: MainState):
        res = self._help_desk.invoke({
            "messages": state["messages"],
            })

        return {"messages": res["messages"]}
    
    def help_desk_with_permission_node(self, state: MainState):
        res = self._help_desk.invoke({
            "messages": state["messages"],
            "permission": state["permission"],
            "sql_results": state["sql_results"]
            })

        return {"messages": res["messages"]}

    def router_node(self, state: MainState):
        res = self._router.invoke({"messages": state["messages"], 
                                "user": state["user"]})
        return {
            "messages": res["messages"],
            "tool": res["tool"], 
            "selected_reason": res["selected_reason"]
        }
    
    def retrieval_node(self, state: MainState):
        res = self._retrieval.invoke({"messages": state["messages"], 
                                     "tool": state["tool"],
                                     "selected_reason": state["selected_reason"]})
        
        return {
            "messages": res["messages"],
            "relavant_context": res["relavant_context"]
        }

    def database_node(self, state: MainState):
        res = self._database.invoke({"messages": state["messages"],
                                     "user": state["user"],
                                     "relavant_context": state["relavant_context"]})

        return {"messages": res["messages"], 
                "sql_results": res["sql_results"]}

    def is_related_to_cm_node(self, state: MainState):
        if state['tool'] == Tools.UNKNOWN:
            return "not_related_to_cm"
        return "related_to_cm"

    def permission_node(self, state: MainState):
        valid = self._is_valid_tool_permission(user_id=state["user"].user_id, 
                                               company_id=state["user"].company_id, 
                                               project_id=state["user"].project_id, 
                                               tool_title=state["tool"].value)
        ai_msg = SystemMessage(content=state["tool"].value + " Permission: " + ("Valid" if valid else "Not Valid"))
        return {
            "messages": [ai_msg],
            "permission": "valid" if valid else "not_valid"
            }
        
    def _is_valid_tool_permission(self, user_id: int, company_id: int, project_id: int, tool_title: str) -> bool:
        """
        Check if the user has permission to use a specific tool.
        """
        params = {
            "user_id": user_id,
            "company_id": company_id,
            "project_id": project_id,
            "tool_title": tool_title,
        }

        a   = Table("auth_user", metadata, schema="public", autoload_with=engine)
        cu  = Table("company_companyuser", metadata, schema="public", autoload_with=engine)
        c   = Table("company_company", metadata, schema="public", autoload_with=engine)
        pu  = Table("project_projectuser", metadata, schema="public", autoload_with=engine)
        p   = Table("project_project", metadata, schema="public", autoload_with=engine)
        cp  = Table("company_permission", metadata, schema="public", autoload_with=engine)
        cpg = Table("company_permissiongroup", metadata, schema="public", autoload_with=engine)
        tl  = Table("company_toollabels", metadata, schema="public", autoload_with=engine)

        stmt = (
            select(tl.c.title).distinct()
            .select_from(
                a
                .join(cu, a.c.id == cu.c.user_id)
                .join(c, c.c.id == cu.c.company_id)
                .join(pu, a.c.id == pu.c.user_id)
                .join(p, p.c.id == pu.c.project_id)
                .join(cp, cp.c.id == pu.c.permission_id)
                .join(cpg, cpg.c.permission_id == pu.c.permission_id)
                .join(tl, tl.c.id == cpg.c.tool_id)
            )
            .where(
                a.c.id == bindparam("user_id"),
                c.c.id == bindparam("company_id"),
                p.c.id == bindparam("project_id"),
                tl.c.title == bindparam("tool_title"),
            )
        )

        with engine.connect() as conn:
            res = conn.execute(stmt, params).scalars().all()
            if res:
                return True

        return False
    
    def build(self, checkpointer, save_graph=False):
        g = StateGraph(MainState)

        g.add_node("router_node", self.router_node)
        g.add_node("check_permission_node", self.permission_node)
        g.add_node("help_desk_node", self.help_desk_node)
        g.add_node("retrieval_node", self.retrieval_node)
        g.add_node("database_node", self.database_node)
        g.add_node("help_desk_with_permission_node", self.help_desk_with_permission_node)
        
        g.add_edge(START, "router_node")
        g.add_conditional_edges(
            "router_node", 
            self.is_related_to_cm_node, 
            {"related_to_cm": "check_permission_node", "not_related_to_cm": "help_desk_node"})
        g.add_conditional_edges(
            "check_permission_node",
            lambda s: s["permission"],
            {"valid": "retrieval_node", "not_valid": "help_desk_with_permission_node"},
        )
        g.add_edge("retrieval_node", "database_node")
        g.add_edge("database_node", "help_desk_with_permission_node")
        g.add_edge("help_desk_with_permission_node", END)

        self.agent = g.compile(checkpointer=checkpointer)

        if save_graph:
            self.save_graph()

        return self.agent

    def save_graph(self):
        try:
            png_bytes = self.agent.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(png_bytes)
            print("Saved graph diagram to graph.png")
        except Exception as e:
            print(f"Could not render graph diagram: {e}")


def chat_with_agent():
    graph = HierarchicalAgent()

    # Setting the agent's model
    graph.help_desk = ChatOllama(model="qwen3:1.7b", temperature=0.1)
    graph.router = ChatOllama(model="qwen3:1.7b", temperature=0.1)
    graph.database = ChatOllama(model="qwen3:1.7b", temperature=0.1)

    # Building the agent
    memory = MemorySaver()
    agent = graph.build(checkpointer=memory, save_graph=False)

    while True:
        question = input("#> ")
        out = agent.invoke({
            "messages": [HumanMessage(content=question)], 
            "user": UserContext(user_id=1, company_id=1, project_id=1)
        }, 
        config={"configurable": {"thread_id": "demo-user-001"}}
        )
        print("Agent: ", out["messages"][-1].content)


if __name__ == "__main__":
    chat_with_agent()
    # question = "Hello, My name is Bob, What's latest document code of submittal"
    #"What's latest document code of submittal"

    # demo(question)
    # for step in agent.stream(
    #     {"messages": [HumanMessage(content=question)],
    #     "user": UserContext(user_id=1, 
    #                         company_id=1, 
    #                         project_id=1)},
    #     stream_mode="values",
    #     config=cfg,
    #     # recursion_limit=100
    # ):
    #     step["messages"][-1].pretty_print()
    
    # res = agent.invoke({
    #     "messages": [HumanMessage(content=question)],
    #     "user": UserContext(user_id=1, company_id=1, project_id=1),
    # }, config=cfg)
    # print(res)
    