from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from utils.is_valid_tool_permission import is_valid_tool_permission
from utils.get_latest_question import get_latest_question
from model.state_model import *
from agent_hub import RouterTeams, ConversationTeams, DatabaseTeams, ResearchTeams

class ChatCM:
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
            "tool": res["tool"], 
            "tool_selected_reason": res["tool_selected_reason"]
        }
    
    def help_desk_node(self, state: MainState):
        res = self._help_desk.invoke({"messages": state["messages"]})
        return {
            "messages": res["messages"]
        }
    
    # def answer_node(self, state: MainState):
    #     question = get_latest_question(state)
    #     max_rows = 10
    #     prompt = (
    #         "You are a precise presenter. Given a user question and SQL results, "
    #         "return ONLY a Markdown table snippet of the top rows, with no extra prose.\n\n"

    #         "Formatting rules:\n"
    #         f"- Use at most the first {max_rows} rows, in the given order (results are already sorted).\n"
    #         "- If there are no rows, return exactly:\n"
    #         "> No results.\n"
    #         "- If the results are a list of objects (dict-like), use the object keys (from the first row) as table headers.\n"
    #         "- If the results are a list of tuples/arrays, and headers are not provided, name columns as col1, col2, col3, ...\n"
    #         "- Truncate cell values longer than 80 characters with an ellipsis …\n"
    #         "- Render ONLY the Markdown table (or the exact no-results line). Do not add any other text.\n"
    #         "- Always change datetime format to ISO 8601 (YYYY-MM-DDTHH:MM:SS).\n\n"

    #         "User Question:\n"
    #         f"{question[-1].content}\n\n"

    #         "SQL Results (Python repr / JSON-like):\n"
    #         f"{state['sql_results']}\n\n"

    #         "Output:\n"
    #     )
    #     res = self._help_desk.invoke({"messages": state["messages"] + [SystemMessage(content=prompt)]})
    #     return {"messages": res["messages"]}

    def research_node(self, state: MainState):
        res = self._research.invoke({"messages": state["messages"]})
        return {
            "relevant_tables": res["relevant_tables"]
        }

    def database_node(self, state: MainState):
        res = self._database.invoke({"messages": state["messages"],
                                     "user": state["user"],
                                     "relevant_tables": state["relevant_tables"],
                                     "tool": state['tool'],
                                     "tool_selected_reason": state['tool_selected_reason']
                                     })
        return {
            "messages": res["messages"],
        }

    def _is_related_to_cm_node(self, state: MainState):
        if state['tool'] == "Unknown":
            return "not_related_to_cm"
        return "related_to_cm"

    def _permission_node(self, state: MainState):
        valid = is_valid_tool_permission(user_id=state["user"].user_id, 
                                               company_id=state["user"].company_id, 
                                               project_id=state["user"].project_id, 
                                               tool_title=state["tool"])
        return {
            "permission": "valid" if valid else "not_valid"
        }
    
    def build(self, checkpointer, save_graph=False):
        g = StateGraph(MainState)
        g.add_node("tool_classification_node", self.router_node)
        g.add_node("check_permission_node", self._permission_node)
        g.add_node("help_desk_node", self.help_desk_node)
        g.add_node("research_node", self.research_node)
        g.add_node("database_node", self.database_node)
        # g.add_node("answer_node", self.answer_node)
        
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
        g.add_edge("database_node", END)
        # g.add_edge("answer_node", END)
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

