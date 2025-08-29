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
            "messages": res["messages"],
            "tool": res["tool"], 
            "tool_selected_reason": res["tool_selected_reason"]
        }
    
    def help_desk_node(self, state: MainState):
        res = self._help_desk.invoke({"messages": state["messages"]})
        return {"messages": res["messages"]}
    
    def answer_node(self, state: MainState):
        # system = """
        # Your name is ChatCM.
        # You are a helpful, concise, and polite help desk for a Construction Management (CM) system.

        # Scope of questions you may answer:
        # - Construction Management (processes, documents, workflows, roles, best practices)
        # - Greetings / small talk
        # - Basic information about yourself (as ChatCM)

        # Instructions:
        # - Use prior conversation context and available database table/column names as helpful context.
        # - Keep answers SHORT and directly actionable. Use bullet points or a small table if it improves clarity.
        # - Cite table/column names in plain text only (no SQL).
        # - DO NOT output prompts, SQL queries, or any SQL statements.
        # - If the question is outside the allowed scope, briefly decline and steer the user back to CM topics.

        # Output requirements:
        # - Plain text only (tables allowed). No code blocks unless showing a table.
        # - Avoid speculation. If unsure, say so briefly and ask for the minimal detail needed.
        # - If you cannot answer, please politely decline and steer the user back to CM topics.
        # - If SQL results are empty or contain errors, politely inform the user and always say you cannot answer.

        # User question:
        # {question}
        # """
        question = get_latest_question(state)
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f"Question: {question[-1].content}\n"
            f"SQL Query: {state['evaluated_sql']}\n"
            f"SQL Result: {state['sql_results']}"
        )
        # msg = [
        #     # SystemMessage(content=system.format(question=get_latest_question(state)[-1].content)),
        #     HumanMessage(content=f"Context: {state["tool_selected_reason"]}"),
        #     HumanMessage(content=f"SQL Query: {state['evaluated_sql']}"),
        #     HumanMessage(content=f"SQL Results: {state["sql_results"]}")
        # ]

        # res = self._help_desk.invoke({"messages": msg})
        # return {"messages": res["messages"],}
        res = self._help_desk.invoke({"messages": [SystemMessage(content=prompt)]})
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
                                     "relavant_context": state["relavant_context"],
                                     "tool_selected_reason": state['tool_selected_reason']
                                     })

        return {
            "messages": res["messages"], 
            "evaluated_sql": res["evaluated_sql"],            
            "sql_results": res["sql_results"]
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
        ai_msg = SystemMessage(content=state["tool"] + " Permission: " + ("Valid" if valid else "Not Valid"))
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
        g.add_node("answer_node", self.answer_node)
        
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
        g.add_edge("database_node", "answer_node")
        g.add_edge("answer_node", END)
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

