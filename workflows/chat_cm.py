import warnings
warnings.filterwarnings("ignore")
from typing import Literal, Optional
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState

from agents import *
from agents.cm_tool_agent import RFIAgent, SubmittalAgent, InspectionAgent
from utils import fetch_permission_tools, get_latest_question
from model.user import UserContext, Permission
from model.cm_tools import CMTools
from constants.constants import *
from prompt_templates.prompts import RFI_SQL_PROMPT, SUBMITTAL_SQL_PROMPT, INSPECTION_SQL_PROMPT

class MainState(MessagesState):
    user: UserContext
    question_type: str
    tool: str

class ChatCM:
    def __init__(self):
        self._question_classifier = None
        self._supervisor_team = None
        self._rfi_team = None
        self._submittal_team = None
        self._inspection_team = None

    @property
    def question_classifier(self):
        return self._question_classifier
    
    @property
    def supervisor_team(self):
        return self._supervisor_team
    
    @property
    def rfi_team(self):
        return self._rfi_team
    
    @property
    def submittal_team(self):
        return self._submittal_team
    
    @property
    def inspection_team(self):
        return self._inspection_team

    @question_classifier.setter
    def question_classifier(self, graph: QuestionClassifier):
        self._question_classifier = graph.build()

    @supervisor_team.setter
    def supervisor_team(self, graph: CMSupervisor):
        self._supervisor_team = graph.build()

    @rfi_team.setter
    def rfi_team(self, graph: RFIAgent):
        self._rfi_team = graph.build()

    @submittal_team.setter
    def submittal_team(self, graph: SubmittalAgent):
        self._submittal_team = graph.build()

    @inspection_team.setter
    def inspection_team(self, graph: InspectionAgent):
        self._inspection_team = graph.build()

    def classifier_node(self, state: MainState) -> MainState:
        res = self._question_classifier.invoke({
            "messages": state["messages"]
        })
        return {
            "messages": res["messages"][-1],
            "question_type": res["question_type"]
        }

    def get_tool_permissions_node(self, state: MainState) -> MainState:
        if not state['user'].tool_permissions:
            # Get valid permission tools from Database (res has capitalize)
            res = fetch_permission_tools(user_id=state["user"].user_id,
                                           project_id=state['user'].project_id,
                                           company_id=state['user'].company_id)
            
            # Filter only RFI, Submittal, Inspection
            tool_names = [tool.name for tool in CMTools]

            # Get user's permission (level, tool_name) for RFI, Submittal and Inspection
            state['user'].tool_permissions = [Permission(level=level, tool=tool.upper())
                                              for level, tool in res if tool.upper() in tool_names]

        return {
            "messages": state['messages'][-1],
            "user": state['user']
        }

    def supervisor_node(self, state: MainState) -> MainState:
        res = self._supervisor_team.invoke({
            "messages": state["messages"],
            "user": state["user"]
        })

        tool = res.get("tool")
        if tool in (CMSupervisor.NON_CM_TOOL, CMSupervisor.NO_VALID, CMSupervisor.NEED_MORE_CNTX):
            # Pass along the AIMessage so the conversation ends gracefully for these cases.
            return {
                "messages": res["messages"][-1],
                "tool": tool, 
                "user": state["user"]
            }
        return {
            "messages": state['messages'],
            "tool": tool,
            "user": state["user"]
        }
    
    def rfi_node(self, state: MainState) -> MainState:
        res = self._rfi_team.invoke({
            "messages": state['messages'],
            'user': state['user']
        })
        return {
            "messages": res['messages'][-1]
        }
    
    def submittal_node(self, state: MainState) -> MainState:
        res = self._submittal_team.invoke({
            "messages": state['messages'],
            'user': state['user']
        })
        return {
            "messages": res['messages'][-1]
        }
    
    def inspection_node(self, state: MainState) -> MainState:
        res = self._inspection_team.invoke({
            "messages": state['messages'],
            'user': state['user']
        })
        return {
            "messages": res['messages'][-1]
        }
    
    def build(self, checkpointer, save_graph=False):
        g = StateGraph(MainState)
        g.add_node("classifier_node", self.classifier_node)
        g.add_node("get_tool_permissions_node", self.get_tool_permissions_node)
        g.add_node("supervisor_node", self.supervisor_node)
        g.add_node("rfi_node", self.rfi_node)
        g.add_node("submittal_node", self.submittal_node)
        g.add_node("inspection_node", self.inspection_node)

        g.add_edge(START, "classifier_node")
        g.add_conditional_edges(
            "classifier_node",
            lambda s: s['question_type'],
            {
                QuestionClassifier.CM: "get_tool_permissions_node", 
                QuestionClassifier.GENERAL: END,
                QuestionClassifier.NEED_MORE_CNTX: END
            }
        )
        g.add_edge("get_tool_permissions_node", "supervisor_node")
        g.add_conditional_edges(
            "supervisor_node",
            lambda s: s['tool'],
            {
                CMSupervisor.RFI: "rfi_node",
                CMSupervisor.SUBMITTAL: "submittal_node",
                CMSupervisor.INSPECTION: "inspection_node",
                CMSupervisor.NON_CM_TOOL: END,
                CMSupervisor.NEED_MORE_CNTX: END,
                CMSupervisor.NO_VALID: END
            }
        )
        g.add_edge("rfi_node", END)
        g.add_edge("submittal_node", END)
        g.add_edge("inspection_node", END)

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
            import traceback
            traceback.print_exc()


def chatcm_agent():
    graph = ChatCM()
    
    graph.question_classifier = QuestionClassifier(
        model=ChatNVIDIA(model="qwen/qwen3-next-80b-a3b-instruct", temperature=0.2, api_key=NVIDIA_LLM_API_KEY),
        )
    
    graph.supervisor_team = CMSupervisor(
        model=ChatNVIDIA(model="qwen/qwen3-next-80b-a3b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        )
    
    graph.rfi_team = RFIAgent(
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path=DB_DOCS,
        db_uri=POSTGRES_URI,
        sql_prompt=RFI_SQL_PROMPT,
        default_tables = ['document_document', "company_company", "project_project"]
    )
    
    graph.submittal_team = SubmittalAgent(
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path=DB_DOCS,
        db_uri=POSTGRES_URI,
        sql_prompt=SUBMITTAL_SQL_PROMPT,
        default_tables = ['document_document', "company_company", "project_project", "document_submittal"]
    )

    graph.inspection_team = InspectionAgent(
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0, api_key=NVIDIA_LLM_API_KEY),
        db_docs_path=DB_DOCS,
        db_uri=POSTGRES_URI,
        sql_prompt=INSPECTION_SQL_PROMPT,
        default_tables = ['document_document', "company_company", "project_project", "document_inspection"]
    )

    return graph.build(checkpointer=MemorySaver(), save_graph=False)

