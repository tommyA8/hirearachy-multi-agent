from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from agents import *
from utils import fetch_permission_tools, get_latest_question
from model.user import UserContext, Permission
from model.cm_tools import CMTools

class MainState(MessagesState):
    user: UserContext
    question_type: str
    tool: str

class ChatCM:
    def __init__(self):
        self._question_classifier = None
        self._general_assistant_team = None
        self._supervisor_team = None
        self._rfi_team = None
        self._submittal_team = None
        self._inspection_yeam = None

    @property
    def question_classifier(self):
        return self._question_classifier
    
    @property
    def general_assistant_team(self):
        return self._general_assistant_team
    
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

    @general_assistant_team.setter
    def general_assistant_team(self, graph: GeneralAssistant):
        self._general_assistant_team = graph.build()

    @supervisor_team.setter
    def supervisor_team(self, graph: CMSupervisor):
        self._supervisor_team = graph.build()

    @rfi_team.setter
    def rfi_team(self, graph: RFIAgent):
        self._rfi_team = graph.build()

    @submittal_team.setter
    def submittal_team(self, graph):
        self._submittal_team = graph.build()


    @inspection_team.setter
    def inspection_team(self, graph):
        self._inspection_team = graph.build()

    def classifier_node(self, state: MainState) -> MainState:
        question = get_latest_question(state)
        res = self._question_classifier.invoke({
            "question": [HumanMessage(content=question)]
        })
        return {
            "question_type": res["question_type"]
        }
    
    def general_assistant_node(self, state: MainState) -> MainState:
        res = self._general_assistant_team.invoke({
            "messages": state["messages"]
        })
        return {
            "messages": res["messages"][-1]
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

        return {"user": state['user']}

    def supervisor_node(self, state: MainState) -> MainState:
        res = self._supervisor_team.invoke({
            "messages": state["messages"],
        })
        # Check if tool == UNKNOWN
        if res["tool"] == "UNKNOWN":
            return {
                "messages": AIMessage(content="Your question is not related to RFI, Submittal or Inspection."),
                "tool": "NO_VALID"
            }

        # Find not valid permission
        not_valid_tools = [pm.tool for pm in state['user'].tool_permissions if pm.level < 1]

        # If res["tool"] (model's tool selected) in not_valid_tools. It means User not allowed to use res["tool"]
        if res["tool"] in not_valid_tools:
            return {  
                "messages": AIMessage(content="User has no permission."),
                "tool": "NO_VALID"
            }

        return {"tool": res["tool"]}
    
    def rfi_node(self, state: MainState) -> MainState:
        res = self._rfi_team.invoke({
            "messages": state['messages']
        })
        return {
            "messages": res['messages'][-1]
        }
    
    def submittal_node(self, state: MainState):
        return {
            "messages": [AIMessage(content="🛠️ Submittals are currently in development. Only RFIs are supported for now")]
        }
    
    def inspection_node(self, state: MainState):
        return {
            "messages": [AIMessage(content="🛠️ Inpsection are currently in development. Only RFIs are supported for now")]
        }

    def build(self, checkpointer, save_graph=False):
        g = StateGraph(MainState)
        g.add_node("classifier_node", self.classifier_node)
        g.add_node("general_assistant_node", self.general_assistant_node)
        g.add_node("get_tool_permissions_node", self.get_tool_permissions_node)
        g.add_node("supervisor_node", self.supervisor_node)
        g.add_node("rfi_node", self.rfi_node)
        g.add_node("submittal_node", self.submittal_node)
        g.add_node("inspection_node", self.inspection_node)

        g.add_edge(START, "classifier_node")
        g.add_conditional_edges(
            "classifier_node",
            lambda s: s['question_type'],
            {"CM": "get_tool_permissions_node", "GENERAL": "general_assistant_node"}
        )
        g.add_edge("get_tool_permissions_node", "supervisor_node")
        g.add_conditional_edges(
            "supervisor_node",
            lambda s: s['tool'],
            {
                "RFI": "rfi_node",
                "SUBMITTAL": "submittal_node",
                "INSPECTION": "inspection_node",
                "NO_VALID": END
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
            print(f"Could not render graph diagram: {e}")

