import enum
from pydantic import BaseModel, Field
from typing import List
from langgraph.graph import MessagesState

class Tools(enum.Enum):
    DOCUMENT = "Document"
    SUBMITTAL = "Submittal"
    RFI = "RFI"
    INSPECTION = "Inspection"
    KANBAN = "Kanban"
    DIRECTORY = "Directory"
    TRANSMITTAL = "Transmittal"
    WORK_ORDER = "Work Order"
    SAFETY = "Safety"
    INSPECTION_TEST_PLAN = "Inspection Test Plan"
    SPECIFICATION = "Specification"
    SCHEDULE = "Schedule"
    FORM = "Form"
    LOCATION = "Location"
    TAG = "Tag"
    WORK_ORDER_GROUP = "Work Order Group"
    MEETING = "Meeting"
    UNKNOWN = "Unknown"

# Define models
class RoutingDecision(BaseModel):
    question: str
    tool: Tools
    tool_selected_reason: str

class UserContext(BaseModel):
    user_id: int
    company_id: int
    project_id: int

class SQLGenerator(BaseModel):
    query: str
    
class RouterState(MessagesState):
    tool: Tools
    tool_selected_reason: str

class PermissionsState(RouterState):
    user: UserContext
    permission: str

class RetrieveState(PermissionsState):
    relavant_context: str

class DBState(RetrieveState):
    generated_sql: str
    evaluated_sql: str
    sql_results: List
    sql_error: str 

class MainState(DBState):
    pass