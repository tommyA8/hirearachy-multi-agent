import enum
from pydantic import BaseModel, Field
from typing import List
from langgraph.graph import MessagesState

class Tools(enum.Enum):
    #TODO: เขียนให่ละเอียดและเยอะขึ้น
    SUBMITTAL = ("Submittal", "Digital review/approval process for materials, shop drawings, and product data.")
    RFI = ("RFI", "Formal clarification process with workflow, deadlines, and status tracking.")
    INSPECTION = ("Inspection", "Field inspections logged digitally with photos, comments, and corrective actions.")
    WORK_ORDER = ("Work Order", "Task assignment for corrective or maintenance work tied to inspections or safety.")
    UNKNOWN = ("Unknown", "No relevant tool identified.")

    def __init__(self, tool: str, description: str):
        self.tool = tool
        self.description = description

# Define models
class RoutingDecision(BaseModel):
    tool: str
    tool_selected_reason: str

class UserContext(BaseModel):
    user_id: int
    company_id: int
    project_id: int

class SQLGenerator(BaseModel):
    query: str
    
class RouterState(MessagesState):
    tool: str
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