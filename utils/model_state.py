import enum
import warnings
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(override=True)
warnings.filterwarnings("ignore")

from langgraph.graph import MessagesState

class Tools(enum.Enum):
    DOCUMENT = "Document"
    SUBMITTAL = "Submittal"
    RFI = "RFI"
    INSPECTION = "Inspection"
    UNKNOWN = "Unknown"
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

# Define models
class RoutingDecision(BaseModel):
    question: str
    tool: Tools
    reasoning: str

class UserContext(BaseModel):
    user_id: int
    company_id: int
    project_id: int

class SQLResponse(BaseModel):
    sql_query: str
    sql_results: str
    
# Define states
class RouterState(MessagesState):
    tool: Tools
    reasoning: str

class PermissionsState(RouterState):
    user_ctx: UserContext
    tool_permission: bool

class RetrieveState(PermissionsState):
    relavant_context: str

class DBState(RetrieveState):
    generated_sql: str

class MainState(RetrieveState):
    final_answer: str
