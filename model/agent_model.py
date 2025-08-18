from pydantic import BaseModel
import enum

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
    selected_reason: str

class UserContext(BaseModel):
    user_id: int
    company_id: int
    project_id: int

class SQLGenerator(BaseModel):
    query: str
    
class SQLEvaluator(BaseModel):
    query: str