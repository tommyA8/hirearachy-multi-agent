import enum
from pydantic import BaseModel
from typing import List, Optional

class PermissionLevel(enum.IntEnum):
    Not_Allowed = 0
    View_Only = 1
    General = 2
    Admin = 3

class Permission(BaseModel):
    level: PermissionLevel
    tool: str

class UserContext(BaseModel):
    user_id: int
    company_id: int
    project_id: int
    tool_permissions: Optional[List[Permission]]

