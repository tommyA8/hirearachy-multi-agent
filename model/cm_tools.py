import enum

class CMTools(enum.Enum):
    RFI = (0, "Formal clarification process with workflow, deadlines, and status tracking.")
    SUBMITTAL = (1, "Digital review/approval process for materials, shop drawings, and product data.")
    INSPECTION = (2, "Field inspections logged digitally with photos, comments, and corrective actions.")

    def __init__(self, tool: str, description: str):
        self.tool = tool
        self.description = description