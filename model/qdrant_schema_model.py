from typing import List
from pydantic import BaseModel

class SchemaMetadata(BaseModel):
    table: str
    fields: List[str]
    relationships: List[str]
    related_tables: List[str]

class SchemaDoc(BaseModel):
    id: str
    title: str
    content: str
    metadata: SchemaMetadata