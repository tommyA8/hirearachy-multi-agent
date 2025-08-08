import json
import os
from typing import List, Dict
from tqdm import tqdm
from pydantic import BaseModel
from sqlalchemy import create_engine, Table, MetaData, ForeignKey
from sqlalchemy.inspection import inspect
from model.qdrant_schema_model import SchemaMetadata, SchemaDoc
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)

class SnippetBuilder:
    def __init__(self, database_uri):
        self.db_uri = database_uri
        self.snippet_docs = None
        self._engine = self._create_engine()
        self._metadata = self._get_metadta()
        self._inspector = self._get_inspector() # Define database inspector

    def _create_engine(self):
        try:
            engine = create_engine(self.db_uri)
            engine.connect()
            
            logger.info("Connect to database succesfully.")
            return engine
        
        except Exception as e:
            logger.error(f"{e}")

    def _get_metadta(self):
        try:
            metadata = MetaData()
            metadata.reflect(bind=self._engine)
        except Exception as e:
            logger.error(f"{e}")

        return metadata

    def _get_inspector(self):
        return inspect(self._engine)
    
    def get_fields(self, table_name):
        # Get fields type
        field_names = []
        field_content = []
        for col in self._inspector.get_columns(table_name=table_name):
            field_names.append(col['name'])
            field_content.append(f"""{col['name']} ({col['type']})""")

        return field_names, field_content

    def get_foreign_keys(self, table_name):
        # Get Foreign key
        relationships = []
        related_tables = []
        for fk in self._inspector.get_foreign_keys(table_name=table_name):
            related_tables.append(f"{fk['referred_table']}")
            relationships.append(
                f"""{"_".join(fk["name"].split("_")[0:2])}.{fk['constrained_columns'][0]} → {fk['referred_table']}.{fk['referred_columns'][0]}"""
            )

        return relationships, related_tables
        
    def create_snippet(self, tables_info) -> SchemaDoc:
        # Get Table Name
        table_name = tables_info.name
        
        # Get fields type
        field_names, field_content = self.get_fields(table_name)

        # Get Foreign key
        relationships, related_tables = self.get_foreign_keys(table_name=table_name)

        # Convert List to str
        field_content_str = ','.join(field_content)
        relationships_str = ",\n".join(relationships) if relationships else None

        # Create vector content
        self.vector_content = f"""Table: {table_name}
        Description: N/A
        Fields: {field_content_str}
        Relationships:{relationships_str}
        """

        return SchemaDoc(
            id=f"Table::{table_name}",
            title=f"Description of {table_name}",
            content=self.vector_content,
            metadata=SchemaMetadata(
                table=table_name,
                fields=field_names,
                related_tables=related_tables,
            )
        )    

    def build(self) -> list:
        self.snippet_docs = []
        tables_info = self._metadata.sorted_tables
        for table in tqdm(tables_info, desc="Generating Database Docs", colour='green'):
            snippet = self.create_snippet(table)
            self.snippet_docs.append(snippet)

        return self.snippet_docs

    def to_json(self, file_name):
        if self.snippet_docs is None:
            raise RuntimeError("Please run build() before calling to_json().")
        
        with open(f"../docs/{file_name}", "w") as f:
            f.write(json.dumps([doc.model_dump() for doc in self.snippet_docs], indent=2))

if __name__ == "__main__":
    POSTGRES_URI = os.getenv("POSTGRES_URI")
    DOCS_NAME = os.getenv("DOCS_NAME")
    snippet_builder = SnippetBuilder(database_uri=POSTGRES_URI)
    snippet_builder.build()
    snippet_builder.to_json(file_name=DOCS_NAME)