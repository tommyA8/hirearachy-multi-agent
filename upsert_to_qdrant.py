import os
from qdrant_client.models import VectorParams, PointStruct, Distance
from utils.snippet_builder import SnippetBuilder
from utils.qdrant_helper import QdrantVector
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")
POSTGRES_URI = os.getenv("POSTGRES_URI")
DOCS_NAME = os.getenv("DOCS_NAME")

def main():
    snippet_builder = SnippetBuilder(database_uri=POSTGRES_URI)
    vector_store    = QdrantVector(qdrant_url=QDRANT_URL, 
                                   collection_name=COLLECTION_NAME, 
                                   model_name=EMBEDED_MODEL_NAME)
    
    # Create a snippet database docs
    docs = snippet_builder.build()

    # Comnvert to dictionary
    docs = [doc.model_dump() for doc in docs]

    # Create Collection
    vector_store.create_collection(size=768, distance=Distance.COSINE)

    # Upsert
    vector_store.upsert_snippets(snippets=docs)

if __name__ == "__main__":
    main()
