import json
import os
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)

class QdrantVector:
    def __init__(self, 
                 qdrant_url: str, 
                 collection_name: str, 
                 model_name: str="nomic-embed-text:latest"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.model_name = model_name

        # Create qdrant collection
        self.client = QdrantClient(qdrant_url)

        # Initial Embedding Model
        self.embedder = OllamaEmbeddings(model=self.model_name)

    def create_collection(self, size=768, distance=Distance.COSINE):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=size, distance=distance),
                # optional: tune shards, replication_factor, etc.
            )
        else:
            logger.info("Collection already created")

    def upsert_snippets(self, snippets):
        points = []
        i = 0
        for snip in tqdm(snippets, desc="Upserting a Snippets Docs", colour='blue'):
            vector = self.embedder.embed_query(snip["content"])
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "text": snip["content"],
                        "title": snip["title"],
                        **snip["metadata"]
                    }
                )
            )
            i += 1
        self.client.upsert(collection_name=self.collection_name, points=points)

    def load_snippet_docs(self, file_name):
        with open(file_name, "r") as f:
            docs = json.load(f)
        return docs

if __name__ == "__main__":
    QDRANT_URL = os.getenv("QDRANT_URL")
    COLLECTION_NAME = "Test" #os.getenv("QDRANT_COLLECTION_NAME")
    EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")

    vector_store = QdrantVector(qdrant_url=QDRANT_URL, 
                                collection_name=COLLECTION_NAME, 
                                model_name=EMBEDED_MODEL_NAME)
    
    # Create Collection
    vector_store.create_collection(size=768, distance=Distance.COSINE)

    # Load example snippets
    docs = vector_store.load_snippet_docs("/home/tommii/swd/retriever-agent/docs/cm-database-knowledges.json")

    # Upsert
    vector_store.upsert_snippets(snippets=docs)