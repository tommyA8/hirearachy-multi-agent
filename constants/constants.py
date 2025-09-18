import os
from dotenv import load_dotenv
load_dotenv(override=True)

DB_DOCS = os.getenv("DB_DOCS")
POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL")
