
import os
from dotenv import load_dotenv
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] # Or set it directly here
# Embedding model to use for chunk embeddings
# EMBEDDING_MODEL = "text-embedding-ada-002" 
EMBEDDING_MODEL = "text-embedding-3-small"
# OpenAI LLM to use for response generation
LLM_MODEL = "gpt-4o-mini" 
# Chunk size and overlap for better context retrieval
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 100  # Number of overlapping characters between chunks
vector_qdrant = True # True for qdrant, false for FAISS
