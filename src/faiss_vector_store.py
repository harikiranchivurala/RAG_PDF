
from openai import OpenAI
import faiss
import numpy as np
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

class FaissVectorStore:
    """Handles storage of chunk embeddings using FAISS."""
    
    def __init__(self):
        # openai.api_key = OPENAI_API_KEY
        self.openai_client = OpenAI(api_key = OPENAI_API_KEY)
        # self.dimension = 1536  # Embedding size of `text-embedding-ada-002`
        self.dimension = 768  # Embedding size of all-mpnet-base-v2`
        print("loading the model...")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance
        
    def embed_text(self, text: str) -> np.ndarray:
        """Embeds the text using OpenAI Embeddings.
        
        Args:
            text (str): The text to embed.
        
        Returns:
            np.ndarray: The vector representation of the text.
        """
        response = self.openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return np.array(response.data[0].embedding)
    
    def sentence_embed_text(self, text: str) -> np.ndarray:
        """Embeds the text using sentence Transformers Embeddings.
        
        Args:
            text (str): The text to embed.
        
        Returns:
            np.ndarray: The vector representation of the text.
        """
        # print("loading the model...")
        # model = SentenceTransformer("all-mpnet-base-v2")
        response = self.model.encode(text)
        # print("embd_size",len(response))
        
        return np.array(response)
    

    def add_chunks(self, chunks: list):
        """Adds multiple chunks to the FAISS index.
        
        Args:
            chunks (list): List of text chunks.
        """
        # embeddings = np.array([self.embed_text(chunk) for chunk in chunks])
        # embeddings = np.array([self.sentence_embed_text(chunk) for chunk in chunks])
        embeddings = np.array(self.sentence_embed_text(chunks) )
        # print("chunks_size",len(chunks))
        
        self.index.add(embeddings)
        
    def search(self, query: str, k=2) -> list:
        """Search for the most similar chunks to the query.
        
        Args:
            query (str): The query to search for.
            k (int): The number of top results to return.
        
        Returns:
            list: Indices of the top matching chunks.
        """
        # query_vector = self.embed_text(query)
        query_vector = self.sentence_embed_text(query)
        
        distances, indices = self.index.search(np.array([query_vector]), k)
        # print("hello...",distances)
        return indices[0]
