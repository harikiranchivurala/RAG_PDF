
# import openai
from openai import OpenAI
from qdrant_client import QdrantClient,models
from qdrant_client.models import Distance, VectorParams,PointStruct
import uuid
import tqdm


import numpy as np
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

class QdrantVectorStore:
    """Handles storage of chunk embeddings using Qdrant."""
    
    def __init__(self):
        # openai.api_key = OPENAI_API_KEY
        self.openai_client = OpenAI(api_key = OPENAI_API_KEY)
        # self.dimension = 1536  # Embedding size of `text-embedding-ada-002`
        self.dimension = 768  # Embedding size of all-mpnet-base-v2`
        self.batch_size = 128
        print("loading the model...")
        self.model = SentenceTransformer("all-mpnet-base-v2")

        
        self.index = QdrantClient(":memory:")
        self.index.create_collection(
        collection_name="pdf_collection",
        vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )


        
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
        response = self.model.encode(text,batch_size=self.batch_size)
        
        return np.array(response)
    

    def add_chunks(self, chunks: list):
        """Adds multiple chunks to the FAISS index.
        
        Args:
            chunks (list): List of text chunks.
        """
        # embeddings = np.array([self.embed_text(chunk) for chunk in chunks])
        # embeddings = np.array([self.sentence_embed_text(chunk) for chunk in chunks])
        
        embeddings = np.array(self.sentence_embed_text(chunks))
        # Calculate the number of batches
        # num_batches = len(chunks) // self.batch_size + (1 if len(chunks) % self.batch_size != 0 else 0)

        # Iterate over batches and upload each batch
        # for batch_idx in tqdm.tqdm(range(num_batches)):
        #     start_idx = batch_idx * self.batch_size
        #     end_idx = min((batch_idx + 1) * self.batch_size, len(chunks))

        batch_points = [
                        models.Record(
                            id=str(uuid.uuid4()),
                            vector=embeddings[idx].tolist(),
                            payload={"data":doc}
                        ) 
                        # for idx, doc in tqdm.tqdm(chunks, total=len(chunks))
                        for idx, doc in enumerate(chunks)
                        
                    ]
        
        self.index.upload_points(
                        collection_name="pdf_collection",
                        points=batch_points
                    )
        # self.index.add(embeddings)
        
    def search(self, query: str, k=2) -> list:
        """Search for the most similar chunks to the query.
        
        Args:
            query (str): The query to search for.
            k (int): The number of top results to return.
        
        Returns:
            list: docs of the top matching chunks.
        """
        # query_vector = self.embed_text(query)
        query_vector = self.sentence_embed_text(query)

        hits = self.index.search(
        collection_name="pdf_collection",
        query_vector=query_vector.tolist(),
        limit=k
            )
        # for hit in hits:
        #     print(hit.payload, "score:", hit.score)
        data = [hit.payload['data'] if hit.score > 0.25 else "data is not available" for hit in hits ]

        
        return data
