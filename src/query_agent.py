
# import openai
from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL

class QueryAgent:
    """Handles the logic of querying the LLM with relevant context."""
    
    def __init__(self):
        # openai.api_key = OPENAI_API_KEY
        self.openai_client = OpenAI(api_key = OPENAI_API_KEY)
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generates an answer using the LLM given context and a question.
        
        Args:
            context (str): The context to provide to the LLM.
            question (str): The question to answer.
        
        Returns:
            str: The answer from the LLM.
        """
        # prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
        prompt = f"you're an helpful assistant, strictly Use the following context to answer the question word to word\n if no data isn't available in context just return data not available:\n\n{context}\n\nQuestion: {question}"
        
        
        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature = 0.1,
            max_tokens = 250
        )

        # print(response)
        return response.choices[0].message.content
