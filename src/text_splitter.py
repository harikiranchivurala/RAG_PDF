import re

class TextSplitter:
    """Splits text into smaller overlapping chunks."""
    
    @staticmethod
    def split_text(text: str, chunk_size: int, overlap: int) -> list:
        """Splits the input text into chunks.
        
        Args:
            text (str): The input text to split.
            chunk_size (int): The size of each chunk.
            overlap (int): The number of overlapping characters between chunks.
        
        Returns:
            list: A list of text chunks.
        """
        tokens = re.split(r'(?<=[.!?])\s+', text)  # Split on sentence boundaries
        chunks = []
        current_chunk = ""
        
        for token in tokens:
            if len(current_chunk) + len(token) < chunk_size:
                current_chunk += token + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = token + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap
        final_chunks = []
        for i in range(len(chunks)):
            chunk_start = max(0, i * (chunk_size - overlap))
            final_chunks.append(text[chunk_start:chunk_start + chunk_size])
        # print(final_chunks)
        # print(len(final_chunks))
        return final_chunks


