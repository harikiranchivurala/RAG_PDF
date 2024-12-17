from src.pdf_parser import PDFParser
from src.text_splitter import TextSplitter
from src.faiss_vector_store import FaissVectorStore
from src.query_agent import QueryAgent
from src.qdrant_vector_store import QdrantVectorStore

from src.config import vector_qdrant
import argparse


def get_answer(query,store,chunks):
    # 4. Query the most relevant chunks
    relevant_indices = store.search(query)
    relevant_chunks = relevant_indices if vector_qdrant else [chunks[i] for i in relevant_indices] 

    # 5. Generate the final answer using LLM
    context = " ".join(relevant_chunks)
    # print("context",context)
    agent = QueryAgent()
    answer = agent.generate_answer(context, query)
    # answer = "NULL"
    # print(answer)
    return answer


def main(pdf_path: str, queries: list):
    response = {}
    # 1. Extract text from PDF
    text = PDFParser.extract_text_from_pdf(pdf_path)
    
    # 2. Split text into chunks
    splitter = TextSplitter()
    chunks = splitter.split_text(text, chunk_size=1000, overlap=100)
    
    # 3. Embed and store chunks in vector database
    # store = FaissVectorStore()
    store = QdrantVectorStore() if vector_qdrant else FaissVectorStore()
    
    store.add_chunks(chunks)
    for query in queries:
        query = query.strip()
        response[query] = get_answer(query,store,chunks)
    
    print("\n💡 Answer:\n\n", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF and answer a list of questions.")
    
    # Argument for a list of questions (comma-separated)
    parser.add_argument(
        '-q', '--questions', 
        type=str, 
        required=True, 
        help="list of questions separated with # to ask about the PDF."
    )
    
    # Argument for PDF file
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=True, 
        help="Path to the PDF file to process."
    )

    args = parser.parse_args()
    
    pdf_path = args.file
    questions = args.questions.split("#")
    
    # print(pdf_path)
    main(pdf_path,questions)