# main.py
from src.pdf_parser import PDFParser
from src.text_splitter import TextSplitter
from src.faiss_vector_store import FaissVectorStore
from src.query_agent import QueryAgent
from src.qdrant_vector_store import QdrantVectorStore
import streamlit as st
from src.template import css, bot_template, user_template
from src.config import vector_qdrant


def get_answer(query,store):

    # 4. Query the most relevant chunks
    # store = QdrantVectorStore()
    relevant_indices = store.search(query)
    # relevant_chunks = relevant_indices if vector_qdrant else [chunks[i] for i in relevant_indices] 
    relevant_chunks = relevant_indices
    # print(relevant_chunks)
    # 5. Generate the final answer using LLM
    context = " ".join(relevant_chunks)
    # print("context",context)
    agent = QueryAgent()
    answer = agent.generate_answer(context, query)
    # answer = "NULL"
    # print(answer)
    return answer

def main(store):
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    
    st.header("Chat with Uploaded PDF :books:")
    question=st.text_input("Ask question from your document:")
    if question and question != st.session_state.conversation:
        st.session_state.conversation = question  # Store the question in session state
        st.write(user_template.replace("{{MSG}}",question),unsafe_allow_html=True) # showing the user message
        st.session_state.chat_history = get_answer(question, store)  # Store the answer in session state

    # Display the retrieved answer
    if st.session_state.chat_history:
        # Write the bot message 
        st.write(bot_template.replace("{{MSG}}",st.session_state.chat_history),unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=False)
        if st.button("Process"):
            if docs is not None:
                # Save the uploaded file temporarily
                file_path = "temp.pdf"
                with open(file_path, "wb") as f:
                    f.write(docs.getbuffer())

            with st.spinner("Processing"):
                
                #get the pdf
                text = PDFParser.extract_text_from_pdf(file_path)
                
                # 2. Split text into chunks
                splitter = TextSplitter()
                
                chunks = splitter.split_text(text, chunk_size=1000, overlap=100)
                
                # 3. Embed and store chunks in vector database
                
                store.add_chunks(chunks)



if __name__ == '__main__':
    # store = QdrantVectorStore()
    if 'vector_store' not in st.session_state:
        # if vector_qdrant:
        #     st.session_state.vector_store = QdrantVectorStore()
        # else:
        #     st.session_state.vector_store = FaissVectorStore()
        st.session_state.vector_store = QdrantVectorStore()
    store = st.session_state.vector_store
    main(store)

