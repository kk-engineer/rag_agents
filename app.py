import streamlit as st
import time
import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# --- CONFIG ---
MODEL_TAG = "llama3.1:8b-instruct-q4_K_M"
PDF_PATH = "docs/Mastery_by_Robert_Greene.pdf"
DB_DIR = "./ollama_chroma_db"

st.set_page_config(page_title="Mastery GPT", page_icon="📖")
st.title("📖 Document Chatbot")


# --- CACHE RAG COMPONENTS ---
@st.cache_resource
def init_rag():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if not os.path.exists(DB_DIR):
        loader = PyPDFLoader(PDF_PATH)
        splits = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(loader.load())
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    else:
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    llm = ChatOllama(model=MODEL_TAG, temperature=0.1, num_ctx=4096)
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using the detailed context below, provide a thorough, insightful, and comprehensive answer. 
    Connect different ideas from the context to explain the 'why' behind the concepts.
    If the answer is not here, say you do not know the answer, politely.
    Context: {context}
    Question: {input}
    Answer:""")

    return vectorstore.as_retriever(search_kwargs={"k": 4}), llm, prompt


retriever, llm, prompt_template = init_rag()

# --- SESSION STATE (CHAT HISTORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if query := st.chat_input("Ask something from the document..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        with st.spinner("Thinking ..."):
            # 1. Retrieval
            start_time = time.time()
            docs = retriever.invoke(query)
            context_text = "\n\n".join([d.page_content for d in docs])
            retrieval_time = time.time() - start_time

            # 2. Generation
            full_response = ""
            chain = prompt_template | llm

            # Streaming implementation in Streamlit
            gen_start = time.time()
            ttft = 0

            for chunk in chain.stream({"context": context_text, "input": query}):
                if not full_response:
                    ttft = time.time() - gen_start
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            total_time = time.time() - start_time

        # 3. Performance Metrics (ChatGPT Style Sidebar or small text)
        st.caption(f"🔍 Retrieval: {retrieval_time:.2f}s | ⏱️ Time to First Token: {ttft:.2f}s | 🚀 Total: {total_time:.2f}s")
        st.session_state.messages.append({"role": "assistant", "content": full_response})
