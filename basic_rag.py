import time
import threading
import sys
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

spinner_visible = False


def spinning_wheel():
    chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    while spinner_visible:
        for char in chars:
            if not spinner_visible: break
            sys.stdout.write(f'\r{char} Thinking and Analyzing ...')
            sys.stdout.flush()
            time.sleep(0.08)
    sys.stdout.write('\r' + ' ' * 40 + '\r')
    sys.stdout.flush()


def build_rag():
    start_setup = time.time()
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if not os.path.exists(DB_DIR):
        print(f"[{time.strftime('%H:%M:%S')}] First run: Creating index (1200 chunk size)...")
        loader = PyPDFLoader(PDF_PATH)
        splits = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(loader.load())
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Loading existing index from disk...")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    llm = ChatOllama(model=MODEL_TAG, temperature=0.2, num_ctx=8192)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using the detailed context below, provide a thorough, insightful, and comprehensive answer. 
    Connect different ideas from the context to explain the 'why' behind the concepts.
    If the answer isn't here, say you don't know the answer, politely.
    Context: {context}
    Question: {input}
    Answer:""")

    print(f"✅ System Ready (Setup took: {time.time() - start_setup:.2f}s)")
    return vectorstore, llm, prompt


if __name__ == "__main__":
    vectorstore, llm, prompt = build_rag()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    while True:
        query = input("\nAsk your question please (or type 'exit'/'quit' to end the conversation: \n").strip()
        if not query or query.lower() in ['exit', 'quit']: break

        start_request = time.time()

        # 1. TIMER: RETRIEVAL
        retrieval_start = time.time()
        docs = retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])
        retrieval_time = time.time() - retrieval_start

        # 2. TIMER: GENERATION (TTFT & TOTAL)
        spinner_visible = True
        t = threading.Thread(target=spinning_wheel)
        t.start()

        print("Assistant: ", end="", flush=True)

        gen_start = time.time()
        ttft_time = 0
        first_token = True

        stream = (prompt | llm).stream({"context": context_text, "input": query})

        for chunk in stream:
            if first_token:
                ttft_time = time.time() - gen_start
                spinner_visible = False
                t.join()
                first_token = False
            print(chunk.content, end="", flush=True)

        total_time = time.time() - start_request

        # --- PRINT PERFORMANCE STATS ---
        print(f"\n\n--- Performance Stats ---")
        print(f"🔍 Retrieval Time:  {retrieval_time:.2f}s")
        print(f"⏱️  Time to First Token: {ttft_time:.2f}s")
        print(f"🚀 Total Request Cycle: {total_time:.2f}s")
        print("-" * 25)
