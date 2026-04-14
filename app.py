import streamlit as st
import time
import os
import tempfile
import uuid
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import numpy as np
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
MODEL_TAG = "llama3.1:8b-instruct-q4_K_M"
CONTEXT_WINDOW = 2048

st.set_page_config(page_title="Document GPT", page_icon="📖")
st.title("📖 Document Chatbot")


# --- RAG COMPONENTS ---
@st.cache_resource
def get_embeddings():
    # Loading this once into RAM via @st.cache_resource
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def run_cluster_summary(docs, num_clusters=5):
    """
    Summarizes massive docs by clustering chunks and only
    summarizing the most 'representative' chunk from each cluster.
    """
    llm = ChatOllama(model=MODEL_TAG, temperature=0)
    embeddings_model = get_embeddings()

    total_chunks = len(docs)

    # If the document is small, don't cluster. Just use all chunks.
    if total_chunks <= num_clusters:
        #st.info(f"📄 Document is small ({total_chunks} sections). Skipping clustering...")
        selected_chunks = docs
    else:
        # 1. Get embeddings for all chunks
        #st.info(f"🧬 Analyzing themes across {len(docs)} sections...")
        # Extract just the text for embedding
        contents = [doc.page_content for doc in docs]
        vectors = embeddings_model.embed_documents(contents)
        vectors = np.array(vectors)

        # 2. Cluster the chunks using K-Means
        # This finds 'num_clusters' groups of similar topics
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

        # 3. Find the chunk closest to the center of each cluster
        # These are our 'representative' chunks
        representative_indices = []
        for i in range(num_clusters):
            # Find indices of points in this cluster
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            # Find the index of the point closest to the center
            closest_index = np.argmin(distances)
            representative_indices.append(closest_index)

        # Sort indices to keep some chronological flow
        representative_indices.sort()
        selected_chunks = [docs[i] for i in representative_indices]

    # 4. Summarize only the representative chunks
    #st.info(f"✨ Distilling {num_clusters} key themes...")
    map_prompt = ChatPromptTemplate.from_template("Summarize the key point of this section:\n\n{context}")
    map_chain = map_prompt | llm | StrOutputParser()

    # --- MAPPING & REDUCING
    # UI Elements
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Sequential Mapping
    summaries = []
    for i, chunk in enumerate(selected_chunks):
        res = map_chain.invoke({"context": chunk.page_content})
        summaries.append(res)

        progress_bar.progress((i + 1) / len(selected_chunks))

    # 4. Parallel Mapping
    # summaries = [None] * len(selected_chunks)
    # total_selected = len(selected_chunks)
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     # Create a mapping of future -> index to keep track of the original order
    #     futures_to_index = {
    #         executor.submit(map_chain.invoke, {"context": chunk.page_content}): i
    #         for i, chunk in enumerate(selected_chunks)
    #     }
    #
    #     completed_count = 0
    #     # as_completed yields futures as soon as they finish
    #     for future in as_completed(futures_to_index):
    #         idx = futures_to_index[future]
    #         try:
    #             res = future.result()
    #             summaries[idx] = res
    #
    #             # Update Progress
    #             completed_count += 1
    #             progress_val = completed_count / total_selected
    #             progress_bar.progress(progress_val)
    #             progress_text.markdown(f"✅ Finished theme {completed_count} of {total_selected}")
    #
    #         except Exception as e:
    #             st.error(f"Error processing chunk {idx}: {e}")
    #             summaries[idx] = "Summary unavailable for this section."

    # Clean up UI
    progress_text.empty()
    progress_bar.empty()

    # 5. Reduction
    reduce_prompt = ChatPromptTemplate.from_template(
        "The following are key thematic summaries from a large document:\n\n{summaries}\n\n"
        "Provide a final high-level executive summary and the top 5 takeaways."
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    progress_bar.empty()
    return reduce_chain.invoke({"summaries": "\n\n".join(summaries)})

def process_documents(uploaded_files):
    """Creates a fresh DB in a unique folder to ensure zero old context."""
    start_proc_time = time.time()

    # Generate a unique path to ensure no file locks/persistence from old runs
    unique_id = str(uuid.uuid4())[:8]
    fresh_db_dir = f"./db_{unique_id}"

    documents = []
    all_pages_text = "" # Create a variable to hold EVERYTHING

    # --- 1. EXTRACTION PROGRESS ---
    extraction_progress = st.empty()
    bar_extract = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        extraction_progress.markdown(f"📄 **Extracting text:** {uploaded_file.name}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents.extend(loader.load())

        # JOIN ALL PAGES TOGETHER
        pages = loader.load()
        for page in pages:
            all_pages_text += page.page_content + "\n\n"

        os.remove(tmp_file_path)
        bar_extract.progress((i + 1) / len(uploaded_files))

    extraction_progress.empty()
    bar_extract.empty()

    st.info("Generating embeddings...")
    retrieval_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    retrieval_chunks = retrieval_splitter.split_documents(documents)

    # STREAM 2: Chunks for Summarization (Map-Reduce)
    # Much larger chunks reduce the number of LLM calls significantly
    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    # Use split_text instead of split_documents for the giant string
    # We wrap it in a Document object so the rest of your code stays compatible
    summary_texts = summary_splitter.split_text(all_pages_text)
    st.session_state.summary_chunks = [
        type('obj', (object,), {'page_content': t}) for t in summary_texts
    ]

    # Initialize Chroma with an empty collection first
    vectorstore = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=fresh_db_dir
    )

    # --- 3. EMBEDDING PROGRESS (BATCHED) ---
    embedding_status = st.empty()
    bar_embed = st.progress(0)

    # Batching: Chroma is faster when you send chunks in groups
    batch_size = 50
    total_chunks = len(retrieval_chunks)

    for i in range(0, total_chunks, batch_size):
        batch = retrieval_chunks[i: i + batch_size]
        embedding_status.markdown(
            f"🧬 **Embedding chunks:**  {i} to {min(i + batch_size, total_chunks)} of {total_chunks}...")

        # Add the batch to the vector store
        vectorstore.add_documents(batch)

        # Update progress
        progress_val = min((i + batch_size) / total_chunks, 1.0)
        bar_embed.progress(progress_val)

    embedding_status.empty()
    bar_embed.empty()

    total_duration = time.time() - start_proc_time
    st.sidebar.success(f"✅ Processed {len(uploaded_files)} files in {total_duration:.2f}s")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# --- INITIALIZATION ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Files")

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("🚀 Start Processing"):
            with st.spinner("Processing..."):
                # This replaces the old retriever in state, dropping old context
                st.session_state.retriever = process_documents(uploaded_files)
    else:
        st.info("Upload PDFs to begin.")

    # Clear Cache Button Explanation
    if st.button("🗑️ Clear App Cache"):
        st.cache_resource.clear()
        st.success("Resource cache cleared!")
        st.rerun()
    st.caption("Clears models from RAM, but not the DB files.")

# --- LLM SETUP ---
@st.cache_resource
def init_llm():
    return ChatOllama(
        model=MODEL_TAG,
        temperature=0.1,
        num_ctx=CONTEXT_WINDOW,
        num_thread=8  # Matches M1 core count
    )

llm = init_llm()
prompt_template = ChatPromptTemplate.from_template("""
Answer the question using the detailed context below, provide a thorough, insightful, and comprehensive answer. 
Connect different ideas from the context to explain the 'why' behind the concepts.
If the answer is not here, say you do not know the answer, politely.
Context: {context}
Question: {input}
Answer:""")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask something from the documents..."):
    if st.session_state.retriever is None:
        st.warning("Please upload and process documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            resp_start_time = time.time()
            response_placeholder = st.empty()
            # We initialize full_response here to ensure it exists for the session state later
            full_response = ""

            with st.spinner("Agent..."):
                # 1. Detection Logic
                summary_keywords = ["summarize", "summary", "about", "takeaway", "take away", "overview"]
                is_summary_request = any(word in query.lower() for word in summary_keywords)

                # 2. Check for the correct key: summary_chunks
                if is_summary_request and "summary_chunks" in st.session_state and st.session_state.summary_chunks:
                    st.caption("📝 Using full document summary...")
                    # Use full_response so it is captured in the message history correctly
                    #full_response = run_lcel_map_reduce(st.session_state.summary_chunks)
                    # Even if you have 1000 chunks, it will only do 12 LLM calls
                    full_response = run_cluster_summary(st.session_state.summary_chunks, num_clusters=2)
                    # Display the result
                    response_placeholder.markdown(full_response)

                else:
                    # Standard Vector Search Path
                    docs = st.session_state.retriever.invoke(query)
                    context_text = "\n\n".join([d.page_content for d in docs])
                    st.caption("🔍 Using vector search ...")

                    chain = prompt_template | llm
                    for chunk in chain.stream({"context": context_text, "input": query}):
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                # 3. Final Metadata
                total_resp_time = time.time() - resp_start_time
                st.caption(f"🏁 Done in {total_resp_time:.2f}s")

        st.session_state.messages.append({"role": "assistant", "content": full_response})