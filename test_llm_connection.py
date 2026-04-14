from langchain_ollama import ChatOllama, OllamaEmbeddings

# 1. Initialize LLM with the EXACT tag and explicit base_url
llm = ChatOllama(
    model="llama3.1:8b-instruct-q4_K_M",
    base_url="http://127.0.0.1:11434"
)

# 2. Initialize Embeddings (Ensure you have pulled this model too!)
# If you haven't, run: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(
    #model="nomic-embed-text",
    model='llama3.1:8b-instruct-q4_K_M',
    base_url="http://127.0.0.1:11434"
)

import time
# Test connection with timer
try:
    print("Testing LLM connection...")

    start_time = time.time()  # Start timer
    response = llm.invoke("Hello, are you there? Tell me who you are?")
    end_time = time.time()  # End timer

    duration = end_time - start_time

    print(f"Success! Response time: {duration:.2f} seconds")
    print(f"Response: {response.content}")

except Exception as e:
    print(f"Connection failed: {e}")
