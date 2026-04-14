# RAG Agents
A simple implementation of Retrieval-Augmented Generation (RAG) using local LLMs and intelligent agents. 
This project focuses on building autonomous systems capable of retrieving relevant context from private pdf document to provide accurate, grounded responses.

# 🚀 Getting Started
1. Install Ollama
Before running the project, you must have Ollama installed to manage your local LLMs.

- For macOS/Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- For Windows:
Download the installer from the [Ollama website](https://ollama.com/download/windows).

# 2. Set Up the Model
This project is optimized for the Llama 3.1 8B Instruct (Q4_K_M) model. 
Once Ollama is installed, pull and run the specific quantized version:

```bash
# Pull the model (this may take some time, depends on your internet speed)
ollama run llama3.1:8b-instruct-q4_K_M

# Run the model (keep it running for RAG to work)
ollama run llama3.1:8b-instruct-q4_K_M
```

# 3. Clone and Setup Project
```bash
# Clone the repository
git clone https://github.com/kk-engineer/rag_agents.git
cd rag_agents

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

# 4. Run Streamlit App
``` bash
streamlit run app.py
```

