# AgenticAI

AgenticAI is a single-file hybrid RAG application that answers questions using a curated local knowledge base. It combines FAISS vector search and BM25 keyword retrieval to surface relevant context, then uses a Hugging Face-hosted LLM to generate grounded answers.

## Overview

The application is designed to answer questions only from the supplied knowledge base. When the requested information is not available in the retrieved context, the agent should respond that the knowledge base does not contain enough information.

## Features

- Hybrid retrieval with FAISS and BM25
- Chunked document indexing for more focused retrieval
- Knowledge-base-only answering behavior
- Simple command-line interface
- Single-file implementation for easy inspection and extension

## Requirements

- Python 3.10 or newer
- A Hugging Face access token with model access
- Internet access for downloading the embedding model and calling the hosted LLM

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required Python packages.

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install faiss-cpu numpy python-dotenv rank-bm25 sentence-transformers langchain-core langchain-text-splitters smolagents
```

## Configuration

Create a `.env` file in the project root and add your Hugging Face token:

```env
HF_TOKEN=your_huggingface_token_here
```

The application uses the following defaults:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM model: `Qwen/Qwen2.5-72B-Instruct`

## Run

Start the application with:

```bash
python Agentic_AI.py
```

After startup, ask a question at the prompt. Type `exit` or `quit` to stop the program.

## How It Works

1. The knowledge base documents are converted into LangChain documents.
2. Each document is split into overlapping chunks.
3. Chunks are embedded and indexed in FAISS.
4. The same chunks are tokenized and indexed with BM25.
5. A hybrid retrieval step combines semantic and keyword scores.
6. Retrieved context is passed to the agent so it can answer from the knowledge base.

## Project Structure

- `Agentic_AI.py` - main application entry point and retrieval logic
- `.gitignore` - ignores local environment and cache files
- `settings.json` - editor or workspace settings

## Notes

- Keep the `.env` file out of version control.
- The project is intended for knowledge-base-grounded responses, not open web search.
- If retrieval returns no useful context, the agent should refuse to guess.

## License

See [LICENSE](LICENSE) for licensing details.
