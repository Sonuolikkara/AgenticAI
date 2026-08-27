# ============================================================
# AGENTIC HYBRID RAG SYSTEM
# Single-file VS Code implementation
# ============================================================

import os
import re

import faiss
import numpy as np

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import CodeAgent, InferenceClientModel, tool


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Improved chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

VECTOR_TOP_K = 6
KEYWORD_TOP_K = 6
FINAL_TOP_K = 4

VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

MIN_VECTOR_SCORE = 0.35
MIN_KEYWORD_SCORE = 0.0

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
}

MAX_AGENT_STEPS = 5


# ============================================================
# 3. KNOWLEDGE BASE
# ============================================================

DOCUMENTS = [
    {
        "id": "AI001",
        "title": "Artificial Intelligence",
        "content": """
Artificial Intelligence is the field of building computer systems
capable of performing tasks that normally require human intelligence.
AI applications include prediction, classification, recommendation,
natural language processing, computer vision and decision support.
""",
    },

    {
        "id": "LLM001",
        "title": "Large Language Models",
        "content": """
Large Language Models are neural network models trained on large
datasets. They process text using tokens and generate responses by
predicting subsequent tokens. LLMs can perform summarisation,
translation, question answering, code generation and information
extraction.
""",
    },

    {
        "id": "RAG001",
        "title": "Retrieval Augmented Generation",
        "content": """
Retrieval Augmented Generation, commonly called RAG, combines
information retrieval with language generation.

A user question is converted into a search query, relevant
information is retrieved from a knowledge base, and the retrieved
information is supplied to a language model as context.

RAG can help language models generate answers grounded in external
knowledge instead of relying only on information contained in the
model parameters.
""",
    },

    {
        "id": "EMB001",
        "title": "Embeddings",
        "content": """
Embeddings are numerical representations of text.

An embedding model converts sentences, paragraphs or documents into
vectors. Texts with similar meanings tend to have similar vector
representations.

Embeddings are commonly used for semantic search, clustering,
recommendations and retrieval systems.
""",
    },

    {
        "id": "CHUNK001",
        "title": "Document Chunking",
        "content": """
Chunking divides large documents into smaller pieces before they are
converted into embeddings.

Chunking allows retrieval systems to find focused passages rather
than retrieving an entire document.

Chunk size and overlap are important parameters.

Very small chunks can lose important context, while very large
chunks can reduce retrieval precision.

Proper chunking helps a RAG system retrieve relevant information
while preserving enough context for the language model to generate
a reliable answer.
""",
    },

    {
        "id": "VECTOR001",
        "title": "Vector Search",
        "content": """
Vector search retrieves documents by comparing the vector embedding
of a query with the embeddings stored in a vector index.

Similarity can be calculated using cosine similarity, dot product or
other distance measures.

Vector search is useful when the query and the relevant document use
different wording but have similar meanings.
""",
    },

    {
        "id": "HYBRID001",
        "title": "Hybrid Search",
        "content": """
Hybrid search combines keyword search and vector search.

Keyword search is useful for exact terms, names, identifiers and
technical expressions.

Vector search is useful for finding semantically similar information
even when the wording is different.

Combining both approaches can improve retrieval coverage because the
two methods provide complementary retrieval capabilities.
""",
    },

    {
        "id": "BM25",
        "title": "BM25 Keyword Retrieval",
        "content": """
BM25 is a keyword-based information retrieval algorithm.

It ranks documents according to how relevant their terms are to the
user query.

BM25 is particularly useful when exact words, technical terms,
identifiers or names are important for retrieval.
""",
    },

    {
        "id": "FAISS",
        "title": "FAISS Vector Index",
        "content": """
FAISS is a library for efficient similarity search over dense
vectors.

In a RAG system, document embeddings can be stored in a FAISS index.
When a user submits a query, the query is converted into an embedding
and compared with the stored vectors to retrieve similar documents.
""",
    },

    {
        "id": "EVAL001",
        "title": "RAG Evaluation",
        "content": """
RAG applications should be evaluated for retrieval relevance,
correctness, groundedness, completeness, latency and cost.

Retrieval quality is important because a language model can only
reliably use information that is supplied to it correctly.

Groundedness measures whether the generated answer is supported by
the retrieved context.
""",
    },
]


# ============================================================
# 4. BUILD DOCUMENT OBJECTS
# ============================================================

def build_documents():
    """
    Convert the knowledge-base records into LangChain Documents.
    """

    documents = []

    for item in DOCUMENTS:

        document = Document(
            page_content=item["content"].strip(),
            metadata={
                "id": item["id"],
                "title": item["title"],
            },
        )

        documents.append(document)

    return documents


# ============================================================
# 5. DOCUMENT CHUNKING
# ============================================================

def create_chunks(documents):
    """
    Split documents into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

def create_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    dimension = (
        model.get_sentence_embedding_dimension()
    )

    print(
        f"Embedding dimension: {dimension}"
    )

    return model


# ============================================================
# 7. CREATE FAISS VECTOR INDEX
# ============================================================

def create_vector_index(
    chunks,
    embedding_model,
):

    print("Creating document embeddings...")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors."
    )

    return index


# ============================================================
# 8. TOKENIZATION FOR BM25
# ============================================================

def tokenize(text):
    """
    Simple tokenizer for BM25.
    """

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


# ============================================================
# 9. CREATE BM25 INDEX
# ============================================================

def create_bm25_index(chunks):

    tokenized_documents = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print("BM25 index created.")

    return bm25


# ============================================================
# 10. VECTOR SEARCH
# ============================================================

def vector_search(
    query,
    embedding_model,
    vector_index,
    chunks,
    top_k=VECTOR_TOP_K,
):

    if not chunks:
        return []

    top_k = min(
        top_k,
        len(chunks),
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    scores, indices = vector_index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0],
    ):

        index = int(index)

        if index < 0 or index >= len(chunks):
            continue

        chunk = chunks[index]

        results.append(
            {
                "id": chunk.metadata["id"],
                "title": chunk.metadata["title"],
                "text": chunk.page_content,
                "score": float(score),
            }
        )

    return results


# ============================================================
# 11. BM25 KEYWORD SEARCH
# ============================================================

def keyword_search(
    query,
    bm25,
    chunks,
    top_k=KEYWORD_TOP_K,
):

    if not chunks:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    top_k = min(
        top_k,
        len(chunks),
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        index = int(index)

        results.append(
            {
                "id": chunks[index].metadata["id"],
                "title": chunks[index].metadata["title"],
                "text": chunks[index].page_content,
                "score": float(scores[index]),
            }
        )

    return results


# ============================================================
# 12. NORMALIZE SCORES
# ============================================================

def normalize_scores(scores):

    if not scores:
        return []

    scores = np.asarray(
        scores,
        dtype=float,
    )

    minimum = np.min(scores)
    maximum = np.max(scores)

    if np.isclose(
        minimum,
        maximum,
    ):
        return np.ones(
            len(scores)
        )

    normalized = (
        (scores - minimum)
        / (maximum - minimum)
    )

    return normalized


# ============================================================
# 13. HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    embedding_model,
    vector_index,
    bm25,
    chunks,
    top_k=FINAL_TOP_K,
):

    vector_results = vector_search(
        query=query,
        embedding_model=embedding_model,
        vector_index=vector_index,
        chunks=chunks,
        top_k=VECTOR_TOP_K,
    )

    keyword_results = keyword_search(
        query=query,
        bm25=bm25,
        chunks=chunks,
        top_k=KEYWORD_TOP_K,
    )

    vector_normalized = normalize_scores(
        [
            result["score"]
            for result in vector_results
        ]
    )

    keyword_normalized = normalize_scores(
        [
            result["score"]
            for result in keyword_results
        ]
    )

    combined = {}

    # --------------------------------------------------------
    # VECTOR RESULTS
    # --------------------------------------------------------

    for position, result in enumerate(
        vector_results
    ):

        document_id = result["id"]

        combined[document_id] = {
            "id": result["id"],
            "title": result["title"],
            "text": result["text"],
            "raw_vector_score": float(
                result["score"]
            ),
            "vector_score": float(
                vector_normalized[position]
            ),
            "keyword_score": 0.0,
            "raw_keyword_score": 0.0,
        }

    # --------------------------------------------------------
    # KEYWORD RESULTS
    # --------------------------------------------------------

    for position, result in enumerate(
        keyword_results
    ):

        document_id = result["id"]

        if document_id not in combined:

            combined[document_id] = {
                "id": result["id"],
                "title": result["title"],
                "text": result["text"],
                "raw_vector_score": 0.0,
                "vector_score": 0.0,
                "raw_keyword_score": float(
                    result["score"]
                ),
                "keyword_score": float(
                    keyword_normalized[position]
                ),
            }

        else:

            combined[document_id][
                "keyword_score"
            ] = float(
                keyword_normalized[position]
            )

            combined[document_id][
                "raw_keyword_score"
            ] = float(
                result["score"]
            )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    for result in combined.values():

        result["hybrid_score"] = (
            VECTOR_WEIGHT
            * result["vector_score"]
            +
            KEYWORD_WEIGHT
            * result["keyword_score"]
        )

    # --------------------------------------------------------
    # SORT RESULTS
    # --------------------------------------------------------

    results = sorted(
        combined.values(),
        key=lambda item: item[
            "hybrid_score"
        ],
        reverse=True,
    )

    return results[:top_k]


def has_relevant_support(result):
    return (
        result["raw_vector_score"] >= MIN_VECTOR_SCORE
        or result["raw_keyword_score"] > MIN_KEYWORD_SCORE
    )


# ============================================================
# 14. RETRIEVE CONTEXT
# ============================================================

def retrieve_context(query):

    results = hybrid_search(
        query=query,
        embedding_model=embedding_model,
        vector_index=vector_index,
        bm25=bm25,
        chunks=chunks,
        top_k=FINAL_TOP_K,
    )

    results = [
        result
        for result in results
        if has_relevant_support(result)
    ]

    if not results:

        return (
            "No relevant information was found "
            "in the knowledge base."
        )

    context = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        context.append(
            f"""
SOURCE {number}
Document ID: {result['id']}
Title: {result['title']}

Content:
{result['text']}
""".strip()
        )

    return "\n\n".join(context)


# ============================================================
# 15. RETRIEVAL TOOL
# ============================================================

@tool
def knowledge_base_search(
    query: str,
) -> str:
    """
    Search the knowledge base using hybrid retrieval.

    The search combines semantic vector retrieval using FAISS
    and keyword retrieval using BM25.

    Args:
        query: Natural-language question or search query.

    Returns:
        Relevant passages from the knowledge base.
    """

    return retrieve_context(query)


# ============================================================
# 16. INITIALIZE KNOWLEDGE BASE
# ============================================================

print()
print("=" * 70)
print("INITIALIZING HYBRID RAG SYSTEM")
print("=" * 70)

documents = build_documents()

chunks = create_chunks(
    documents
)

print(
    f"Documents loaded: {len(documents)}"
)

print(
    f"Chunks created: {len(chunks)}"
)

embedding_model = create_embedding_model()

vector_index = create_vector_index(
    chunks,
    embedding_model,
)

bm25 = create_bm25_index(
    chunks
)

print("=" * 70)
print("KNOWLEDGE BASE READY")
print("=" * 70)


# ============================================================
# 17. CREATE AGENT
# ============================================================

def create_agent():

    hf_token = os.getenv(
        "HF_TOKEN"
    )

    if not hf_token:

        raise RuntimeError(
            """
HF_TOKEN is not configured.

Create a .env file in the project folder:

HF_TOKEN=your_huggingface_token_here

Then run the program again.
"""
        )

    print()
    print("Connecting to Hugging Face...")

    model = InferenceClientModel(
        model_id=LLM_MODEL,
        token=hf_token,
        max_tokens=512,
        temperature=0.2,
    )

    agent = CodeAgent(
        tools=[
            knowledge_base_search
        ],
        model=model,
        max_steps=MAX_AGENT_STEPS,

        # Hide detailed agent execution logs
        verbosity_level=0,

        instructions="""
You are an AI assistant with access to a knowledge base.

Your job is to answer the user's question accurately and
concisely.

IMPORTANT RULES:

1. Use the knowledge_base_search tool when information from
   the knowledge base is needed.

2. Base answers on the retrieved information.

3. Do not invent information that is not supported by the
   retrieved context.

4. Do not use general world knowledge, internet knowledge, or
    prior training to answer.

5. If the knowledge base does not contain enough information
    to answer the question, clearly say that the knowledge base
    does not contain enough information.

6. Do not expose internal reasoning.

7. If the retrieved context says that no relevant information
    was found, return that the knowledge base does not contain
    enough information.

8. Return only the final answer to the user.

9. Keep the answer clear and reasonably concise.
""",
    )

    print("Agent created successfully.")

    return agent


# ============================================================
# 18. MAIN APPLICATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AGENTIC HYBRID RAG QUESTION ANSWERING SYSTEM")
    print("=" * 70)

    try:

        agent = create_agent()

    except Exception as error:

        print()
        print("ERROR WHILE CREATING AGENT:")
        print(error)

        return

    print()
    print("System is ready.")
    print("Ask questions about the knowledge base.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            print("Exiting...")
            break

        if not question:

            print(
                "Please enter a question."
            )
            print()

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "Goodbye!"
            )

            break

        print()
        print("Assistant:")

        try:

            answer = agent.run(
                question
            )

            print(answer)

        except Exception as error:

            print(
                "Sorry, an error occurred "
                "while processing your question."
            )

            print(
                f"Details: {error}"
            )

        print()


# ============================================================
# 19. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()