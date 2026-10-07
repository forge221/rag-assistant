import hashlib
import re
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import ollama
import pymupdf
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---- Settings ----
DOCS_FOLDER = "docs"
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
CACHE_DIR = Path("cache")
DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 5
NUM_CANDIDATES = 20
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MMR_LAMBDA = 0.7  # 1.0 = pure relevance (MMR off), lower = more variety
NUM_CTX = 2048
MAX_HISTORY_TURNS = 2
CACHE_VERSION = 3  # bump this if you change how chunking or embedding works


def list_document_files(folder: str) -> list[Path]:
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    # Rejoin words that were split across lines with a hyphen ("intelli-\ngence")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove junk patterns FIRST, so lines left nearly empty get dropped below
    text = re.sub(r"https?://\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)                 # email addresses
    text = re.sub(r"\[\d+(?:[,\-–]\s*\d+)*\]", "", text)     # citations like [20] or [3, 4]
    text = re.sub(r"[_=\-*]{4,}", "", text)                  # separator lines like ______

    # Fix stray spaces before hyphens ("cost -effective" -> "cost-effective")
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)

    lines = [line.strip() for line in text.splitlines()]

    # Replace numbers with "#" when counting, so headers that differ only
    # by page number ("... 356", "... 357") count as the same line
    def header_key(line: str) -> str:
        return re.sub(r"\d+", "#", line)

    line_counts = Counter(header_key(line) for line in lines if line)

    kept_lines = []
    for line in lines:
        if not line:
            continue
        # Repeated short lines are usually page headers or footers
        if line_counts[header_key(line)] >= 3 and len(line) < 100:
            continue
        # Lines that are only a number are usually page numbers
        if re.fullmatch(r"\d+", line):
            continue
        # Lines that are mostly symbols or numbers are usually table or figure junk
        letters = sum(ch.isalpha() for ch in line)
        if letters / len(line) < 0.5:
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def load_documents(folder: str) -> list[dict]:
    cleaned_dir = CACHE_DIR / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for path in list_document_files(folder):
        try:
            raw_text = read_document(path)
        except Exception as error:
            print(f"Skipping {path.name}: couldn't read it ({error})")
            continue

        text = clean_text(raw_text)

        if not text.strip():
            print(f"Warning: no text found in {path.name}. It may be a scanned PDF.")
            continue

        (cleaned_dir / f"{path.name}.txt").write_text(text, encoding="utf-8")

        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append({"text": chunk, "source": path.name})
    return all_chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings


def compute_fingerprint(folder: str) -> str:
    hasher = hashlib.sha256()

    settings = f"{CACHE_VERSION}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_MODEL}"
    hasher.update(settings.encode("utf-8"))

    for path in list_document_files(folder):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())

    return hasher.hexdigest()


def load_or_build_collection(model: SentenceTransformer, folder: str):
    CACHE_DIR.mkdir(exist_ok=True)
    fingerprint_file = CACHE_DIR / "fingerprint.txt"
    current_fingerprint = compute_fingerprint(folder)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if (
        fingerprint_file.exists()
        and fingerprint_file.read_text() == current_fingerprint
        and collection.count() > 0
    ):
        print("Loading saved database...")
        return collection

    print("Documents or settings changed (or no database yet). Building database...")
    client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    chunks = load_documents(folder)
    embeddings = embed_chunks(chunks, model)

    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )

    fingerprint_file.write_text(current_fingerprint)
    return collection


def search(question: str, collection, model: SentenceTransformer, top_k: int):
    question_vector = model.encode(question, normalize_embeddings=True)

    results = collection.query(
        query_embeddings=[question_vector.tolist()],
        n_results=top_k,
    )

    found = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = 1 - distance / 2
        found.append(({"text": text, "source": metadata["source"]}, score))
    return found


def rerank(question: str, candidates: list, reranker: CrossEncoder, top_k: int):
    for rank, (chunk, vector_score) in enumerate(candidates, start=1):
        chunk["vector_rank"] = rank
        chunk["vector_score"] = vector_score

    pairs = [(question, chunk["text"]) for chunk, vector_score in candidates]
    rerank_scores = reranker.predict(pairs)

    scored = [
        (chunk, float(rerank_score))
        for (chunk, vector_score), rerank_score in zip(candidates, rerank_scores)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    for rank, (chunk, rerank_score) in enumerate(scored, start=1):
        chunk["rerank_rank"] = rank

    return scored[:top_k]


def mmr_select(ranked: list, model: SentenceTransformer, top_k: int, lambda_: float):
    if len(ranked) <= top_k:
        return ranked

    texts = [chunk["text"] for chunk, score in ranked]
    vectors = model.encode(texts, normalize_embeddings=True)
    similarity = vectors @ vectors.T

    scores = np.array([score for chunk, score in ranked])
    relevance = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    selected = [0]
    remaining = list(range(1, len(ranked)))

    while len(selected) < top_k:
        best_index = None
        best_value = float("-inf")
        for i in remaining:
            redundancy = max(similarity[i][j] for j in selected)
            value = lambda_ * relevance[i] - (1 - lambda_) * redundancy
            if value > best_value:
                best_index = i
                best_value = value
        selected.append(best_index)
        remaining.remove(best_index)

    return [ranked[i] for i in selected]


def rewrite_question(question: str, history: list[dict]) -> str:
    if not history:
        return question

    conversation = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history
    )

    prompt = f"""Here is a conversation and a follow-up question.
Rewrite the follow-up question so it makes sense on its own, without the conversation.
Replace words like "it", "that", or "they" with what they refer to.
If the question already makes sense on its own, return it unchanged.
Reply with only the rewritten question and nothing else.

Conversation:
{conversation}

Follow-up question: {question}

Rewritten question:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": NUM_CTX},
    )
    return response["message"]["content"].strip()


def generate_answer(question: str, results: list, history: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk, score in results
    )

    system_message = """You are a helpful assistant that answers questions using provided document excerpts.
Base your answer on the excerpts. If they only partly answer the question, give the best answer you can from what they contain.
Only say you couldn't find the answer if the excerpts are completely unrelated to the question.
At the end, list the source files you used."""

    messages = [{"role": "system", "content": system_message}]

    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    user_message = f"""Document excerpts:
{context}

Question: {question}"""
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"num_ctx": NUM_CTX},
    )
    return response["message"]["content"]


model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

collection = load_or_build_collection(model, DOCS_FOLDER)
print(f"Ready! ({collection.count()} chunks in database)")
print("Type 'reset' to start a new conversation, or 'quit' to exit.")

history = []

while True:
    question = input("\nAsk a question: ")

    if question.lower() == "quit":
        break
    if question.lower() == "reset":
        history = []
        print("Conversation cleared.")
        continue

    search_query = rewrite_question(question, history)
    if search_query != question:
        print(f"(Searching for: {search_query})")

    candidates = search(search_query, collection, model, NUM_CANDIDATES)
    reranked = rerank(search_query, candidates, reranker, NUM_CANDIDATES)
    results = mmr_select(reranked, model, TOP_K, MMR_LAMBDA)

    print("\nThinking...")
    answer = generate_answer(question, results, history)

    print(f"\nAnswer:\n{answer}")
    print("\nRetrieved from:")
    for chunk, rerank_score in results:
        print(
            f"  - {chunk['source']} | rerank #{chunk['rerank_rank']} ({rerank_score:.2f}) | "
            f"vector #{chunk['vector_rank']} ({chunk['vector_score']:.3f})"
        )