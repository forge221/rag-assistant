import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import ollama
import pymupdf
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---- Settings ----
APP_NAME = "RAG Assistant"
DEFAULT_DATA_DIR = Path.home() / APP_NAME
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}
COLLECTION_NAME = "documents"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama3.2"
NUM_CANDIDATES = 20
TOP_K = 5
MMR_LAMBDA = 0.7  # 1.0 = pure relevance (MMR off), lower = more variety
NUM_CTX = 2048
MAX_HISTORY_TURNS = 2
CACHE_VERSION = 3  # bump this if you change how chunking or embedding works

SYSTEM_MESSAGE = """You are a helpful assistant that answers questions using provided document excerpts.
Base your answer on the excerpts. If they only partly answer the question, give the best answer you can from what they contain.
Only say you couldn't find the answer if the excerpts are completely unrelated to the question.
At the end, list the source files you used."""


# ---- Text processing (plain functions, no engine state needed) ----

def list_document_files(folder: Path) -> list[Path]:
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

    # Remove junk patterns first, so lines left nearly empty get dropped below
    text = re.sub(r"https?://\S+|www\.\S+", "", text)       # URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)                 # email addresses
    text = re.sub(r"\[\d+(?:[,\-–]\s*\d+)*\]", "", text)     # citations like [20] or [3, 4]
    text = re.sub(r"[_=\-*]{4,}", "", text)                  # separator lines like ______

    # Fix stray spaces before hyphens ("cost -effective" -> "cost-effective")
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)

    lines = [line.strip() for line in text.splitlines()]

    def header_key(line: str) -> str:
        return re.sub(r"\d+", "#", line)

    line_counts = Counter(header_key(line) for line in lines if line)

    kept_lines = []
    for line in lines:
        if not line:
            continue
        if line_counts[header_key(line)] >= 3 and len(line) < 100:
            continue
        if re.fullmatch(r"\d+", line):
            continue
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


# ---- The engine ----

class RagEngine:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, status=print):
        self.data_dir = Path(data_dir)
        self.docs_dir = self.data_dir / "docs"
        self.cache_dir = self.data_dir / "cache"
        self.db_dir = self.data_dir / "chroma_db"

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.status = status
        self.embedder = None
        self.reranker = None
        self.client = None
        self.collection = None
        self.history = []

    # ---- Public methods (what an interface calls) ----

    def load(self):
        self.status("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.status("Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.refresh_index()

    def check_ollama(self) -> tuple[bool, str]:
        try:
            ollama.show(LLM_MODEL)
            return True, "Ollama is ready."
        except ollama.ResponseError:
            return False, (
                f"Ollama is running, but the model '{LLM_MODEL}' isn't installed.\n"
                f"Open a terminal and run:  ollama pull {LLM_MODEL}"
            )
        except Exception:
            return False, (
                "Couldn't connect to Ollama.\n"
                "Install it from ollama.com, then make sure it's running."
            )

    def document_names(self) -> list[str]:
        return [path.name for path in list_document_files(self.docs_dir)]

    def add_documents(self, paths) -> list[str]:
        added = []
        for path in map(Path, paths):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                shutil.copy2(path, self.docs_dir / path.name)
                added.append(path.name)
        if added:
            self.refresh_index()
        return added

    def reset_conversation(self):
        self.history = []

    def ask(self, question: str) -> dict:
        if self.collection is None or self.collection.count() == 0:
            return {
                "answer": "There are no documents yet. Add some documents first.",
                "sources": [],
                "search_query": question,
            }

        search_query = self._rewrite_question(question)
        candidates = self._search(search_query, NUM_CANDIDATES)
        reranked = self._rerank(search_query, candidates)
        results = self._mmr_select(reranked, TOP_K)
        answer = self._generate_answer(question, results)

        self.history.append({"question": question, "answer": answer})
        self.history = self.history[-MAX_HISTORY_TURNS:]

        sources = [
            {
                "source": chunk["source"],
                "rerank_rank": chunk["rerank_rank"],
                "rerank_score": score,
                "vector_rank": chunk["vector_rank"],
                "vector_score": chunk["vector_score"],
            }
            for chunk, score in results
        ]
        return {"answer": answer, "sources": sources, "search_query": search_query}

    # ---- Index building ----

    def refresh_index(self):
        fingerprint_file = self.cache_dir / "fingerprint.txt"
        current_fingerprint = self._compute_fingerprint()

        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

        if (
            fingerprint_file.exists()
            and fingerprint_file.read_text() == current_fingerprint
            and self.collection.count() > 0
        ):
            self.status(f"Loaded saved database ({self.collection.count()} chunks).")
            return

        self.status("Reading documents...")
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.create_collection(name=COLLECTION_NAME)

        chunks = self._load_documents()
        if chunks:
            self.status(f"Embedding {len(chunks)} chunks...")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.collection.add(
                ids=[f"chunk-{i}" for i in range(len(chunks))],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[{"source": chunk["source"]} for chunk in chunks],
            )

        fingerprint_file.write_text(current_fingerprint)
        self.status(f"Database ready ({self.collection.count()} chunks).")

    def _compute_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        settings = f"{CACHE_VERSION}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_MODEL}"
        hasher.update(settings.encode("utf-8"))
        for path in list_document_files(self.docs_dir):
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    def _load_documents(self) -> list[dict]:
        cleaned_dir = self.cache_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = []
        for path in list_document_files(self.docs_dir):
            try:
                raw_text = read_document(path)
            except Exception as error:
                self.status(f"Skipping {path.name}: couldn't read it ({error})")
                continue

            text = clean_text(raw_text)
            if not text.strip():
                self.status(f"Warning: no text found in {path.name}. It may be a scanned PDF.")
                continue

            (cleaned_dir / f"{path.name}.txt").write_text(text, encoding="utf-8")

            for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append({"text": chunk, "source": path.name})
        return all_chunks

    # ---- Retrieval ----

    def _search(self, question: str, top_k: int) -> list:
        top_k = min(top_k, self.collection.count())
        question_vector = self.embedder.encode(question, normalize_embeddings=True)
        results = self.collection.query(
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

    def _rerank(self, question: str, candidates: list) -> list:
        for rank, (chunk, vector_score) in enumerate(candidates, start=1):
            chunk["vector_rank"] = rank
            chunk["vector_score"] = vector_score

        pairs = [(question, chunk["text"]) for chunk, vector_score in candidates]
        rerank_scores = self.reranker.predict(pairs)

        scored = [
            (chunk, float(rerank_score))
            for (chunk, vector_score), rerank_score in zip(candidates, rerank_scores)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        for rank, (chunk, rerank_score) in enumerate(scored, start=1):
            chunk["rerank_rank"] = rank
        return scored

    def _mmr_select(self, ranked: list, top_k: int) -> list:
        if len(ranked) <= top_k:
            return ranked

        texts = [chunk["text"] for chunk, score in ranked]
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
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
                value = MMR_LAMBDA * relevance[i] - (1 - MMR_LAMBDA) * redundancy
                if value > best_value:
                    best_index = i
                    best_value = value
            selected.append(best_index)
            remaining.remove(best_index)

        return [ranked[i] for i in selected]

    # ---- Generation ----

    def _rewrite_question(self, question: str) -> str:
        if not self.history:
            return question

        conversation = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in self.history
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

    def _generate_answer(self, question: str, results: list) -> str:
        context = "\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk, score in results
        )

        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        for turn in self.history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({
            "role": "user",
            "content": f"Document excerpts:\n{context}\n\nQuestion: {question}",
        })

        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options={"num_ctx": NUM_CTX},
        )
        return response["message"]["content"]