from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Для чтения DOCX и PDF
import docx
from pypdf import PdfReader



KB_DIR = Path(__file__).with_name("docs")  # папка с документами
CHROMA_DIR = Path(__file__).with_name("chroma_db")
COLLECTION_NAME = "gaz_docs"
FINGERPRINT_PATH = CHROMA_DIR / "kb_fingerprint.sha256"


def _read_docx(file_path: Path) -> list[tuple[str, int]]:
    """Возвращает список (текст, номер 'страницы' = блок), т.к. docx постранично не делится надёжно."""
    doc = docx.Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs])
    return [(full_text, 1)] if full_text.strip() else []




def _read_pdf(file_path: Path) -> list[tuple[str, int]]:
    """Возвращает список (текст страницы, номер страницы)."""
    out: list[tuple[str, int]] = []

    reader = PdfReader(file_path)

    for idx, page in enumerate(reader.pages):
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""

        if txt and txt.strip():
            out.append((txt, idx + 1))

    return out


def _iter_documents() -> list[dict]:
    """
    Собирает по каждому файлу списки чанков с метаданными:
    {"text": ..., "source": имя файла, "page": N или None}.
    """
    if not KB_DIR.exists() or not any(KB_DIR.iterdir()):
        raise FileNotFoundError(
            f"Не найдены документы в папке: {KB_DIR}. "
            f"Добавьте туда txt, docx или pdf файлы."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ".", "!", "?", ";", ":", ""],
    )

    docs: list[dict] = []
    for file in sorted(KB_DIR.iterdir(), key=lambda p: p.name):
        if not file.is_file():
            continue
        try:
            file_lower = file.suffix.lower()
            per_pages: list[tuple[str, int | None]] = []
            if file_lower == ".txt":
                text = file.read_text(encoding="utf-8")
                per_pages = [(text, None)]
            elif file_lower == ".docx":
                per_pages = _read_docx(file)
            elif file_lower == ".pdf":
                per_pages = _read_pdf(file)
            else:
                continue

            for page_text, page_num in per_pages:
                if not page_text.strip():
                    continue
                chunks = [c for c in splitter.split_text(page_text) if c and c.strip()]
                for local_idx, chunk in enumerate(chunks):
                    docs.append(
                        {
                            "text": chunk,
                            "source": file.name,
                            "page": page_num,
                            "chunk_index": local_idx,
                        }
                    )
        except Exception:
            # Один проблемный файл не должен ломать индексацию всей базы
            continue

    return docs


def _compute_fingerprint() -> str:
    """
    Отпечаток набора документов: имена + размер + mtime.
    Нужен, чтобы автоматически переиндексировать при обновлении файлов.
    """
    h = hashlib.sha256()
    if not KB_DIR.exists():
        return h.hexdigest()
    for p in sorted([x for x in KB_DIR.iterdir() if x.is_file()], key=lambda x: x.name):
        st = p.stat()
        h.update(p.name.encode("utf-8", errors="ignore"))
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    russian_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=russian_ef)

    # Переиндексация, если документы изменились (или коллекция пуста).
    current_fp = _compute_fingerprint()
    stored_fp = None
    try:
        if FINGERPRINT_PATH.exists():
            stored_fp = FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        stored_fp = None

    needs_reindex = (collection.count() == 0) or (stored_fp != current_fp)
    if needs_reindex:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=russian_ef)

        docs = _iter_documents()
        if docs:
            documents = [d["text"] for d in docs]
            ids = [f"{d['source']}::p{d['page'] or 0}::c{d['chunk_index']}" for d in docs]
            metadatas = [
                {
                    "source": d["source"],
                    "page": d["page"],
                    "chunk_index": d["chunk_index"],
                }
                for d in docs
            ]
            # Chroma имеет лимит на размер батча — добавляем порциями
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                collection.add(
                    documents=documents[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                )
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            FINGERPRINT_PATH.write_text(current_fp, encoding="utf-8")
        except Exception:
            pass

    return collection


def search_docs(query: str, n_results: int = 45) -> list[str]:
    """Вернёт список релевантных фрагментов текста из базы знаний (без метаданных)."""
    docs, _, _ = search_docs_with_scores(query=query, n_results=n_results)
    return docs


def search_docs_with_scores(query: str, n_results: int = 45) -> tuple[list[str], list[float], list[dict]]:
    """
    Вернёт (docs, distances, metadatas). Чем меньше distance — тем ближе (зависит от метрики коллекции).
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return docs, distances, metadatas
