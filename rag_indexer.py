# Этот файл делает подготовку базы знаний для поиска по документам

# import os
from math import nan
from pathlib import Path                        # удобная работа с путями и файлами

import chromadb                                 # векторная база данных
from chromadb.utils import embedding_functions  # функция для превращения текста в векторы
from langchain_text_splitters import RecursiveCharacterTextSplitter # разбивает длинный текст на чанки
from pypdf import PdfReader                     # читает PDF
from docx import Document                       # читает DOCX


DOCS_DIR = "docs"                               # папка, откуда брать документы
CHROMA_DIR = "chroma_db"                        # папка, где будет храниться база Chroma
COLLECTION_NAME = "oil_gas_docs"                # имя коллекции внутри Chroma


# Чтение PDF
def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)

        # Если PDF зашифрован, пробуем открыть пустым паролем
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass

        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)

        return "\n".join(pages_text)

    except Exception as e:
        print(f"Ошибка чтения PDF {file_path}: {e}")
        return ""


# Чтение DOCX
def read_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Ошибка чтения DOCX {file_path}: {e}")
        return ""


# Универсальное чтение файла
def read_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    print(file_path)

    if suffix == ".pdf":
        return read_pdf(str(file_path))
    if suffix == ".docx":
        return read_docx(str(file_path))
    
    return ""


# Загрузка всех документов из папки docs
def load_documents_from_docs() -> list[dict]:
    docs = []
    docs_path = Path(DOCS_DIR)

    if not docs_path.exists():
        print(f"Папка {DOCS_DIR} не найдена.")
        return docs

    # Проход по всем файлам в папке docs.
    for file_path in docs_path.glob("*"):
        if file_path.suffix.lower() not in [".pdf", ".docx"]:
            continue
        
        # читается текст
        text = read_file(file_path)
        # Если текст пустой — файл пропускается
        if not text.strip():
            # print(f"Файл пустой или текст не извлечён: {file_path.name}")
            continue

        # Текст добавляется в список в виде словаря
        docs.append({
            "source": file_path.name,
            "text": text,
        })

    return docs


def build_index():
    # Загружаются документы
    documents = load_documents_from_docs()

    # Если документов нет — функция завершается
    if not documents:
        print("Нет документов для индексации.")
        return

    # Разбиение текста на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ".", ";", ":", ",", " ", ""],
    )

    # Подключение к Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Функция эмбеддингов
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Создание или получение коллекции
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # Чтобы не копить дубликаты, очищаем коллекцию перед полной переиндексацией
    existing_count = collection.count()
    if existing_count > 0:
        all_items = collection.get()
        existing_ids = all_items.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

    # Подготовка данных для записи
    all_chunks = []
    all_ids = []
    all_metadatas = []

    # Разбиение документов на чанки
    idx = 0
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{idx}")
            all_metadatas.append({"source": doc["source"]})
            idx += 1

    if all_chunks:
        collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadatas,
        )

    print(f"Готово. Проиндексировано чанков: {len(all_chunks)}")


if __name__ == "__main__":
    build_index()