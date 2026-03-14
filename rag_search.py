# Этот файл ищет по локальной базе

import chromadb                                     # клиент для работы с ChromaDB
from chromadb.utils import embedding_functions      # готовые функции эмбеддингов

# Константы
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "oil_gas_docs"


# Функция открывает коллекцию Chroma
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

# Функция делает из найденных чанков один текстовый блок для передачи в LLM
def format_rag_context(documents, metadatas=None, distances=None) -> str:
    parts = []

    for i, doc in enumerate(documents, 1):
        source = ""
        if metadatas and i - 1 < len(metadatas):
            source_name = metadatas[i - 1].get("source", "")
            if source_name:
                source = f" [источник: {source_name}]"

        parts.append(f"Фрагмент {i}{source}: {doc}")

    return "\n\n".join(parts) if parts else "Локальная информация не найдена."

# Основная функция локального поиска
def get_rag_result(user_query: str, k: int = 4, distance_threshold: float = 1.2) -> dict:
    try:
        # Функция подключается к коллекции
        collection = get_collection()
        
        # Проверяем, есть ли вообще что-то в коллекции
        if collection.count() == 0:
            return {
                "source": "none",
                "context": "Локальная база пуста. Документы ещё не проиндексированы.",
                "reason": "empty_collection",
                "best_distance": None,
            }

        results = collection.query(
            query_texts=[user_query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Извлечение первого результата
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Очистка документов
        # cleaned_documents = [doc for doc in documents if doc and doc.strip()]

        valid_items = [
            (doc, meta, dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if doc and doc.strip()
        ]

        if not valid_items:
            return {
                "source": "none",
                "context": "Локальная информация не найдена.",
                "reason": "empty_documents",
                "best_distance": None,
            }

        cleaned_documents = [x[0] for x in valid_items]
        cleaned_metadatas = [x[1] for x in valid_items]
        cleaned_distances = [x[2] for x in valid_items]

        

        # # Если после очистки ничего не осталось
        # if not cleaned_documents:
        #     return {
        #         "source": "none",
        #         "context": "Локальная информация не найдена.",
        #         "reason": "empty_documents",
        #         "best_distance": None,
        #     }

        # Вычисление лучшей дистанции
        best_distance = min(cleaned_distances) if cleaned_distances else None

        # Если даже лучший фрагмент слишком далёк от запроса, результат отвергается
        if best_distance is not None and best_distance > distance_threshold:
            return {
                "source": "none",
                "context": "Локальная информация недостаточно релевантна.",
                "reason": "distance_too_high",
                "best_distance": best_distance,
            }

        # Если всё хорошо возвращаем RAG-контекст
        return {
            "source": "rag",
            "context": format_rag_context(cleaned_documents, cleaned_metadatas, cleaned_distances),
            "reason": "ok",
            "best_distance": best_distance,
        }

    except Exception as e:
        return {
            "source": "none",
            "context": f"Ошибка при поиске в локальной базе: {e}",
            "reason": "exception",
            "best_distance": None,
        }