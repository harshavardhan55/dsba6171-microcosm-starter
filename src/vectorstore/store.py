from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_vectorstore(persist_dir: str = "./chroma_db"):
    """Initializes and returns a ChromaDB vectorstore instance."""
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=persist_dir, embedding_function=embedding_function
    )


def add_texts_to_store(texts: list, persist_dir: str = "./chroma_db"):
    """Adds raw string texts directly to ChromaDB."""
    vector_db = get_vectorstore(persist_dir)
    vector_db.add_texts(texts)
    return vector_db