from src.vectorstore.store import get_vectorstore

def query_knowledge_base(query: str, k: int = 2):
    """Retrieves relevant document chunks for a user prompt."""
    vector_db = get_vectorstore()
    results = vector_db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
