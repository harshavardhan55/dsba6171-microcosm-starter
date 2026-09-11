from src.vectorstore.chain import get_rag_chain
from src.vectorstore.store import add_texts_to_store

if __name__ == "__main__":
    sample_texts = [
        "ChromaDB stores high-dimensional vector embeddings locally.",
        "LangChain allows building LLM applications via composable chains.",
        "RAG combines document retrieval with text generation.",
    ]
    print("--- Ingesting Sample Data ---")
    add_texts_to_store(sample_texts)
    print("Data ingested successfully.\n")

    print("--- Running RAG Chain ---")
    query = "What does ChromaDB do?"
    chain = get_rag_chain()
    response = chain.invoke(query)

    print(f"Query: {query}")
    print(f"LLM Response:\n{response}")