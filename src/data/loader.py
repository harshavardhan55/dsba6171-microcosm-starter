from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_document(file_path: str = "data/structured/raw/synthetic_knowledge_items.csv"):
    """Loads a text, PDF, or CSV file and splits it into manageable chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(
            file_path=file_path,
            content_columns=["ki_topic", "ki_text"],
            metadata_columns=["ki_id"],
            encoding="utf-8"
        )
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)
