import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint


def get_rag_chain(persist_dir: str = "./chroma_db"):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    # 1. Initialize retriever
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_dir, embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. Define prompt template
    template = """Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate.from_template(template)

    # 3. Initialize Hugging Face LLM endpoint
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature=0.1,
        huggingfacehub_api_token=token,
    )

    # 4. Construct RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain