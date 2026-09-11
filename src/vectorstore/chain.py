import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)


def get_rag_chain(persist_dir: str = "./chroma_db"):
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
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Retrieve token safely
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. Please export your Hugging Face token."
        )

    # 4. Initialize Hugging Face LLM endpoint using correct token parameter
    llm_engine = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.1,
        huggingfacehub_api_token=token,
    )
    llm = ChatHuggingFace(llm=llm_engine)

    # 5. Construct RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain