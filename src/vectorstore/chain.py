import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint


def get_rag_chain(persist_dir: str = "./chroma_db"):
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_dir, embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    template = """Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain