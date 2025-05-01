# src/rag_pipeline.py

from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain # https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html#create-retrieval-chain

def format_graph_info(records):
    return "\n".join([f"{r['from']} -[{r['relationship']}]-> {r['to']}" for r in records])

def build_offline_chatbot(content, model_name="llama3.2:latest"):
    # 1. Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(content)

    # 2. Load local embedding model
    embedder = HuggingFaceEmbeddings(model_name="./local_models/offline_model")

    # 3. Create retriever (in-memory FAISS)
    vectorstore = FAISS.from_texts(chunks, embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Define prompt template
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""
            You are an expert assistant for petroleum geology software.
            Only use the following content to answer the question.
            If the answer is not in the content, say you don't know.

            **Content**:
            {context}

            **Question**:
            {input}

            **Answer**:
        """
    )

    # 5. Load local LLM (via Ollama)
    llm = ChatOllama(model=model_name, temperature=0, num_predict=150)

    # 6. Build document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 7. Build retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain