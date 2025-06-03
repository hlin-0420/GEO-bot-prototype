# src/rag_pipeline.py

try:
    from langchain.prompts import PromptTemplate
    from langchain_ollama import ChatOllama
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain  # https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html#create-retrieval-chain
except ImportError:  # pragma: no cover - provide minimal fallbacks
    class PromptTemplate:
        def __init__(self, input_variables=None, template=""):
            self.input_variables = input_variables or []
            self.template = template

    class ChatOllama:
        def __init__(self, *_, **__):
            pass

    class FAISS:
        @classmethod
        def from_texts(cls, texts, _embedder):
            instance = cls()
            instance.texts = texts
            return instance

        def as_retriever(self, search_kwargs=None):
            return self

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100):
            self.chunk_size = chunk_size

        def split_text(self, text):
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    class HuggingFaceEmbeddings:
        def __init__(self, model_name=None):
            self.model_name = model_name

    def create_stuff_documents_chain(llm, prompt):
        class _DummyChain:
            def __init__(self, llm, prompt):
                self.llm = llm
                self.prompt = prompt

        return _DummyChain(llm, prompt)

    def create_retrieval_chain(retriever, document_chain):
        class _DummyRetriever:
            def invoke(self, data):
                return {"answer": "offline"}

        return _DummyRetriever()

def format_graph_info(records):
    return "\n".join([f"{r['from']} -[{r['relationship']}]-> {r['to']}" for r in records])

def build_offline_chatbot(content, model_name="llama3.2:1b"):
    # 1. Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(content)

    # 2. Load local embedding model
    embedder = HuggingFaceEmbeddings(model_name="./local_models/offline_model")

    # 3. Create retriever (in-memory FAISS)
    vectorstore = FAISS.from_texts(chunks, embedder)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # 4. Define prompt template
    prompt = PromptTemplate(
    input_variables=["context", "input"],
        template="""
    You are a highly knowledgeable assistant trained exclusively on GEO petroleum geology software documentation.

    Answer the user's question **using only the GEO documentation provided below**. If you cannot find the answer, respond with: "I don't know based on the provided documentation."

    Respond with **only the final answer**—do not repeat the question or the content.

    ---
    GEO Documentation:
    {context}

    User Question:
    {input}

    Answer:
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
