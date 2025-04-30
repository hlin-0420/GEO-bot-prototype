from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def format_graph_info(records):
    return "\n".join([f"{r['from']} -[{r['relationship']}]-> {r['to']}" for r in records])

def build_neo4j_rag_pipeline(uri, username, password, model_name="llama3.2:latest"):
    graph = Neo4jGraph(url=uri, username=username, password=password)

    cypher_prompt = PromptTemplate(
        template="""
        GEO is a well log authoring, analysis, and reporting system for petroleum geologists, geoscientists, and engineers.  
        Use the following **graph context** to answer the question. Do not use outside knowledge.

        **Graph Context:**
        {graph_context}

        **Question:** {question}

        **Answer:**
        """,
        input_variables=["question", "graph_context"]
    )

    llm = ChatOllama(model=model_name, temperature=0, num_predict=150)
    rag_chain = cypher_prompt | llm | StrOutputParser()

    return rag_chain, graph
