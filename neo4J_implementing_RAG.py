# ------------------- IMPORTS -------------------
import os
from typing import Tuple, List

from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)

from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector, remove_lucene_chars
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq

from neo4j import GraphDatabase
from pyvis.network import Network
from IPython.display import display, HTML
from pydantic import BaseModel, Field

# ------------------- ENVIRONMENT SETUP -------------------
# Set Neo4j credentials
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "gihanlakmal"
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

# ------------------- INITIALIZE GRAPH -------------------
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# ------------------- LOAD AND SPLIT DOCUMENTS -------------------
# Load Wikipedia documents
raw_document = WikipediaLoader(query="Elizabeth queen").load()

# Count characters in first 6 docs
total = sum(len(raw_document[i].page_content) for i in range(6))
print(f"Total number of characters in the first 6 documents: {total}")

# Split text into chunks
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_document[:3])

# ------------------- CONVERT TO GRAPH DOCUMENTS -------------------
llm = ChatGroq(
    api_key="your_groq_api_key_here",  # Replace with your real key
    model="deepseek-r1-distill-llama-70b"
)
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Add to Neo4j
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

# ------------------- VISUALIZE NEO4J GRAPH -------------------
def show_graph(cypher: str):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    session = driver.session()
    result = session.run(cypher)

    net = Network(notebook=True, height="600px", width="100%", bgcolor="#222222", font_color="white")
    nodes, edges = set(), []

    for record in result:
        for key in record.keys():
            value = record[key]
            if hasattr(value, 'element_id') and hasattr(value, 'labels'):
                node_id = str(value.element_id)
                if node_id not in nodes:
                    label = value.get("name") or value.get("title") or f"Node {node_id}"
                    net.add_node(node_id, label=label)
                    nodes.add(node_id)
            elif hasattr(value, 'start_node') and hasattr(value, 'end_node'):
                start_id = str(value.start_node.element_id)
                end_id = str(value.end_node.element_id)
                rel_type = value.type
                edges.append((start_id, end_id, rel_type))

    for start_id, end_id, rel_type in edges:
        if start_id in nodes and end_id in nodes:
            net.add_edge(start_id, end_id, label=rel_type)

    session.close()
    net.show("graph.html")
    display(HTML("graph.html"))

# Show sample graph
show_graph("MATCH (s)-[r:!MENTIONS]->(t) RETURN s, r, t LIMIT 50")

# ------------------- VECTOR INDEXING -------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_index = Neo4jVector.from_existing_graph(
    embedding=embedding,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# ------------------- ENTITY EXTRACTION MODEL -------------------
class Entities(BaseModel):
    names: List[str] = Field(..., description="All the person, organization, or business entities in the text")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}")
])
entity_chain = prompt | llm.with_structured_output(Entities)

# ------------------- STRUCTURED SEARCH -------------------
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return " AND ".join([f"{w}~2" for w in words])

def structured_retriever(question: str) -> list[str]:
    entities = entity_chain.invoke({"question": question})
    results = ""

    for entity in entities.names:
        response = graph.query("""
            CALL db.index.fulltext.queryNodes('keyword', $query, {limit: 2})
            YIELD node, score
            CALL {
                WITH node
                MATCH (node)-[r:MENTIONS]->(neighbor)
                RETURN node.id + ' -[' + type(r) + ']-> ' + neighbor.id AS output
                UNION
                MATCH (node)<-[r:MENTIONS]-(neighbor)
                RETURN neighbor.id + ' -[' + type(r) + ']-> ' + node.id AS output
            }
            RETURN output LIMIT 50
        """, {"query": generate_full_text_query(entity)})

        results += "\n".join([el["output"] for el in response])
    return results

# ------------------- UNIFIED RETRIEVER -------------------
def retriever(question: str):
    print(f"search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:\n{structured_data}\n\nUnstructured data:\n{"Document ".join(unstructured_data)}"""
    return final_data

# ------------------- QUESTION CONDENSER -------------------
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question:
Chat History:
{chat_history}Follow Up Input: {question}
Rephrased Question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(history):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        ) | CONDENSE_QUESTION_PROMPT | llm | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

# ------------------- FINAL QA CHAIN -------------------
qa_prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the context:\n{context}\nQuestion: {question}\nUse natural language and be concise.\nAnswer:"
)

chain = (
    RunnableParallel({
        "context": _search_query | retriever,
        "question": RunnablePassthrough()
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ------------------- EXAMPLE INFERENCE -------------------
print(chain.invoke({"question": "WHICH HOUSE DOES Elizabeth BELONGS TO ?"}))




