
# 🧠 Graph-Based RAG QA System with LangChain + Neo4j + Groq

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using **Neo4j** as a **graph and vector database**, combined with **LangChain**, **Groq LLMs**, and **Wikipedia data** to perform **knowledge graph-driven question answering** (QA).

---

## 📌 Overview

We combine **structured knowledge** (graph-based) and **unstructured knowledge** (text embeddings) to answer user questions with rich context. Using Wikipedia as the data source, we:

* Build a **Knowledge Graph** using LLMs
* Store it in **Neo4j** with relationships and node embeddings
* Perform **semantic search** + **entity-based reasoning**
* Visualize the knowledge graph interactively
* Answer complex natural language queries using an LLM

---

## 🔎 What is Neo4j and Why Use It?

**Neo4j** is a **graph database** that stores data as nodes and relationships, instead of rows and columns like SQL. It allows:

* **Efficient querying of connected data**
* **Relationship-centric reasoning** (e.g., find how two people are connected)
* **Graph-based semantic search** and link prediction
* Integration with **vector search**, enabling hybrid search (text + graph)

In this project:

* We use **Neo4j** to store documents and extracted entity relationships
* We use **vector search in Neo4j** for LLM-based semantic retrieval
* We use **Cypher queries** to fetch connected entities and visualize them

---

## 🔧 Tech Stack

| Tool                                  | Purpose                                 |
| ------------------------------------- | --------------------------------------- |
| **Neo4j**                             | Graph + vector database                 |
| **LangChain**                         | LLM pipeline framework                  |
| **Groq**                              | High-performance LLM provider           |
| **HuggingFace Sentence-Transformers** | Text embedding generation               |
| **WikipediaLoader**                   | Document ingestion                      |
| **LLMGraphTransformer**               | Extract KG triplets from text           |
| **Pyvis**                             | Interactive network graph visualization |
| **Pydantic**                          | Entity schema validation                |

---

## ⚙️ What This Code Does

### ✅ Step-by-step Workflow:

1. **Load Wikipedia Document**
   Using `WikipediaLoader`, we pull content for a given query (e.g., "Queen Elizabeth").

2. **Split Text for Processing**
   Use `TokenTextSplitter` to split long content into manageable chunks for LLM input.

3. **Generate Graph from Text**
   Using `LLMGraphTransformer`, the text chunks are converted into **triplets** (`subject -> predicate -> object`) and added to Neo4j as nodes and relationships.

4. **Store Embeddings in Neo4j**
   Each text chunk is embedded using `sentence-transformers` and stored in Neo4j for **vector-based similarity search**.

5. **Extract Entities from User Query**
   The LLM (`ChatGroq`) extracts key entities like persons or organizations from the query (e.g., "Queen Elizabeth").

6. **Hybrid Retrieval (Structured + Unstructured)**

   * Use Neo4j **full-text + graph traversal** to retrieve structured info
   * Use **vector similarity** to fetch unstructured relevant documents
   * Combine both into a single context

7. **Question Rewriting (Optional)**
   If there's chat history, the question is rewritten to be standalone using the LLM.

8. **Answer Generation**
   Final prompt with both structured and unstructured context is sent to the LLM for answer generation.

9. **Graph Visualization**
   Graph is rendered using `pyvis` for visual exploration of entity connections.

---

## 📊 Example Output

**Input**

> *"Which house does Elizabeth belong to?"*

**LLM Output**

> *"Elizabeth belongs to the House of Windsor, a royal house of the United Kingdom."*

**Graph View**
Shows `Elizabeth -> BELONGS_TO -> House of Windsor` and other relations like `MENTIONS`, `BORN_IN`, etc.

---

## 📁 Folder Structure

```bash
├── graph_qa_pipeline.py   # Main pipeline logic
├── requirements.txt       # Python dependencies
├── graph.html             # Pyvis interactive graph output
├── README.md              # This file
```

---

## 🔑 How to Run

1. ✅ Install dependencies

```bash
pip install -r requirements.txt
```

2. ✅ Start Neo4j locally (or via Docker)
   Ensure it runs on `neo4j://127.0.0.1:7687` with appropriate credentials.

3. ✅ Replace API key
   Update your Groq API key in the script.

4. ✅ Run the script

```bash
python graph_qa_pipeline.py
```

5. ✅ View the graph
   Open `graph.html` to explore the knowledge graph.

---

## 📈 Potential Use Cases

* ✅ Personalized Knowledge Agents
* ✅ Graph-Aware Chatbots
* ✅ Research Assistants
* ✅ Relationship Mining
* ✅ Biomedical or Legal Domain Reasoning

---

## 🤝 Contributions Welcome

If you have ideas to extend this project (e.g., using LangGraph, streamlit-based frontend, multi-hop reasoning, agent integration), feel free to fork or raise a PR!

---

## 📬 Connect

Feel free to reach out if you're exploring similar ideas or want to collaborate.


