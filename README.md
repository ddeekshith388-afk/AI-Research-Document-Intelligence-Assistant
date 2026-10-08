# 🤖 AI Research & Document Intelligence Assistant

An AI-powered **Research and Document Intelligence Assistant** that allows users to upload documents, ask questions, retrieve relevant information, and receive accurate, context-grounded answers using Generative AI.

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline using **Python, LangChain, LangGraph, Google Gemini, ChromaDB, Streamlit, FastAPI/LangServe, and LangSmith**.

---

## 🎯 Project Objective

The main objective of this project is to develop an intelligent document assistant that can understand user-uploaded documents and answer questions based specifically on their content.

Traditional Large Language Models may sometimes generate incorrect or unsupported information because they rely on their pre-trained knowledge.

To address this problem, this project uses **Retrieval-Augmented Generation (RAG)**.

The system first retrieves relevant information from the uploaded documents and then provides that information as context to the Gemini LLM.

This approach helps to:

- Reduce hallucinations
- Improve answer accuracy
- Provide document-grounded responses
- Retrieve relevant information efficiently
- Build a practical Generative AI application

---

## 🚀 Key Features

- 📄 Upload and process documents
- ✂️ Text extraction and document chunking
- 🧠 Generate embeddings for document chunks
- 🗄️ Store embeddings in ChromaDB
- 🔎 Semantic similarity-based document retrieval
- 🤖 Google Gemini for answer generation
- 🔗 Retrieval-Augmented Generation using LangChain
- 🔄 Workflow orchestration using LangGraph
- 🛡️ Context-grounded answer generation
- ✅ Grounding check for generated responses
- 📊 LangSmith tracing and monitoring
- 🌐 Interactive Streamlit interface
- ⚡ API-ready architecture using FastAPI and LangServe

---

## 🧩 How the System Works

The application follows an end-to-end RAG workflow.

```text
User Uploads Document
        ↓
Document Loading
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Embedding Generation
        ↓
ChromaDB Vector Storage
        ↓
User Asks Question
        ↓
Similarity Search
        ↓
Relevant Document Chunks
        ↓
Gemini LLM
        ↓
Grounding Check
        ↓
Final Answer
