# Data Science Side Projects  

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)  
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)  
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4+-f7931e.svg)  
![Status](https://img.shields.io/badge/Projects-Mixed%20Complete%20%2F%20In--Progress-yellow.svg)  

This repository collects exploratory and applied projects spanning **information retrieval**, **multi-agent orchestration**, **deep learning architectures**, **classical ML**, **workflow automation**, and **algorithmic tinkering**.  

Some projects are polished end-to-end systems; others are **in-progress research prototypes** included to reflect the process of building and iterating.  

---

## 🔎 Retrieval-Augmented Generation (RAG) Pipeline
- Full pipeline with **web scraping, embeddings, PostgreSQL + pgvector, BM25 and HNSW indexes, hybrid retrieval**, and a **Gradio demo**.  
- Supports **dense, lexical, and hybrid search** with LLM-based answer generation.  
- **Status:** Fully functional, tested end-to-end.  

---

## 🤖 Multi-Agent System
- Multi-agent orchestration combining RAG with **external APIs (NBA stats)**.  
- Implements **Coordinator, Scraper, Embedding, and RAG agents**, plus domain-specific agents (Stats, Efficiency).  
- **Status:** Fully working demo with group chat protocol and function registration.  

---

## 📈 Classical ML & DL Experiments
- **FastICA**: Manual implementation compared against `sklearn.FastICA`; validated using my own EEG/ERP data (kurtosis, Q_Q plots) and against MATLAB EEGLAB ICA weights.
- **Neural network from scratch**: Forward/backward pass and Adam optimizer, tested on **Iris dataset**.  
- **Elman RNN**: Forward dynamics implemented; **backpropagation still under progress**, left as an exploratory prototype for furture work.  
- **Seq2Seq with Attention**: Custom **Encoder–Decoder LSTM** with explicit **Attention layer**. Includes training loop, batching, gradient clipping, and teacher forcing.<br>
> **✅ STATUS**: Mixed—some complete (NN, ICA), some exploratory (RNN, Seq2Seq) and not yet tested on full datasets; scaffolded for extension.  

---

## 📬 Workflow Automation
- **Automated Gmail sender**:  
  - Authenticates via Google API.  
  - Pulls recent drafts, converts to Word docs, generates HTML email bodies, attaches images, and **auto-submits to newspaper addresses**.
- **Automated Google Sheet for Clinical Management**:
  - Auto calculates pill counts per week and flags overdose.
  - Integrates **command line** function within sheet.
  - Displays and removes images on command.
> **✅ STATUS**: Fully functional productivity script.  

---

## 🔐 Algorithm & Math Explorations
- **SHA-256 implemented from scratch**: bitwise ops, padding, chaining.  
- **Elliptic curve arithmetic**: exploratory implementation of point addition and doubling.  
- Other math/crypto experiments recorded in notebooks.  
- **Status:** Open-ended exploration, intentionally unfinished in parts.  

---

## 🗂 Repository Structure



---

## 🚧 Note on Project Status
Not every project here is fully polished. Some (RAG, automation) are production-ready demos, while others (Seq2Seq, RNN) are partial implementations included to demonstrate algorithmic curiosity and technical scaffolding. Together, they highlight **breadth, depth, and process**—not just final results.
