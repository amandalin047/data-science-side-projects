# Data Science Side Projects  

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-ff6f00.svg)  
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4+-f7931e.svg)<br>
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)    
![Google Apps Script](https://img.shields.io/badge/Google%20Apps%20Script-V8%2B-4285F4.svg)<br>
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

## 📈 Classical ML & DL Experiments/Explorations
- **FastICA**: Manual implementation **from sctatch with only `Numpy`**. Compared against `sklearn.FastICA`; tested on **my own EEG/ERP data** (colleced in 2022) using metrics such as kurtosis and Q_Q plots, then compared against `MATLAB` `EEGLAB` ICA weights.
- **3-layer Neural Network**: Forward and backward pass + Adam optimizer + dropout manually implemented **from scratch**, tested on `Iris` and `Wine` datasets for accuracy, training loss plot, and confusion matrix.
- **Elman RNN**: Forward dynamics implemented **from scratch**; _backpropagation still in progress_, left as an exploratory prototype for furture work.  
- **Seq2Seq with Attention**: Custom **Encoder–Decoder LSTM** with explicit **Attention layer**. Includes training loop, batching, gradient clipping, and teacher forcing; _dataset validation still in progress_.
- **CNN on ECG**: Exploratory CNN trained on the [ECG Heartbeat Categorization Dataset](https://share.google/kMg85rxKAcdgvHKUS); demonstrates feature extraction with convolution + pooling, with training loss/accuracy plots and confusion matrix for evaluation.
> **✅ STATUS**: Mixed — some complete (NN, ICA, CNN); some exploratory (RNN, Seq2Seq) and not yet tested on full datasets, scaffolded for extension.  

---

## 📬 Workflow Automation for _Real-World Problems_
- **Automated Newspaper Submission Script**:  
  - Pulls recent Gmail drafts and parses newspaper name, title, and content (via Google API interactions).
  - Generates `.doc` + `.docx` files automatically.
  - Formats HTML email bodies with word counts and metadata.
  - Sends submissions (with images/attachments) to the correct editorial desk.
  - Deletes the draft once successfully sent.
  - [![Watch the demo video](https://img.youtube.com/vi/5s4rt3fWWJM/3.jpg)](https://www.youtube.com/watch?v=5s4rt3fWWJM)  
[Watch demo video](https://www.youtube.com/watch?v=5s4rt3fWWJM) [![YouTube](https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=5s4rt3fWWJM)
- **Automated Google Sheet for Clinical Management**:
  - `Google Apps Script` implementation.
  - Auto calculates pill counts per week and flags overdose.
  - Integrates **command line** function _within sheet_.
  - Displays and removes images on command.
> **✅ STATUS**: Fully functional. **Solves real-world workflow needs** (family clinical/log management, auto-assisted newspaper submissions).

---

## 🔐 Algorithm & Math Explorations
- **Path with Minimum Cost & Interval Scheduling**: explored different algorithms
- **SHA-256 implemented from scratch**: bitwise ops, padding, chaining.  
- **Elliptic curve arithmetic**: exploratory implementation of point addition and doubling.  
> **✅ STATUS**: Open-ended explorations; details documented in notebook markdown. 


---

## 🚧 Note on Project Status
Not every project here is fully polished. Some (RAG, automations) are fully functional demos, while others (Seq2Seq, RNN) are partial implementations included to demonstrate algorithmic curiosity and technical scaffolding. Together, they highlight **breadth, depth, and process**—not just final results.
