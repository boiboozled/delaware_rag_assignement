# Internal Engineering Copilot (Agentic RAG)

## The Business Problem
Software engineers and IT staff spend countless hours searching through disparate, fragmented internal documentation (Jira tickets, Confluence pages) to find historical context on system architecture and resolved bugs. 

This project implements an **Agentic RAG (Retrieval-Augmented Generation) Workflow** to act as an Internal Engineering Copilot. It ingests simulated corporate documentation and uses a local, mobile-class LLM to accurately answer engineering queries while strictly maintaining data privacy.

## Key Technologies (Local Execution)
* **LLM:** Qwen2.5-1.5B-Instruct (Chosen for lightning-fast local CPU inference).
* **Framework:** LangGraph / LangChain (Agentic Workflow).
* **Vector Store:** FAISS (Local in-memory execution).
* **Dataset:** Hugging Face `aeriesec/orgforge` (Simulated internal corporate Wiki & Ticketing system).

## Quick Start Guide

### 1. Prerequisites
You must have Python 3.10+ installed. This application is designed to run entirely locally on a standard laptop without relying on external API keys (OpenAI, Anthropic, etc.).

### 2. Setup Instructions
Clone the repository and install the required dependencies:
```bash
git clone <your-repo-link>
cd <your-repo-folder>
pip install -r requirements.txt