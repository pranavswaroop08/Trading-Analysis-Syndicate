# Trading-Analysis-Syndicate

An enterprise-grade, multi-agent artificial intelligence system designed to autonomously analyze financial markets and execute simulated trading decisions. 

Built with **LangGraph**, this project utilizes a Directed Acyclic Graph (DAG) state machine to orchestrate specialized AI agents powered by local Large Language Models (LLMs) via **Ollama**.

---

## 🧠 Architecture Overview

Unlike standard sequential agent frameworks, this system treats the trading floor as a precise **State Machine**. Market data is explicitly typed and passed through a series of specialized nodes (agents), ensuring deterministic, highly reliable execution without infinite loops or hallucinations.

The workflow consists of four primary nodes:

1. **Data Gatherer:** Ingests live market data (price action, volume) and recent financial news headlines using the `yfinance` library.
2. **Quantitative Analyst:** A local Llama 3.2 model prompted to act as a technical analyst, evaluating trends, consolidations, and volume patterns.
3. **Sentiment Analyst:** A local Llama 3.2 model that reads news headlines to gauge broader market sentiment (Bullish, Bearish, or Neutral).
4. **Chief Risk Officer (CRO):** The final execution node. It synthesizes the technical and sentiment reports to make a strict, justified `BUY`, `SELL`, or `HOLD` decision.

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph, LangChain Core
* **Local LLM Engine:** Ollama (Llama 3.2 - 3B Parameters)
* **Financial Data Integration:** `yfinance`
* **Terminal UI:** `Rich` (for colorized, panel-based reporting)
* **Language:** Python 3.10+

---

## ⚙️ Prerequisites

Because this project prioritizes data privacy and zero API costs, it runs the LLMs entirely locally. You must have **Ollama** installed on your machine.

1. Install [Ollama](https://ollama.com/).
2. Pull the lightweight Llama 3.2 model via terminal/command prompt:
   ```bash
   ollama run llama3.2


  Steps to run:

  1. git clone [https://github.com/pranavswaroop08/Trading-Analysis-Syndicate](https://github.com/pranavswaroop08/Trading-Analysis-Syndicate)
     cd "Trading-Analysis-Syndicate"
  2. python -m venv venv
     .\venv\Scripts\activate
  3. pip install langgraph langchain-community langchain-core yfinance rich bs4
  4. python main.py
