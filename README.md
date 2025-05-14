# JARVIS - Local AI Assistant

A scalable, local-first AI assistant inspired by Iron Man's JARVIS.

## Features
- 100% local execution with Ollama (Llama3.2)
- Task orchestration with LangGraph
- Modular agent architecture
- Docker-based deployment

## Requirements
- Python 3.8+
- Docker
- Ollama server running locally

## Installation
1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate virtual environment: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Configuration
Create a `.env` file with:
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

```

## Usage
```bash
python core/orchestrator.py
```

```python
from agents.summarization_agent import SummarizationAgent

agent = SummarizationAgent("http://localhost:11434", "llama3.2")
input_data = SummarizationInput(
    text="The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox was feeling adventurous.",
    max_sentences=2
)
result = agent.summarize(input_data)
print(result.summary if not result.error else result.error)
```
