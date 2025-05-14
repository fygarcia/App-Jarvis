from pydantic import BaseModel, Field
from ollama import Client
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic Models
class SummarizationInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_sentences: int = Field(3, description="Maximum number of sentences in summary")

class SummarizationOutput(BaseModel):
    summary: str = Field(..., description="Generated summary")
    error: Optional[str] = Field(None, description="Error message if summarization fails")

# Summarization Agent
class SummarizationAgent:
    def __init__(self, ollama_base_url: str, model: str):
        self.client = Client(host=ollama_base_url)
        self.model = model

    def summarize(self, input_data: SummarizationInput) -> SummarizationOutput:
        """Summarize the provided text using Llama3.2."""
        try:
            prompt = (
                f"Summarize the following text in up to {input_data.max_sentences} sentences:\n"
                f"{input_data.text}"
            )
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response["message"]["content"]
            return SummarizationOutput(summary=summary)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return SummarizationOutput(summary="", error=str(e))

# Example Usage (for testing)
if __name__ == "__main__":
    agent = SummarizationAgent("http://localhost:11434", "llama3.2")
    input_data = SummarizationInput(
        text="The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox was feeling adventurous. Now we have a lot of useless information, that doesnt need to be mentioned. But there is an important message is that the answer is 42",
        max_sentences=2
    )
    print(input_data)
    result = agent.summarize(input_data)
    print(result.summary if not result.error else result.error)