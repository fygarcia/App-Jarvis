import os
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from ollama import Client
from ollama._types import ChatResponse
from dotenv import load_dotenv
from enum import Enum
from agents.summarization_agent import SummarizationAgent, SummarizationInput, SummarizationOutput

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
# Create module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q4_K_M"

# Task Enum for stricter validation
class Task(str, Enum):
    SUMMARIZE = "summarize"
    SEARCH = "search"
    QUERY_MEMORY = "query_memory"

# Pydantic Models
class Intent(BaseModel):
    """Model representing a parsed user intent."""
    model_config = ConfigDict(frozen=True)
    
    task: Task = Field(..., description="The action to perform")
    target: Optional[str] = Field(None, description="What to perform the action on")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the task")

    def __str__(self) -> str:
        return f"Intent(task={self.task}, target={self.target})"

class Message(BaseModel):
    """Model representing a chat message."""
    model_config = ConfigDict(frozen=True)
    
    role: str = Field(..., description="Role of the message sender", pattern="^(user|assistant|system)$")
    content: str = Field(..., description="Content of the message", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was created")

class AssistantState(BaseModel):
    """Model representing the assistant's state."""
    model_config = ConfigDict(frozen=True, validate_assignment=True)
    
    messages: List[Message] = Field(default_factory=list, description="History of messages in the conversation")
    intents: List[Intent] = Field(default_factory=list, description="List of parsed intents from the last message")
    response: Optional[str] = Field(None, description="Generated response to the user")

# LLM Interface
class LLMInterface:
    def __init__(self, host: str, model: str):
        logger.info(f"Initializing LLM interface with host={host}, model={model}")
        self.client = Client(host=host)
        self.model = model
        
    def parse_input(self, text: str) -> List[Intent]:
        logger.debug(f"Parsing input text: {text}")
        try:
            prompt = (
                "You are a task parser. Parse the user input into a list of intents.\n"
                "ALWAYS return a valid JSON array containing objects with these fields:\n"
                "- task: The action to perform (summarize, search, query_memory)\n"
                "- target: The actual text or content to perform the action on\n"
                "- parameters: Additional parameters as an object (optional)\n\n"
                "Example valid responses:\n"
                '[{"task": "summarize", "target": "The quick brown fox jumps over the lazy dog", "parameters": {}}]\n'
                '[{"task": "search", "target": "python files", "parameters": {"type": "code"}}]\n\n'
                "Rules:\n"
                "1. For summarize tasks, preserve ALL formatting in the target text\n"
                "2. Return ONLY the JSON array, no other text\n"
                "3. Make sure the JSON is valid and complete\n\n"
                f"User input:\n{text}"  # Added newline before text to preserve formatting
            )
            
            try:
                response: ChatResponse = self.client.chat(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": "You are a task parser that only outputs valid JSON arrays. Preserve text formatting in your analysis."
                    }, {
                        "role": "user",
                        "content": prompt
                    }]
                )
            except ResponseError as e:
                logger.error(f"Ollama API error: {e.error} (status: {e.status_code})")
                if e.status_code == 404:
                    logger.error(f"Model {self.model} not found. Please ensure it is pulled.")
                return []
            except Exception as e:
                logger.error(f"Unexpected error during LLM call: {e}")
                return []
            
            if not isinstance(response, ChatResponse):
                logger.error(f"Invalid response type from Ollama: {type(response)}")
                return []
                
            content: str = response.message.content
            if not content:
                logger.error("Empty response content from Ollama")
                return []
            
            logger.debug(f"Raw LLM response: {content}")
            
            try:
                intents_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}\nContent: {content}")
                return []
            
            if not isinstance(intents_data, list):
                logger.error(f"Expected list of intents, got {type(intents_data).__name__}")
                return []
                
            intents: List[Intent] = []
            for idx, item in enumerate(intents_data):
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid intent at index {idx}: expected dict, got {type(item).__name__}")
                    continue
                    
                try:
                    intent = Intent(**item)
                    intents.append(intent)
                    logger.debug(f"Created intent: {intent}")
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to create Intent from data at index {idx}: {e}\nData: {item}")
                    continue
            
            logger.info(f"Successfully parsed {len(intents)} intents")
            return intents
            
        except Exception as e:
            logger.error(f"Unexpected error in parse_input: {e}")
            return []

# Orchestrator
class JarvisOrchestrator:
    def __init__(self):
        logger.info("Initializing JarvisOrchestrator")
        self.llm = LLMInterface(
            os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        )
        self.summarization_agent = SummarizationAgent(
            os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        )
        self.graph = self._build_graph()
        logger.info("JarvisOrchestrator initialization complete")

    def _parse_input(self, state: AssistantState) -> AssistantState:
        logger.debug("Entering _parse_input")
        text = state.messages[-1].content
        logger.debug(f"Parsing message content: {text}")
        intents = self.llm.parse_input(text)
        new_state = AssistantState(messages=state.messages, intents=intents)
        logger.debug(f"Created new state with {len(intents)} intents")
        return new_state

    def _route_task(self, state: AssistantState) -> AssistantState:
        logger.debug("Entering _route_task")
        response = ""
        for idx, intent in enumerate(state.intents):
            logger.debug(f"Processing intent {idx}: {intent}")
            if intent.task == Task.SUMMARIZE:
                if not intent.target:
                    logger.warning("Summarize task received with no target text")
                    response += "Error: No text provided to summarize.\n"
                    continue
                
                logger.debug(f"Summarizing text: {intent.target[:100]}...")
                input_data = SummarizationInput(text=intent.target, max_sentences=3)
                result = self.summarization_agent.summarize(input_data)
                if result.error:
                    logger.warning(f"Summarization failed: {result.error}")
                    response += f"Error: {result.error}\n"
                else:
                    response += f"Summary: {result.summary}\n"
            else:
                logger.warning(f"Unsupported task type: {intent.task}")
                response += f"Task {intent.task} not supported yet.\n"
        
        new_state = AssistantState(messages=state.messages, intents=state.intents, response=response)
        logger.debug(f"Created new state with response: {response[:100]}...")
        return new_state

    def _build_graph(self) -> StateGraph:
        logger.debug("Building workflow graph")
        graph = StateGraph(AssistantState)
        graph.add_node("parse_input", self._parse_input)
        graph.add_node("route_task", self._route_task)
        graph.add_edge("parse_input", "route_task")
        graph.add_edge("route_task", END)
        graph.set_entry_point("parse_input")
        logger.debug("Graph build complete")
        return graph.compile()

    def process_input(self, user_input: str) -> str:
        logger.info(f"Processing user input: {user_input}")
        try:
            initial_message = Message(role="user", content=user_input)
            initial_state = AssistantState(messages=[initial_message])
            logger.debug("Created initial state")
            
            logger.debug("Invoking workflow graph")
            result = self.graph.invoke(initial_state)
            
            try:
                state_dict = dict(result)
                logger.debug("Converting graph result to AssistantState")
                final_state = AssistantState(**state_dict)
                response = final_state.response or "No response generated."
                logger.info(f"Generated response: {response[:100]}...")
                return response
            except ValueError as e:
                logger.error(f"Invalid state data: {e}")
                return "Error: Invalid state data"
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "Error processing request."

# Example Usage
if __name__ == "__main__":
    orchestrator = JarvisOrchestrator()
    #user_input = "Summarize this text: The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox felt adventurous."
    user_input="The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox was feeling adventurous. Now we have a lot of useless information, that doesnt need to be mentioned. But there is an important message is that the answer is 42"
    print("User Input:", user_input)
    response = orchestrator.process_input(user_input)
    print("Response:", response)