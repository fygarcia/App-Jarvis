'''
orchestrator.py
Description: The orchestrator.py file is the core of the JARVIS AI assistant, implementing the central logic for managing agentic workflows. It uses LangGraph to define a stateful graph that processes user input, parses intents using Llama3.2 (via Ollama), routes tasks to specialized agents (e.g., summarization), and aggregates responses. It coordinates interactions between the LLM, agents, and future microservices, maintaining conversational state (short-term memory).

Scope:

Input Processing: Parse user text input into structured intents and entities using Llama3.2.
Task Routing: Dynamically route tasks to appropriate agents based on intent using LangGraph workflows.
State Management: Manage short-term memory (conversational context) via LangGraph’s AssistantState.
Response Aggregation: Collect and format agent outputs into a cohesive user response.
Error Handling: Handle errors from LLM, agents, or workflow execution, returning user-friendly messages.
Extensibility: Support modular addition of new agents without altering core logic.
Excludes long-term memory (handled by memory_manager.py), microservice interactions (via MCPs), and UI, focusing solely on backend orchestration for a single user.

'''

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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Create module logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"

# Pydantic Models
class Intent(BaseModel):
    """Model representing a parsed user intent."""
    model_config = ConfigDict(frozen=True)  # Make instances immutable
    
    task: str = Field(
        ...,  # Required field
        description="The action to perform",
        pattern="^[a-z_]+$"  # Only lowercase letters and underscores
    )
    target: Optional[str] = Field(
        None,
        description="What to perform the action on"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the task"
    )

    def __str__(self) -> str:
        return f"Intent(task={self.task}, target={self.target})"

class Message(BaseModel):
    """Model representing a chat message."""
    model_config = ConfigDict(frozen=True)
    
    role: str = Field(
        ...,
        description="Role of the message sender",
        pattern="^(user|assistant|system)$"
    )
    content: str = Field(
        ...,
        description="Content of the message",
        min_length=1
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created"
    )

class AssistantState(BaseModel):
    """Model representing the assistant's state."""
    model_config = ConfigDict(
        frozen=True,  # Make instances immutable
        validate_assignment=True,  # Validate on attribute assignment
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "user", "content": "Summarize this text"}
                ],
                "intents": [
                    {"task": "summarize", "target": "text", "parameters": {}}
                ],
                "response": "Here's your summary..."
            }
        }
    )
    
    messages: List[Message] = Field(
        default_factory=list,
        description="History of messages in the conversation"
    )
    intents: List[Intent] = Field(
        default_factory=list,
        description="List of parsed intents from the last message"
    )
    response: Optional[str] = Field(
        None,
        description="Generated response to the user"
    )

# LLM Interface
class LLMInterface:
    def __init__(self, host: str, model: str):
        """Initialize LLM interface."""
        logger.info(f"Initializing LLM interface with host={host}, model={model}")
        self.client = Client(host=host)
        self.model = model
        
    def parse_input(self, text: str) -> List[Intent]:
        """
        Parse user input into structured intents using Llama3.2.
        
        Args:
            text (str): The user input text to parse
            
        Returns:
            List[Intent]: List of parsed intents. Empty list if parsing fails.
            
        Raises:
            None: All exceptions are caught and logged
        """
        logger.debug(f"Parsing input text: {text}")
        try:
            prompt = (
                "You are a task parser. Parse the user input into a list of intents.\n"
                "ALWAYS return a valid JSON array containing objects with these fields:\n"
                "- task: The action to perform\n"
                "- target: The actual text or content to perform the action on\n"
                "- parameters: Additional parameters as an object (optional)\n\n"
                "Example valid responses:\n"
                '[{"task": "summarize", "target": "The quick brown fox jumps over the lazy dog", "parameters": {}}]\n'
                '[{"task": "search", "target": "python files", "parameters": {"type": "code"}}]\n\n'
                "For summarize tasks, put the text to summarize in the target field.\n"
                "ONLY return the JSON array, no other text or code.\n"
                f"User input: {text}"
            )
            # Make LLM call
            logger.debug("Sending prompt to LLM")
            try:
                response: ChatResponse = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                logger.debug("Received response from LLM")
            except ConnectionError as e:
                logger.error(f"Failed to connect to Ollama server: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error during LLM call: {e}")
                return []
            
            # Extract content from Ollama ChatResponse
            if not isinstance(response, ChatResponse):
                logger.error(f"Invalid response type from Ollama: {type(response)}")
                return []
                
            content: str = response.message.content
            if not content:
                logger.error("Empty response content from Ollama")
                return []
            
            logger.debug(f"Raw LLM response: {content}")
            
            # Parse JSON response
            try:
                intents_data = json.loads(content)
                logger.debug(f"Parsed intents data: {intents_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}\nContent: {content}")
                return []
            
            # Validate response structure
            if not isinstance(intents_data, list):
                logger.error(f"Expected list of intents, got {type(intents_data).__name__}")
                return []
                
            # Create Intent objects
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

# Summarization Agent (Placeholder)
class SummarizationAgent:
    def __init__(self, llm: LLMInterface):
        """Initialize summarization agent."""
        logger.info("Initializing SummarizationAgent")
        self.llm = llm

    def summarize(self, text: str) -> str:
        """Summarize the given text."""
        logger.debug(f"Attempting to summarize text: {text[:100]}...")
        try:
            prompt = f"Summarize the following text in 2-3 sentences:\n{text}"
            response = self.llm.client.chat(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.message.content
            logger.debug(f"Generated summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return f"Error generating summary: {str(e)}"

# Orchestrator
class JarvisOrchestrator:
    def __init__(self):
        """Initialize the orchestrator."""
        logger.info("Initializing JarvisOrchestrator")
        self.llm = LLMInterface(
            os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        )
        self.summarization_agent = SummarizationAgent(self.llm)
        self.graph = self._build_graph()
        logger.info("JarvisOrchestrator initialization complete")

    def _parse_input(self, state: AssistantState) -> AssistantState:
        """Parse user input into intents."""
        logger.debug("Entering _parse_input")
        text = state.messages[-1].content
        logger.debug(f"Parsing message content: {text}")
        intents = self.llm.parse_input(text)
        new_state = AssistantState(messages=state.messages, intents=intents)
        logger.debug(f"Created new state with {len(intents)} intents")
        return new_state

    def _route_task(self, state: AssistantState) -> AssistantState:
        """Route tasks to appropriate agents."""
        logger.debug("Entering _route_task")
        response = ""
        for idx, intent in enumerate(state.intents):
            logger.debug(f"Processing intent {idx}: {intent}")
            if intent.task == "summarize":
                if not intent.target:
                    logger.warning("Summarize task received with no target text")
                    response += "Error: No text provided to summarize.\n"
                    continue
                
                logger.debug(f"Summarizing text: {intent.target[:100]}...")
                summary = self.summarization_agent.summarize(intent.target)
                response += f"Summary: {summary}\n"
            else:
                logger.warning(f"Unsupported task type: {intent.task}")
                response += f"Task {intent.task} not supported yet.\n"
        
        new_state = AssistantState(messages=state.messages, intents=state.intents, response=response)
        logger.debug(f"Created new state with response: {response[:100]}...")
        return new_state

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
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
        """Process user input and return response."""
        logger.info(f"Processing user input: {user_input}")
        try:
            # Create initial state with validated Message object
            initial_message = Message(role="user", content=user_input)
            initial_state = AssistantState(messages=[initial_message])
            logger.debug("Created initial state")
            
            # Invoke graph
            logger.debug("Invoking workflow graph")
            result = self.graph.invoke(initial_state)
            
            # Convert the LangGraph state result to a dictionary and validate
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
    user_input = "Summarize this text: The quick brown fox jumps over the lazy dog."
    print("User Input:", user_input)
    response = orchestrator.process_input(user_input)
    print("Response:", response)