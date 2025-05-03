"""
Llama4 Integration with LangChain

This module provides integration with Llama4 LLM via Ollama, using LangChain as
the core framework for model interaction.
"""

# Standard library imports
import sys

import structlog

# Third-party imports
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama  # Changed from OllamaLLM to ChatOllama
from pydantic import BaseModel, Field

# Setup structured logging
logger = structlog.get_logger()

# ------------------------------------------------------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for model."""

    model_name: str = Field(..., description="Name of the Llama4 model")
    model_url: str = Field(..., description="URL of the Ollama service")
    temperature: float = Field(0.7, description="Temperature for response generation")
    system_prompt: str = Field(
        "You are a helpful assistant.", description="System prompt for the model"
    )


# ------------------------------------------------------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for the application.

    This function:
    1. Creates a model configuration for the Llama4 LLM
    2. Initializes the LangChain integration with the model
    3. Sends a test query to the model
    4. Handles the response or any errors that occur

    The function demonstrates a simple workflow for integrating with Llama4
    via LangChain and Ollama, showing how to configure, initialize, and
    query the model in a structured way.

    Returns:
        None
    """

    # Create configuration with required parameters for the Llama4 model
    # This defines how the model will behave, including the temperature setting
    # that controls creativity and the system prompt that sets the model's role
    config = ModelConfig(
        model_name="llama4",
        model_url="http://mercury.local:11434",
        temperature=0.7,
        system_prompt="You are a helpful assistant powered by Llama4. Provide informative and thoughtful responses.",
    )

    # Initialize Llama4 integration by creating a chain of prompt template and LLM
    # This may fail if the Ollama server is not reachable or if the model is unavailable
    try:

        llm = ChatOllama(
            model=config.model_name,
            base_url=config.model_url,
            temperature=config.temperature,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(config.system_prompt),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        chain = prompt | llm

        # Test the model with a sample AI-related query
        # The query function handles sending the request and processing the response
        query_text = "write a haiku about NASCAR racing."
        response = chain.invoke({"input": query_text})

        print(response.model_dump_json(indent=2))

    except ValueError as e:
        # Log the error with detailed information and exit the application
        logger.error("value_error", error=str(e))
        sys.exit(1)
    except TypeError as e:
        # Log the error with detailed information and exit the application
        logger.error("type_error", error=str(e))
        sys.exit(1)
    except RuntimeError as e:
        # Log the error with detailed information and exit the application
        logger.error("runtime_error", error=str(e))
        sys.exit(1)
    except ConnectionError as e:
        # Log the error with detailed information and exit the application
        logger.error("connection_error", error=str(e))
        sys.exit(1)
    except TimeoutError as e:
        # Log the error with detailed information and exit the application
        logger.error("timeout_error", error=str(e))
        sys.exit(1)


# ------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
