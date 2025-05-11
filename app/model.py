"""Define configuration classes for LLM models.

This module provides abstract and concrete configuration classes for language models.
Use these classes to create, configure, and instantiate language model instances
with appropriate settings loaded from environment variables.
"""

import os
from abc import abstractmethod

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# ------------------------------------------------------------------------------------------------------------------------

load_dotenv()

# ------------------------------------------------------------------------------------------------------------------------


class XChatConfig(BaseModel):
    @abstractmethod
    def build(self) -> ChatOpenAI:
        """Factory method to create a ChatOpenAI instance."""


# ------------------------------------------------------------------------------------------------------------------------


class XOpenAIConfig(XChatConfig):
    """Configuration for ChatOpenAI parameters."""

    base_url: str = Field(default=os.getenv("MODEL_BASE_URL", ""), description="Specify the base URL for the model API")
    model_name: str = Field(default=os.getenv("MODEL_NAME", "ai/llama3.3"), description="Specify the model identifier")
    api_key: SecretStr = Field(
        default=SecretStr(os.getenv("MODEL_API_KEY", "no-key-required")),
        description="Provide the API key for authentication",
    )
    temperature: float = Field(
        default=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
        description="Set the temperature for response generation",
    )

    def build(self) -> ChatOpenAI:
        """Create a ChatOpenAI instance."""
        return ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )
