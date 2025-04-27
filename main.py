# Standard library imports
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, TypeAlias

# Third-party imports
import requests
import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()

# Configure standard logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Set log level from environment or default to INFO
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.getLogger().setLevel(log_level)

logger = structlog.get_logger()

# Type aliases for better readability
Message: TypeAlias = Dict[str, str]
Messages: TypeAlias = list[Message]

class ModelSettings(BaseSettings):
    """Configuration settings for the model."""
    model_runner_url: str = Field(
        default=os.getenv("MODEL_RUNNER_URL", ""),
        description="Base URL for the model runner service"
    )
    default_model: str = Field(
        default=os.getenv("DEFAULT_MODEL", ""),
        description="Default model to use for completions"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    initial_wait: float = Field(default=1.0, description="Initial wait time for retry backoff")
    max_wait: float = Field(default=10.0, description="Maximum wait time for retry backoff")

    model_config = SettingsConfigDict(
        env_file="./.env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: str
    content: str
    tool_calls: Optional[list[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Model for chat completion responses."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Dict[str, Any]]
    usage: Dict[str, int]

class ChatCompletionError(Exception):
    """Custom exception for chat completion errors."""
    pass

# Load settings
settings = ModelSettings()

def make_chat_completion_request(
    messages: list[Dict[str, str]],
    **kwargs: Any
) -> Any:
    """Make a chat completion request to the Ollama endpoint."""
    url = settings.model_runner_url
    
    payload = {
        "model": settings.default_model,
        "messages": messages,
        "stream": False,  # Changed from "False" string to False boolean
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 500
        }
     }

    try:
        logger.debug("making_chat_request", 
                    url=url, 
                    payload=json.dumps(payload, indent=2))
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
            
        return ChatResponse(**response.json())
    
    except requests.RequestException as e:
        logger.error(
            "chat_request_failed",
            error=str(e),
            url=url,
            response_content=getattr(e.response, 'text', None)
        )
        raise ChatCompletionError(f"Failed to make chat completion request: {str(e)}") from e

def main() -> None:
    """Main entry point for the application."""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is 25 + 17?"
        }
    ]

    try:
        response = make_chat_completion_request(
            messages
        )

        print(response.model_dump_json(indent=2))
        
    except ChatCompletionError as e:
        logger.error("chat_completion_failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":

    main()