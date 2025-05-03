# Standard library imports
import json
import os
import sys
from typing import Any, Dict, Optional, TypeAlias

# Third-party imports
import requests
import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Type aliases for better readability
Message: TypeAlias = Dict[str, str]
Messages: TypeAlias = list[Message]


class ModelSettings(BaseSettings):
    """Configuration settings for the model."""

    model_runner_url: str = Field(
        default=os.getenv("MODEL_RUNNER_URL", "http://localhost:11434"),
        description="Base URL for the model runner service",
    )
    default_model: str = Field(
        default=os.getenv("DEFAULT_MODEL", ""), description="Default model to use for completions"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    initial_wait: float = Field(default=1.0, description="Initial wait time for retry backoff")
    max_wait: float = Field(default=10.0, description="Maximum wait time for retry backoff")

    model_config = SettingsConfigDict(env_file=".env")


class ChatMessage(BaseModel):
    """Model for chat messages."""

    role: str
    content: str
    tool_calls: Optional[list[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Model for chat completion responses."""

    model: str
    created_at: str
    message: Dict[str, Any]
    done: bool
    total_duration: int
    load_duration: int
    prompt_eval_duration: int
    eval_duration: int


class ChatCompletionError(Exception):
    """Custom exception for chat completion errors."""

    pass


# Load settings
settings = ModelSettings()


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=settings.initial_wait, max=settings.max_wait),
    retry=retry_if_exception_type(requests.RequestException),
    before=before_log(logger, 10),  # DEBUG level
    after=after_log(logger, 10),  # DEBUG level
)
def make_chat_completion_request(
    messages: list[ChatMessage], tools: Optional[list[Dict[str, Any]]] = None, **kwargs: Any
) -> ChatResponse:
    """Make a chat completion request to the Ollama endpoint."""
    url = settings.model_runner_url

    payload = {
        "model": settings.default_model,  # Specify Ollama model
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        "stream": False,
    }

    try:
        logger.debug("making_chat_request", url=url, payload=json.dumps(payload, indent=2))
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, json=payload, headers=headers)

        # Add response debug logging
        logger.debug(
            "received_response", status_code=response.status_code, content=response.text[:500]
        )  # Log first 500 chars

        if not response.ok:
            logger.error(
                "request_failed", status_code=response.status_code, error=response.text, url=url
            )
            response.raise_for_status()

        return ChatResponse.model_validate(response.json())

    except requests.RequestException as e:
        logger.error(
            "chat_request_failed",
            error=str(e),
            url=url,
            response_content=getattr(e.response, "text", None),
        )
        raise ChatCompletionError(f"Failed to make chat completion request: {str(e)}") from e


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def execute_tool_call(tool_call: Dict[str, Any]) -> Any:
    """Execute a tool call and return the result."""
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])

    if function_name == "add_numbers":
        return add_numbers(**arguments)
    else:
        raise ValueError(f"Unknown tool: {function_name}")


def main() -> None:
    """Main entry point for the application."""
    tools = [
        {
            "name": "add_numbers",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        }
    ]

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(
            role="user",
            content="What is 25 + 17? Please use the add_numbers function to calculate this.",
        ),
    ]

    try:
        response = make_chat_completion_request(messages, tools=tools, tool_choice="auto")
        # logger.info(
        #     "chat_response_received",
        #     choices=len(response.choices),
        #     usage=response.usage,
        #     model=response.model
        # )
        print(response.model_dump_json(indent=2))

    except ChatCompletionError as e:
        logger.error("chat_completion_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
