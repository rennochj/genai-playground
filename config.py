# Standard library imports
import logging
import os
import sys
from typing import Dict, List, TypeAlias

# Third-party imports
import structlog
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
if not load_dotenv():
    print("Warning: .env file not found or empty", file=sys.stderr)

# --------------------------------------------------------------------------------


def setup_logging() -> None:
    """Configure both standard and structured logging."""
    try:
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
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in logging._nameToLevel:
            raise ValueError(f"Invalid log level: {log_level}")
        logging.getLogger().setLevel(log_level)
    except Exception as e:
        print(f"Failed to configure logging: {str(e)}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------------

# Type aliases for better readability
Message: TypeAlias = Dict[str, str]
Messages: TypeAlias = List[Message]

# --------------------------------------------------------------------------------


class ModelSettings(BaseSettings):
    """Configuration settings for the model."""

    model_url: str = Field(
        default=os.getenv("MODEL_RUNNER_URL", ""),
        description="Base URL for the model runner service",
    )
    model_name: str = Field(
        default=os.getenv("DEFAULT_MODEL", ""), description="Default model to use for completions"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    initial_wait: float = Field(default=1.0, description="Initial wait time for retry backoff")
    max_wait: float = Field(default=10.0, description="Maximum wait time for retry backoff")

    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8", extra="ignore")

    @property
    def is_valid(self) -> bool:
        """Check if required settings are properly configured."""
        return bool(self.model_url and self.model_name)


# --------------------------------------------------------------------------------

# Initialize settings and logger with error handling
try:
    settings = ModelSettings()
    if not settings.is_valid:
        raise ValueError(
            "Required settings are missing. Check MODEL_RUNNER_URL and DEFAULT_MODEL environment variables."
        )

    setup_logging()
    logger = structlog.get_logger()
except Exception as e:
    print(f"Configuration error: {str(e)}", file=sys.stderr)
    sys.exit(1)
