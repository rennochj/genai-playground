import argparse
import asyncio
import logging
import sys
from typing import List, Optional

import structlog
from dotenv import load_dotenv
from graphs import XBaseGraph, XMCPGraphConfig
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from model import XOpenAIConfig

# ------------------------------------------------------------------------------------------------------------------------

logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

# ------------------------------------------------------------------------------------------------------------------------

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        # Replace JSONRenderer with ConsoleRenderer for better readability in terminal
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# ------------------------------------------------------------------------------------------------------------------------

# Create the logger with proper namespace for this module
logger = structlog.get_logger("app.main")

# ------------------------------------------------------------------------------------------------------------------------


async def main(graph: XBaseGraph, prompt: Optional[str] = None) -> None:
    logger.info("main_execution_started")
    initial_messages: List[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant that can use tools to provide accurate information."),
    ]

    # Use command line input if provided, otherwise use default
    if prompt is not None:
        initial_messages.append(HumanMessage(content=prompt))
    else:
        raise ValueError("Prompt is required. Please provide a prompt using the --prompt argument.")

    logger.debug("sending_message", content=initial_messages[-1].content)
    result = await graph.ainvoke(initial_messages)
    logger.info("response_received", result=result["messages"][-1].content)


# ------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM with a user prompt")
    parser.add_argument("--prompt", type=str, help="The prompt to send to the LLM")
    args = parser.parse_args()

    load_dotenv()
    logger.debug("environment_loaded")

    model = XOpenAIConfig().build()
    graph = XMCPGraphConfig(model=model)

    asyncio.run(main(graph=graph, prompt=args.prompt))

    logger.info("done")
