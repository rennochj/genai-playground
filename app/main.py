import asyncio
from dotenv import load_dotenv
from graphs import XBaseGraph, XMCPGraphConfig
from langchain_core.messages import HumanMessage, SystemMessage
from model import XOpenAIConfig
import logging
import structlog
import sys

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


async def main(graph: XBaseGraph) -> None:
    logger.info("main_execution_started")
    initial_messages = [
        SystemMessage(content="You are a helpful assistant that can use tools to provide accurate information."),
        # HumanMessage(content="What's the weather in New York? Also, what's 25 * 16?"),
        HumanMessage(content="What's the weather in New York? Also, is it raining? Also, what's 25 * 16?"),
        # HumanMessage(content="what's (3 + 5) x 12?"),
    ]

    logger.debug("sending_message", content=initial_messages[-1].content)
    result = await graph.ainvoke(initial_messages)
    logger.info("response_received", result=result["messages"][-1].content)

    # print(dumps(result, indent=2))


# ------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()
    logger.debug("environment_loaded")

    model = XOpenAIConfig().build()
    # graph = XSimpleGraphConfig(model=model)
    graph = XMCPGraphConfig(model=model)

    asyncio.run(main(graph=graph))

    logger.info("done")
