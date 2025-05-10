"""
Llama4 Integration with LangChain

This module provides integration with Llama4 LLM via Ollama, using LangChain as
the core framework for model interaction.
"""

import asyncio
import os

# Standard library imports
import sys
from typing import Any, Dict, Optional, TypedDict

import structlog

# Third-party imports
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
from langchain_ollama import ChatOllama  # Changed from OllamaLLM to ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Setup structured logging
# ------------------------------------------------------------------------------------------------------------------------
logger = structlog.get_logger()


class MessagesState(TypedDict):
    """Type definition for the state containing messages in a conversation."""

    messages: Any


def tools_condition(state: MessagesState) -> str | Any:
    """Determine the next node based on whether a tool call is needed."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    transport: str = Field(..., description="Transport protocol (stdio or sse)")
    command: Optional[str] = Field(None, description="Command to start the server (for stdio)")
    args: Optional[list[str]] = Field(None, description="Arguments for the command (for stdio)")
    url: Optional[str] = Field(None, description="URL of the server (for sse)")


class ModelConfig(BaseModel):
    """Configuration for model."""

    model_name: str = Field(..., description="Name of the Llama4 model")
    model_url: str = Field(..., description="URL of the Ollama service")
    temperature: float = Field(0.7, description="Temperature for response generation")
    system_prompt: str = Field("You are a helpful assistant.", description="System prompt for the model")
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict, description="MCP server configurations")
    use_mcp: bool = Field(False, description="Whether to use MCP servers")


# ------------------------------------------------------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for the application that integrates with the Llama4 LLM.

    This function orchestrates the complete workflow for interacting with the Llama4
    language model through LangChain's abstractions and Ollama's hosting platform.

    The implementation follows these steps:
    1. Creates a model configuration with parameters tailored for Llama4
       - Sets appropriate temperature for balanced responses
       - Configures system prompt to define the model's behavior and persona

    2. Initializes the LangChain components:
       - Constructs a ChatOllama instance configured for the target model
       - Creates a structured chat prompt template with system and user message slots
       - Chains these components using LangChain's pipe operator

    3. Queries the model with a demonstration prompt
       - Sends a creative writing request to showcase model capabilities
       - Formats and displays the JSON response

    4. Implements comprehensive error handling:
       - Catches a variety of potential exceptions (type errors, connection issues, etc.)
       - Logs detailed error information with structured logging
       - Gracefully exits with appropriate status codes

    Returns:
        None: This function doesn't return a value but prints the model's response
              to standard output and exits with an appropriate status code.
    """

    # Create configuration with required parameters for the Llama4 model
    # This defines how the model will behave, including the temperature setting
    # that controls creativity and the system prompt that sets the model's role
    config = ModelConfig(
        model_name="llama3.1:latest",
        # model_name="llama3.3:latest",
        # model_name="llama4:latest",
        # model_name="smollm2:latest",
        model_url="http://mars.local:11434",
        temperature=0.7,
        system_prompt="You are a helpful assistant powered by Llama4. Provide informative and thoughtful responses.",
        use_mcp=True,
        mcp_servers={
            "math": MCPServerConfig(
                transport="stdio",
                command="python",
                args=["/workspaces/genai-playground/math_server.py"],
                url=None,
            ),
            "weather": MCPServerConfig(
                transport="sse",
                url="http://localhost:8000/sse",
                command=None,
                args=None,
            ),
        },
    )

    # Initialize Llama4 integration by creating a chain of prompt template and LLM
    # This may fail if the Ollama server is not reachable or if the model is unavailable
    try:
        # model = ChatOllama(
        #     model=config.model_name,
        #     base_url=config.model_url,
        #     temperature=config.temperature,
        # )

        model = ChatOpenAI(
            model="ai/llama3.3:latest",
            base_url="http://model-runner.docker.internal/engines/v1",
            api_key="ignore",
        )

        # model = ChatOpenAI(
        #     model=config.model_name,
        #     base_url=config.model_url,
        #     temperature=config.temperature,
        # )

        if config.use_mcp:
            # Convert the MCP server configs to the format expected by MultiServerMCPClient
            mcp_config = {
                name: {
                    "transport": server.transport,
                    **({"command": server.command, "args": server.args} if server.transport == "stdio" else {}),
                    **({"url": server.url} if server.transport == "sse" else {}),
                }
                for name, server in config.mcp_servers.items()
            }

            # Create an async context manager for the MCP client
            async def run_with_mcp() -> None:
                async with MultiServerMCPClient(mcp_config) as client:

                    tools = client.get_tools()

                    def call_model(state: MessagesState) -> MessagesState:
                        response = model.bind_tools(tools).invoke(state["messages"])
                        return {"messages": state["messages"] + [response]}  # Append the response to existing messages

                    builder = StateGraph(MessagesState)
                    builder.add_node("call_model", call_model)  # Use named node
                    builder.add_node("tools", ToolNode(tools))
                    builder.add_node("tools_condition", tools_condition)
                    builder.add_node("final", call_model)

                    builder.add_edge(START, "call_model")
                    # builder.add_edge("call_model", "tools")
                    builder.add_conditional_edges(
                        "call_model",
                        tools_condition,
                    )
                    builder.add_edge("tools", "final")
                    builder.add_edge("final", END)

                    # Compile the graph after generating visualization
                    graph = builder.compile()

                    mermaid_diagram = graph.get_graph().draw_mermaid_png()
                    with open("model_graph.png", "wb") as f:
                        f.write(mermaid_diagram)

                    print(f"Graph visualization saved to {os.path.abspath('model_graph.md')}")

                    math_response = await graph.ainvoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content="using available tools, calculate and return the result of the following expression (3 * 10) * 4?"
                                )
                            ]
                        }
                    )

                    print("Math Response:")
                    print(math_response)

                    # weather_response = await graph.ainvoke(
                    #     {"messages": [HumanMessage(content="what is the weather in nyc?")]}
                    # )

                    # print("Weather Response:")
                    # print(weather_response)

                    # # Create a ReAct agent with the MCP tools
                    # agent = create_react_agent(model, client.get_tools())

                    # # Test the agent with a sample query
                    # query_text = "what's (3 + 5) x 12?"
                    # response = await agent.ainvoke({"messages": query_text}, debug=False)

                    # print("MCP Agent Response:")
                    # print(response)

                    # # You can test with another example
                    # if "weather" in config.mcp_servers:
                    #     weather_query = "what is the weather in nyc?"
                    #     weather_response = await agent.ainvoke({"messages": weather_query})
                    #     print("\nWeather Query Response:")
                    #     print(weather_response)

            asyncio.run(run_with_mcp(), debug=True)

        else:
            # Standard LLM chain without MCP tools
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(config.system_prompt),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )

            chain = prompt | model

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
