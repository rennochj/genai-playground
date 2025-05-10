# Standard library imports
import sys

# Local imports
from archive.chat import ChatCompletionError, ChatMessage, request
from model import logger, settings

"""
Weather Information Assistant

This module implements a chat-based interface to retrieve weather information
using AI model capabilities. It defines the necessary tools for weather data 
retrieval and handles the communication with the AI model.

The script sets up a conversation with specific weather-related queries and
processes the AI model's responses, including potential tool calls for
retrieving actual weather data.
"""

# --------------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for the weather assistant application.

    This function:
    1. Defines the tools available to the AI model (get_weather function)
    2. Sets up the initial messages for the conversation
    3. Sends the request to the AI model
    4. Prints the response or handles errors

    Returns:
        None
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get today's weather and temperature for a location. Use this function to determine the temperature, precipitation like rain, and other weather conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        ChatMessage(
            role="system",
            content="You are a helpful assistant. Use the tools to retreive weather information to resond to any questions related to weather, temperature, and precipitaton.",
            tool_calls=[],
            tool_call_id="",
            name=None,
            function_call=None,
        ),
        ChatMessage(
            role="user",
            content="Is it currently raining in San Antonio?",
            # content="Get the current weather in san antonio and then determine if it is raining?",
            tool_calls=[],
            tool_call_id="",
            name=None,
            function_call=None,
        ),
    ]

    try:
        # Send the request to the AI model with the defined tools and messages
        # Use MCP client if enabled via environment variable
        use_mcp = settings.use_mcp if hasattr(settings, "use_mcp") else False

        response = request(
            model_url=settings.model_url,
            model_name=settings.model_name,
            messages=messages,
            tools=tools,
            use_mcp=use_mcp,  # Enable MCP client when desired
        )
        # Output the model's response as formatted JSON
        print(response.model_dump_json(indent=2))

    except ChatCompletionError as e:
        # Log and exit if there's an error with the chat completion
        logger.error("chat_completion_failed", error=str(e))
        sys.exit(1)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
