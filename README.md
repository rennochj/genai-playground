# GenAI Playground

A playground for integrating with Large Language Models (LLMs) using LangChain and Model Context Protocol (MCP).

## Features

- Integration with Llama4 via Ollama
- LangChain as the core framework for LLM interactions
- Tool support using Model Context Protocol (MCP)
- Weather tool demonstration

## Setup

1. Install dependencies:
   ```
   make install-deps
   ```

2. Ensure Ollama is running with Llama4 model installed:
   ```
   ollama pull llama4
   ollama serve
   ```

3. Run the application:
   ```
   make run
   ```

## Architecture

- `main.py`: Entry point containing the Llama4 integration with MCP
- `config.py`: Configuration settings for the application
- `chat.py`: Basic chat functionality (legacy code)
- `mcp_client.py`: MCP client implementation

## Tools

The application demonstrates tool usage with a weather tool that can:
- Retrieve weather information for a specific location
- Support different temperature units (celsius/fahrenheit)

## Using MCP

The Model Context Protocol (MCP) provides standardized communication between the application and language models. Our implementation:

1. Creates an MCP server that exposes LangChain tools
2. Registers these tools for discovery by the model
3. Handles tool execution and returns results to the model