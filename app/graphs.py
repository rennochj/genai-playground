"""
Define graph configurations for language model workflow execution.

This module provides configuration classes for creating LangGraph execution graphs
that control the flow of requests and responses through the language model system.
"""

import json
from abc import abstractmethod
from typing import List, TypedDict

import structlog
from langchain_core.messages import BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import Field

# ------------------------------------------------------------------------------------------------------------------------

# Create the logger with proper namespace for this module
logger = structlog.get_logger("app.graphs")

# ------------------------------------------------------------------------------------------------------------------------


class XMessageState(TypedDict):
    """Define the state structure for message-based graphs."""

    messages: List[BaseMessage]
    """A dictionary containing a list of messages exchanged in the graph."""


# ------------------------------------------------------------------------------------------------------------------------


class XBaseGraph:
    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> XMessageState:
        """Invoke the graph with a list of messages and return the final state."""
        pass


# ------------------------------------------------------------------------------------------------------------------------


class XSimpleGraphConfig(XBaseGraph):
    """Configure a simple sequential graph for language model execution."""

    model: ChatOpenAI = Field(..., description="The language model runnable to use in the graph")
    graph: CompiledGraph = Field(..., description="The LangGraph instance to use for the graph execution")

    def __init__(self, model: ChatOpenAI) -> None:
        def llm_node(state: XMessageState) -> XMessageState:
            logger.debug("starting llm_node", state=state)

            messages = state["messages"]
            response = model.invoke(messages)

            logger.debug("ending llm_node", response=response)

            return {"messages": messages + [response]}

        builder = StateGraph(XMessageState)
        builder.add_node("model", llm_node)

        builder.add_edge(START, "model")
        builder.add_edge("model", END)

        self.model = model
        self.builder = builder
        self.graph = builder.compile()

    async def ainvoke(self, messages: List[BaseMessage]) -> XMessageState:
        result = self.graph.invoke({"messages": messages})

        return XMessageState(messages=result["messages"])


# ------------------------------------------------------------------------------------------------------------------------


class XMCPGraphConfig(XBaseGraph):
    """Configure a simple sequential graph for language model execution."""

    _model: ChatOpenAI = Field(..., description="The language model runnable to use in the graph")

    def __init__(self, model: ChatOpenAI) -> None:
        self._model = model

    async def ainvoke(self, messages: List[BaseMessage]) -> XMessageState:
        logger.debug("starting invoke", messages=messages)

        with open("./app/tools.json", "r") as f:
            config = json.load(f)

        client = MultiServerMCPClient(config)

        # async with MultiServerMCPClient(config) as client:
        tools = await client.get_tools()
        logger.debug("starting invoke", messages=messages, tools=tools)
        agent = create_react_agent(model=self._model, tools=tools)
        response = await agent.ainvoke({"messages": messages})
        logger.debug("ending invoke", response=response)

        return XMessageState(messages=response["messages"])
