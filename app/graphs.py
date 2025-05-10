"""
Define graph configurations for language model workflow execution.

This module provides configuration classes for creating LangGraph execution graphs
that control the flow of requests and responses through the language model system.
"""

from typing import Any, Dict, List, TypedDict

from langchain.schema.runnable import Runnable
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph


class MessageState(TypedDict):
    """Define the state structure for message-based graphs."""

    messages: List[BaseMessage]
    """A dictionary containing a list of messages exchanged in the graph."""


class XSimpleGraphConfig:
    """Configure a simple sequential graph for language model execution."""

    def build(self, llm: Runnable[list[Any], Any]) -> StateGraph:
        """
        Create a simple LangGraph with a single LLM node.

        Args:
            llm: The language model runnable to use in the graph

        Returns:
            A configured StateGraph ready for compilation
        """

        # Define the LLM node function
        def llm_node(state: MessageState) -> MessageState:
            # Extract messages from state
            messages = state["messages"]
            # Invoke the LLM with the messages
            response = llm.invoke(messages)
            # Return updated state with response added
            return {"messages": messages + [response]}

        # Create the graph
        builder = StateGraph(MessageState)
        builder.add_node("llm", llm_node)

        # Add edges
        builder.add_edge(START, "llm")
        builder.add_edge("llm", END)

        return builder
