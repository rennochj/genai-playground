import json
from typing import Any

from dotenv import load_dotenv
from graphs import XSimpleGraphConfig
from langchain.load.dump import dumps
from langchain.schema.runnable import Runnable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from model import XOpenAIConfig

# ------------------------------------------------------------------------------------------------------------------------


def main(llm: Runnable[list[Any], Any], graph: StateGraph) -> None:
    initial_messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="generate a haiku about the ocean"),
    ]

    result = graph.compile().invoke({"messages": initial_messages})

    print(dumps(result, indent=2))


# ------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    load_dotenv()

    model = XOpenAIConfig().build()
    graph = XSimpleGraphConfig().build(model)

    main(model, graph)
