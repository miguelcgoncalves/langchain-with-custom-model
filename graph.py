from typing import Annotated, List, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from tools import lookup_policy
from custom_model import CustomModel


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class Assistant:
    def __init__(self, model: BaseChatModel, tools: List[BaseTool]):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful customer support assistant."),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.runnable = prompt | model.bind_tools(tools)

    def __call__(self, state):
        return {"messages": self.runnable.invoke(state)}


def handle_tool_error(state):
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(content=repr(error), tool_call_id=tc["id"]) for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list):
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


model = CustomModel()
tools = [lookup_policy]
builder = StateGraph(GraphState)
builder.add_node("assistant", Assistant(model, tools))
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=MemorySaver())
