import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from graph import graph, GraphState

state = GraphState(
    messages=[HumanMessage("Hi there, am I allowed to cancel my order?")]
)

config = RunnableConfig({"configurable": {"thread_id": str(uuid.uuid4())}})

events = graph.stream(state, config, stream_mode="values")

for event in graph.stream(state, config, stream_mode="values"):
    messages = event.get("messages")
    if messages and messages[-1]:
        print(messages[-1].pretty_repr(html=True))
