from typing import List, Sequence

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    previous_message = ""
    for message in state:
        if isinstance(message, AIMessage):
            current_message = message.content
            if current_message == previous_message:
                return END
            previous_message = current_message

    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
    The Beaver Moon rises tonight, Nov. 15, as 2024’s last supermoon. This full moon, which may get its name from beavers preparing for winter, will appear a bit bigger and brighter than usual. Details: https://go.nasa.gov/3Z5IhMi"
    """)

    graph.invoke(input=inputs)
