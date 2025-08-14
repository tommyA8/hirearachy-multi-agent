from utils.tools import get_current_weather, search_tool
from typing import Annotated, List
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages, AIMessage

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
# from IPython.display import Image, display

llm = ChatOllama(model="llama3.2", temperature=0)

# Agent teams
search_agent = create_react_agent(llm, tools=[search_tool])
weather_agent = create_react_agent(llm, tools=[get_current_weather])

class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*(options)]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        response = llm.with_structured_output(Router).invoke(messages)

        # goto = response["next"]
        goto = "FINISH" # FORCE

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node

def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
            
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

def forecast_weather_node(state: State) -> Command[Literal["supervisor"]]:
    result = weather_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="get_current_weather")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


research_supervisor_node = make_supervisor_node(llm, ["search"])#, "get_current_weather"])

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
# research_builder.add_node("get_current_weather", forecast_weather_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()


# display(Image(data=research_graph.get_graph().draw_mermaid_png("graph.png")))
# png = research_graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png)

if __name__ == "__main__":
    question = "Could you named the top 5 rock album all the time?"
    # "Which genre on average has the longest tracks?"

    # for step in research_graph.stream(
    #     {"messages": [{"role": "user", "content": question}]},
    #     stream_mode="values",
    # ):
    #     step["messages"][-1].pretty_print()

    for s in research_graph.stream(
        {"messages": [("user", "Could you named the top 5 rock album all the time?")]},
        {"recursion_limit": 10},
    ):
        print(s)
        print("---")