# pip install -U langgraph langchain-core
import os
from pprint import pprint
from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import BraveSearch
from dotenv import load_dotenv
load_dotenv(override=True)

api_key=os.getenv("BRAVE_API_KEY")

# ---------- 1) Tools ----------
@tool
def web_search(q: str) -> str:
    """Dummy search results"""
    search_tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
    res = search_tool.invoke({"query": q})
    return f"Search results for {q}: {res}"

@tool
def outline_tool(topic: str) -> List[str]:
    """Make a quick outline"""
    return [f"Intro to {topic}", "Key facts", "Implications", "Conclusion"]

# ---------- 2) ทีมย่อย (Subgraphs) ----------
# 2.1 Research Team: ใช้ agent แบบ ReAct + tools
def build_research_team(model):
    agent = create_react_agent(model, tools=[web_search], 
                               prompt="You are a research assistant.")
    
    class RState(MessagesState):
        research_notes: str

    g = StateGraph(RState)

    def research_node(state: RState):
        # ต่อบทสนทนาด้วย role แบบแชต
        msgs = state["messages"] + [
            HumanMessage(content="Research and summarize not over 5 sentences about \
                         the user's topic. Use the web_search tool if helpful.")
        ]
        res = agent.invoke({"messages": msgs})
        out_msgs = res["messages"]
        notes = out_msgs[-1].content if out_msgs else ""
        return {"messages": out_msgs, "research_notes": notes}

    g.add_node("research", research_node)
    g.add_edge(START, "research")
    g.add_edge("research", END)
    return g.compile()

# 2.2 Writing Team: วางโครง + เขียนสรุป
def build_writing_team(model):
    agent = create_react_agent(model, tools=[outline_tool], prompt="You are a concise technical writer.")
    
    class WState(MessagesState):
        source: str
        draft: str

    g = StateGraph(WState)

    def write_node(state: WState):
        msgs = state["messages"] + [
            HumanMessage(content=(
                "Create a clear outline (use outline_tool) and then write a concise report "
                "for the user. Base it on the following research notes:\n\n" + state["source"]
            ))
        ]
        res = agent.invoke({"messages": msgs})
        out_msgs = res["messages"]
        draft = out_msgs[-1].content if out_msgs else ""
        return {"messages": out_msgs, "draft": draft}

    g.add_node("write", write_node)
    g.add_edge(START, "write")
    g.add_edge("write", END)
    return g.compile()

# ---------- 3) Top-level Supervisor ----------
# รวมสองทีมเป็น "ลำดับชั้น": Supervisor → Research → Writing → Final
class TopState(MessagesState):
    research_notes: str
    final_report: str

def build_hierarchical_app(model):
    research_team = build_research_team(model)
    writing_team = build_writing_team(model)

    g = StateGraph(TopState)

    def plan(state: TopState):
        # เพิ่ม assistant message อธิบายแผน (ยังคงรูปแบบ role)
        msgs = state["messages"] + [
            AIMessage(content="Plan: I'll research the topic first, then write a concise report.")
        ]
        return {"messages": msgs}

    def call_research(state: TopState):
        # ส่ง messages ลงทีมย่อย แล้วดึง both messages + research_notes กลับขึ้นมา
        res = research_team.invoke({"messages": state["messages"]})
        msgs = res["messages"]
        notes = res.get("research_notes", "")
        # เก็บโน้ตไว้ใน TopState + ต่อบทสนทนา
        msgs = msgs + [AIMessage(content="(Research completed. Notes captured.)")]
        return {"messages": msgs, "research_notes": notes}
    
    def call_writing(state: TopState):
        # ส่ง messages + source (research_notes) ไปยังทีมเขียน
        res = writing_team.invoke({"messages": state["messages"], "source": state["research_notes"]})
        msgs = res["messages"]
        draft = res.get("draft", "")
        return {"messages": msgs, "final_report": draft}

    g.add_node("plan", plan)
    g.add_node("research_team", call_research)
    g.add_node("writing_team", call_writing)

    g.add_edge(START, "plan")
    g.add_edge("plan", "research_team")
    g.add_edge("research_team", "writing_team")
    g.add_edge("writing_team", END)
    return g.compile()

# ---------- 4) Run ----------
model = ChatOllama(model="llama3.2", temperature=0)

model.bind_tools([web_search, outline_tool])
app = build_hierarchical_app(model)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    for step in app.stream(
        # {"user_task": "What is BRICS?"},
        {"messages": [
             HumanMessage(content="Give me a short," \
            " accurate report about "
            " Which is latest year of Thailand-Cambodia conflicts?")
            ]
        },
        stream_mode="values"
    ):
        print(step)
        print("-" * 50)
    # init_messages = [
    #     HumanMessage(content="Give me a short," \
    #     " accurate report about "
    #     " Which is latest year of Thailand-Cambodia conflicts?")
    # ]
    # out = app.invoke({"messages": init_messages})
    # print("=== FINAL REPORT ===")
    # print(out["final_report"])
