
import uuid
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st
from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent, load_tools
# ----- your agent bits (unchanged) -----
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from main import HierarchicalAgent
from utils.model_state import UserContext

def ensure_defaults():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    if "user_id" not in st.session_state:
        st.session_state.user_id = 1
    if "company_id" not in st.session_state:
        st.session_state.company_id = 1
    if "project_id" not in st.session_state:
        st.session_state.project_id = 1

ensure_defaults()

st.sidebar.image("assets/logo_siteAround-sm.svg")

# ---- Thread & user context UI (stable across reruns) ----
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"

st.sidebar.header("Session")
st.session_state.thread_id = st.sidebar.text_input("thread_id", st.session_state.thread_id)

user_id    = st.sidebar.number_input("user_id",    min_value=1, value=1, step=1)
company_id = st.sidebar.number_input("company_id", min_value=1, value=1, step=1)
project_id = st.sidebar.number_input("project_id", min_value=1, value=1, step=1)


@st.cache_resource(show_spinner=False)
def build_agent_once():
    # Persistent checkpointer
    cp = MemorySaver()
    #SQLiteSaver.from_conn_string("checkpoints.db")

    graph = HierarchicalAgent()
    graph.help_desk = ChatOllama(model="qwen3:1.7b", temperature=0.1)
    graph.router    = ChatOllama(model="qwen3:1.7b", temperature=0.1)
    graph.database  = ChatOllama(model="sqlcoder:7b", temperature=0)

    agent = graph.build(checkpointer=cp, save_graph=False)
    return agent

agent = build_agent_once()


# Display-only UI history (optional)
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).write(content)

if prompt := st.chat_input("Type your message"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container())
        response = agent.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "user": UserContext(user_id=user_id, company_id=company_id, project_id=project_id),
            },
            config={
                "callbacks": [cb],
                # 👇 This must stay the same to reuse memory
                "configurable": {"thread_id": st.session_state.thread_id},
            },
        )

        answer = answer = response["messages"][-1].content.split("</think>\n")[-1]
        st.write(answer)
        st.session_state.history.append(("assistant", answer))