
import uuid
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
# ----- your agent bits (unchanged) -----
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from main import HierarchicalAgent
from utils.model_state import UserContext

# Build the agent ONCE, keep shared memory across requests
memory = MemorySaver()
graph = HierarchicalAgent()
graph.help_desk = ChatOllama(model="qwen3:1.7b", temperature=0.1)
graph.router = ChatOllama(model="qwen3:1.7b", temperature=0.1)
graph.database = ChatOllama(model="qwen3:1.7b", temperature=0.1)
agent = graph.build(checkpointer=memory, save_graph=False)

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

# ----------------------------
# Sidebar: รับ thread_id + context
# ----------------------------
st.sidebar.header("Session / Thread")
col1, col2 = st.sidebar.columns([3, 1])

with col1:
    thread_id_input = st.text_input(
        "Thread ID",
        value=st.session_state.thread_id,
        help="กำหนดรหัส thread ที่จะใช้กับ Memory/Checkpoint ของ Agent",
        key="thread_id_text",
    )
with col2:
    if st.button("Random"):
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.thread_id_text = st.session_state.thread_id
        st.rerun()

# Sync ค่าจาก text_input -> session_state
st.session_state.thread_id = thread_id_input.strip() or st.session_state.thread_id

st.sidebar.header("User Context")
st.session_state.user_id = st.sidebar.number_input("user_id", min_value=1, value=st.session_state.user_id, step=1)
st.session_state.company_id = st.sidebar.number_input("company_id", min_value=1, value=st.session_state.company_id, step=1)
st.session_state.project_id = st.sidebar.number_input("project_id", min_value=1, value=st.session_state.project_id, step=1)

# st.sidebar.caption(
#     "คำแนะนำ: ใช้รูปแบบเช่น "
#     f"`user:{st.session_state.user_id}|company:{st.session_state.company_id}|project:{st.session_state.project_id}` "
#     "เพื่อผูกความจำกับผู้ใช้/โปรเจ็กต์"
# )

# ----------------------------
# ส่วนสนทนา
# ----------------------------
st.title("Agent Chat")

# แสดงข้อความแชตที่ค้างไว้ (ถ้ามี)
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).write(content)


if prompt := st.chat_input():
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        # Single invoke. Put callbacks & thread_id in the config dict.
        response = agent.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "user": UserContext(user_id=1, company_id=1, project_id=1),
            },
            config={
                "callbacks": [st_callback],
                "configurable": {"thread_id": "streamlit-test"},
            },
        )

        # Safely extract the final text
        if isinstance(response, dict) and "messages" in response and response["messages"]:
            answer = response["messages"][-1].content.split("</think>\n")[-1]
            
        elif isinstance(response, dict) and "output" in response:
            # some executors return "output"
            answer = response["output"]
        else:
            answer = str(response)

        st.write(answer)