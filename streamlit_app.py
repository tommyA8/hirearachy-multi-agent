import os
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore")
import uuid
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import streamlit as st
from langchain_core.messages import HumanMessage
from workflows.chat_cm import chatcm_agent
from agents import *
from model.user import UserContext

POSTGRES_URI = os.getenv("POSTGRES_URI")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
EMBEDED_MODEL_NAME = os.getenv("EMBEDED_MODEL_NAME")
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY")

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
    return chatcm_agent()

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
        if len(st.session_state.history) > 1:
            response = agent.invoke(
                {
                    "messages": [HumanMessage(content=prompt)]
                },
                config={
                    "callbacks": [cb],
                    "configurable": {"thread_id": st.session_state.thread_id},
                },
            )
        else:
            response = agent.invoke(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "user": UserContext(user_id=user_id, company_id=company_id, project_id=project_id, tool_permissions=None),
                },
                config={
                    "callbacks": [cb],
                    "configurable": {"thread_id": st.session_state.thread_id},
                },
            )

        answer = response["messages"][-1].content.split("</think>\n")[-1]
        st.write(answer)
        st.session_state.history.append(("assistant", answer))