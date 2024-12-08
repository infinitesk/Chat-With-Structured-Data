import pandas as pd
import streamlit as st
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langsmith import traceable
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv()

# streamlit web app configuration
st.set_page_config(
    page_title="Chat With Structured Data",
    page_icon="💬",
    layout="centered"
)


def read_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


# streamlit page title
st.title("🤖 Chat With Structured Data")

# initialize chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# initiate df iin session state
if "df" not in st.session_state:
    st.session_state.df = None


# file upload widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    st.session_state.df = read_data(uploaded_file)
    st.write("DataFrame Preview:")
    st.dataframe(st.session_state.df.head())


# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message
user_prompt = st.chat_input("Ask LLM...")


@traceable
def get_response(messages):
    response = pandas_df_agent.invoke(messages)
    assistant_response = response["output"]
    return assistant_response


if user_prompt:
    # add user's message to chat history and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user","content": user_prompt})

    # loading the LLM
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    #uncomment below to use ollama
    # llm = ChatOllama(model="phi3.5:latest" , temperature=0)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        st.session_state.df,
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True
    )

    messages = [
        {"role":"system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]
    

    assistant_response = get_response(messages=messages)

    st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})

    # display LLM response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)


