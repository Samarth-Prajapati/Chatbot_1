import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set up Streamlit page
st.set_page_config(page_title="Grok ChatBot", page_icon="🤖")
st.title("Grok-Powered ChatBot")
st.write("Ask me anything, and I'll respond with accurate and precise answers!")

# Initialize Groq model via LangChain
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",
    temperature=0.7
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and accurate AI assistant. Provide precise and concise responses to user queries."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create LangChain chain
chain = prompt | llm

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Prepare chat history for LangChain
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # Invoke the chain with user input and chat history
    try:
        response = chain.invoke({"input": user_input, "chat_history": chat_history})
        assistant_reply = response.content

        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.write(assistant_reply)
    except Exception as e:
        st.error(f"Error: {str(e)}")