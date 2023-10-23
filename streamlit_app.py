import os
import streamlit as st
import requests
import time


# LangChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import openai


# Pinecone
import pinecone

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

openai.api_key = OPENAI_API_KEY

pinecone.init(       
 api_key=PINECONE_API_KEY,
 environment=PINECONE_INDEX_NAME 
)      
index = pinecone.Index('chatbot')

# OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Hugging Face Credentials
hf_token = os.getenv('HF_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/tmcgirr/BERT-squad-chatbot-AAI"
headers = {"Authorization":  f"Bearer {hf_token}"}

def query(payload, retries=5, delay=5):
    for i in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        response_json = response.json()
        
        # If the model is still loading, wait and retry
        if 'error' in response_json and 'loading' in response_json['error']:
            print(f"Model is still loading. Waiting for {delay} seconds before retrying...")
            time.sleep(delay)
            continue
        
        return response_json

    # If the model is still loading after all retries, return the last response
    return response_json

# Check if Hugging Face credentials are provided
if not hf_token:
    st.error('HuggingFace Login credentials not provided!')
    st.stop()

# App title
st.set_page_config(page_title="BERT Chatbot")

# Define VectorStore
vectorstore = Pinecone(index, embeddings.embed_query, "text")


# Initialize conversation history if it doesn't exist
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "assistant", "content": "Hello! What is your question?"}]

st.sidebar.header("Your Context:")
st.sidebar.markdown("Provide your context here for a more direct QA experience, otherwise leave blank for a more general QA")
user_context = st.sidebar.text_area("", value="", height=500)

# Display conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating search report
def generate_search_report(prompt: str, k: int):
    xq = openai.Embedding.create(input=[prompt], engine="text-embedding-ada-002")['data'][0]['embedding']
    docs = index.query([xq], top_k=k, include_metadata=True)
    return docs

def generate_response(prompt_input, user_context=None):
    # If user-provided context is not empty, use it instead
    if user_context:
        context = user_context
        print("\n\nStart of user_context")
        print(context)
        print("End of user_context\n\n")
    else:
        # Generate search based on question (and the last 500 characters of the conversation history)
        conversation_history = [message['content'] for message in st.session_state.conversation]
        
        # Generate search based on question and the last 500 characters of the conversation history
        search_context = " ".join(conversation_history[-500:]) + " " + prompt_input
        
        # generate search based on question
        docs = generate_search_report(search_context, 5)

        # Extract the context from the output
        context = " ".join([match['metadata']['text'] for match in docs['matches']])
        print("\n\nStart of search context")
        print(context)
        print("End of search context\n\n")
    
    payload = {
        "inputs": {
            "question": prompt_input,
            "context": context
        },
    }
    
    output = query(payload)
    print("\n\nStart of output")
    print(output)
    print("End of output\n\n")

    if 'answer' in output and 'score' in output:
        score = output['score']
        answer = output['answer']
        
        # Add a prelude based on the score
        if score < 0.5:
            answer = "I'm unsure, but my guess might be " + answer + ". " "\nPlease restate the question and add more context."
        elif score < 0.75:
            answer = "I think it's " + answer
        else:
            answer = answer[0].upper() + answer[1:]
        return answer
    else:
        # Handle error: 'answer' not in output
        return "I'm sorry, I couldn't find an answer to your question. Please restate the question and add more context."

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.conversation[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, user_context)  # Pass user_context here
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.conversation.append(message)

