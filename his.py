import os
import faiss
import pickle
import numpy as np
import openai
import time
import streamlit as st

# Paths to the FAISS index and chunk files
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_FILE_PATH = "chunks.pkl"
EMBEDDINGS_FILE_PATH = "embeddings.pkl"
TEXT_FILE_PATH = "fair (1).txt"

# Set OpenAI API details
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_API_BASE', "https://aj-open-ai.openai.azure.com/")
openai.api_version = "2023-12-01-preview"
openai.api_key = os.getenv('OPENAI_API_KEY', "7ec51e85779e41bca5bc29b9e533ee47")


# Function to read text from a local file
def read_text_from_local(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Function to chunk text by paragraphs
def chunk_text_by_paragraphs(text, max_chunk_size=1000):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Helper function to normalize embeddings
def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

# Load FAISS index
def load_faiss_index(faiss_index_path):
    return faiss.read_index(faiss_index_path)

# Load chunks
def load_chunks(chunks_file_path):
    with open(chunks_file_path, "rb") as f:
        return pickle.load(f)

# Load embeddings
def load_embeddings(embeddings_file_path):
    with open(embeddings_file_path, "rb") as f:
        return pickle.load(f)

# Embed query using OpenAI API
def embed_query(query):
    try:
        response = openai.Embedding.create(
            deployment_id="text-embedding-ada-002",
            input=[query]
        )
        embedding = response['data'][0]['embedding']
        return normalize_embedding(embedding)
    except Exception as e:
        st.error(f"Error generating embedding for query: {e}")
        return None

# Query the FAISS index
def query_faiss_index(index, query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    return indices[0] if len(indices) > 0 else []

# Rerank chunks based on cosine similarity
def rerank_chunks(query_embedding, chunks, retrieved_indices, chunk_embeddings):
    similarities = []
    for i in retrieved_indices:
        chunk_embedding = chunk_embeddings[i]
        similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
        similarities.append((chunks[i], similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Select top relevant chunks
def select_top_chunks(relevant_chunks, top_n=5):
    return " ".join([chunk for chunk, _ in relevant_chunks[:top_n]])

# Generate dynamic prompt
def get_dynamic_prompt(query, context):
    return f"Based on the following context, answer the query briefly: '{query}'\n\nContext:\n{context}\n\nPlease provide a short and to-the-point answer."

# Get GPT-35-Turbo response
def get_gpt_response(messages):
    try:
        response = openai.ChatCompletion.create(
            deployment_id="gpt-35-turbo",
            messages=messages,
            max_tokens=80
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error getting GPT response: {e}")
        return "Error generating response."


# Main function to load FAISS index, embeddings, and chunks
def load_index_data():
    if os.path.exists(FAISS_INDEX_PATH):
        index = load_faiss_index(FAISS_INDEX_PATH)
        chunks = load_chunks(CHUNKS_FILE_PATH)
        embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)
        return index, chunks, embeddings
    else:
        st.error("FAISS index or embeddings not found. Please generate them first.")
        return None, None, None


# Streamlit UI
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #003366;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .content-box {
        padding: 15px;
        background-color: #F5F5F5;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .button {
        background-color: #003366;
        color: #FFFFFF;
    }
    .button:hover {
        background-color: #002244;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Unilever Multi Product AI Agent</h1>", unsafe_allow_html=True)

# Query input
query = st.text_input("Enter your FAQ:", placeholder="Type your query here...", help="Input your question or query here.", max_chars=150)
submit_query = st.button("Submit Query", key="submit_query", help="Click to submit your query.", use_container_width=True)

if submit_query and query:
    query_embedding = embed_query(query)

    if query_embedding is not None:
        # Load the FAISS index, chunks, and embeddings
        index, chunks, chunk_embeddings = load_index_data()

        if index is not None:
            retrieved_indices = query_faiss_index(index, query_embedding, k=10)
            if len(retrieved_indices) > 0:
                relevant_chunks = rerank_chunks(query_embedding, chunks, retrieved_indices, chunk_embeddings)
                selected_context = select_top_chunks(relevant_chunks, top_n=5)

                # Generate response using GPT-35-Turbo
                dynamic_prompt = get_dynamic_prompt(query, selected_context)
                response = get_gpt_response([{"role": "system", "content": dynamic_prompt}])

                # Display response directly
                st.markdown("<h2 class='section-header'>Response</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='content-box'>{response}</div>", unsafe_allow_html=True)
            else:
                st.error("No relevant chunks found for the query.")
        else:
            st.error("Error loading FAISS index or embeddings.")
else:
    st.info("Please enter a query to get started.")










# import streamlit as st
# import faiss
# import pickle
# import numpy as np
# import openai
# import os

# # Paths to the FAISS index and chunk files
# FAISS_INDEX_PATH = "faiss_index.bin"
# CHUNKS_FILE_PATH = "chunks.pkl"
# EMBEDDINGS_FILE_PATH = "embeddings.pkl"

# # Set OpenAI API details
# openai.api_type = "azure"
# openai.api_base = os.getenv('OPENAI_API_BASE', "https://aj-open-ai.openai.azure.com/")
# openai.api_version = "2023-12-01-preview"
# openai.api_key = os.getenv('OPENAI_API_KEY', "7ec51e85779e41bca5bc29b9e533ee47")

# # Helper function to normalize embeddings
# def normalize_embedding(embedding):
#     return embedding / np.linalg.norm(embedding)

# # Load FAISS index
# def load_faiss_index(faiss_index_path):
#     return faiss.read_index(faiss_index_path)

# # Load chunks
# def load_chunks(chunks_file_path):
#     with open(chunks_file_path, "rb") as f:
#         return pickle.load(f)

# # Load embeddings
# def load_embeddings(embeddings_file_path):
#     with open(embeddings_file_path, "rb") as f:
#         return pickle.load(f)

# # Embed query using OpenAI API
# def embed_query(query):
#     try:
#         response = openai.Embedding.create(
#             deployment_id="text-embedding-ada-002",
#             input=[query]
#         )
#         embedding = response['data'][0]['embedding']
#         return normalize_embedding(embedding)
#     except Exception as e:
#         st.error(f"Error generating embedding for query: {e}")
#         return None

# # Query the FAISS index
# def query_faiss_index(index, query_embedding, k=5):
#     distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
#     return indices[0] if len(indices) > 0 else []

# # Rerank chunks based on cosine similarity
# def rerank_chunks(query_embedding, chunks, retrieved_indices, chunk_embeddings):
#     similarities = []
#     for i in retrieved_indices:
#         chunk_embedding = chunk_embeddings[i]
#         similarity = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
#         similarities.append((chunks[i], similarity))
#     return sorted(similarities, key=lambda x: x[1], reverse=True)

# # Select top relevant chunks
# def select_top_chunks(relevant_chunks, top_n=5):
#     return " ".join([chunk for chunk, _ in relevant_chunks[:top_n]])

# # Generate dynamic prompt
# def get_dynamic_prompt(query, context):
#     return f"Based on the following context, answer the query briefly: '{query}'\n\nContext:\n{context}\n\nPlease provide a short and to-the-point answer."

# # Get GPT-35-Turbo response
# def get_gpt_response(messages):
#     try:
#         response = openai.ChatCompletion.create(
#             deployment_id="gpt-35-turbo",
#             messages=messages,
#             max_tokens=80
#         )
#         return response.choices[0].message['content'].strip()
#     except Exception as e:
#         st.error(f"Error getting GPT response: {e}")
#         return "Error generating response."

# # Initialize conversation memory
# if "conversation" not in st.session_state:
#     st.session_state.conversation = []

# # Streamlit UI
# st.markdown("""
#     <style>
#     .title {
#         text-align: center;
#         color: #003366;
#         font-size: 36px;
#         margin-bottom: 20px;
#     }
#     .content-box {
#         padding: 15px;
#         background-color: #F5F5F5;
#         border-radius: 8px;
#         margin-bottom: 20px;
#         box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
#     }
#     .button {
#         background-color: #003366;
#         color: #FFFFFF;
#     }
#     .button:hover {
#         background-color: #002244;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("<h1 class='title'>Unilever Multi Product AI Agent</h1>", unsafe_allow_html=True)

# # Query input
# query = st.text_input("Enter your FAQ:", placeholder="Type your query here...", help="Input your question or query here.", max_chars=150)
# submit_query = st.button("Submit Query", key="submit_query", help="Click to submit your query.", use_container_width=True)

# if submit_query and query:
#     query_embedding = embed_query(query)

#     if query_embedding is not None:
#         # Load the FAISS index, chunks, and embeddings
#         index = load_faiss_index(FAISS_INDEX_PATH)
#         chunks = load_chunks(CHUNKS_FILE_PATH)
#         chunk_embeddings = load_embeddings(EMBEDDINGS_FILE_PATH)

#         retrieved_indices = query_faiss_index(index, query_embedding, k=10)
#         if len(retrieved_indices) > 0:
#             relevant_chunks = rerank_chunks(query_embedding, chunks, retrieved_indices, chunk_embeddings)
#             selected_context = select_top_chunks(relevant_chunks, top_n=5)

#             # Update conversation history
#             st.session_state.conversation.append({"role": "user", "content": query})

#             # Append previous conversation to context
#             conversation_history = st.session_state.conversation.copy()
#             conversation_history.append({"role": "system", "content": selected_context})

#             # Generate response using GPT-35-Turbo
#             response = get_gpt_response(conversation_history)
#             st.session_state.conversation.append({"role": "assistant", "content": response})

#             # Display response directly
#             st.markdown("<h2 class='section-header'>Response</h2>", unsafe_allow_html=True)
#             st.markdown(f"<div class='content-box'>{response}</div>", unsafe_allow_html=True)
#         else:
#             st.error("No relevant chunks found for the query.")
#     else:
#         st.error("Error generating query embedding.")

# # Display previous conversation
# if st.session_state.conversation:
#     st.markdown("<h2 class='section-header'>Conversation History</h2>", unsafe_allow_html=True)
#     for message in st.session_state.conversation:
#         role = "User" if message['role'] == 'user' else "Assistant"
#         st.markdown(f"**{role}:** {message['content']}")
