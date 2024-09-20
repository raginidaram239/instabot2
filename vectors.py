import os
import faiss
import pickle
import numpy as np
import openai
import time
 
# Paths to the FAISS index and chunk files
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_FILE_PATH = "chunks.pkl"
EMBEDDINGS_FILE_PATH = "embeddings.pkl"
 
# Local file path
TEXT_FILE_PATH = "fair (1).txt"
 
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
 
# Function to generate embeddings in batches
def generate_embeddings_in_batches(chunks, batch_size=10):
    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_API_BASE', "https://aj-open-ai.openai.azure.com/")
    openai.api_version = "2023-12-01-preview"
    openai.api_key = os.getenv('OPENAI_API_KEY', "7ec51e85779e41bca5bc29b9e533ee47")
 
    embeddings = []
    max_tokens = 8192  # Adjust based on OpenAI model's token limit
    for i in range(0, len(chunks), batch_size):
        # Filter out empty or oversized chunks
        batch_chunks = [chunk for chunk in chunks[i:i + batch_size] if chunk.strip() and len(chunk) <= max_tokens]
       
        if not batch_chunks:
            print(f"Skipping empty or invalid chunks in batch {i // batch_size + 1}.")
            continue
 
        try:
            response = openai.Embedding.create(
                deployment_id="text-embedding-ada-002",
                input=batch_chunks
            )
            batch_embeddings = [embedding['embedding'] for embedding in response['data']]
            embeddings.extend(batch_embeddings)
            print(f"Generated embeddings for batch {i // batch_size + 1}.")
            time.sleep(2)  # Adding a delay to manage rate limits
        except openai.error.OpenAIError as e:
            print(f"Error with chunk: {batch_chunks}")
            print(f"OpenAI API error generating embeddings for batch {i // batch_size + 1}: {e}")
            time.sleep(10)  # Delay before retrying on error
        except Exception as e:
            print(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
            time.sleep(10)  # Delay before retrying on error
   
    return embeddings
 
# Function to generate and save FAISS index
def generate_and_save_faiss_index(chunks, embeddings, faiss_index_path, chunks_file_path, embeddings_file_path):
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Total embeddings generated: {len(embeddings)}")
 
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS index saved to {faiss_index_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")
 
    try:
        with open(chunks_file_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"Chunks saved to {chunks_file_path}")
    except Exception as e:
        print(f"Error saving chunks: {e}")
 
    try:
        with open(embeddings_file_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {embeddings_file_path}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
 
# Main processing function
def main():
    text = read_text_from_local(TEXT_FILE_PATH)
    if not text:
        print("No text data retrieved.")
        return
   
    chunks = chunk_text_by_paragraphs(text)
    print(f"Total chunks created: {len(chunks)}")
 
    # Print the first few chunks for verification
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i + 1}:\n{chunk}\n")
   
    embeddings = generate_embeddings_in_batches(chunks)
    if embeddings:
        generate_and_save_faiss_index(chunks, embeddings, FAISS_INDEX_PATH, CHUNKS_FILE_PATH, EMBEDDINGS_FILE_PATH)
    else:
        print("No embeddings generated.")
 
if __name__ == "__main__":
    main()