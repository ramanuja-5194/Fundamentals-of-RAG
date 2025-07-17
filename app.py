import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI 
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1" # OpenRouter API base URL

"""OpenRouter's primary goal is to provide a single API endpoint through which you can access many different large language models (LLMs)
 from various providers (like DeepSeek, Mixtral, Llama, etc.)."""
# --- Embedding Function Setup for Chroma (using an OpenRouter-hosted embedding model) ---
openrouter_embedding_model = "sentence-transformers/all-MiniLM-L6-v2" # Example embedding model on OpenRouter

openrouter_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"   # HuggingFace model
)

# Initialize the Chroma client with persistence
# PersistentClient: This means the database data (documents and their embeddings) will be saved to disk in the specified path. So, if you run the script again, it will load the existing data instead of starting from scratch.
# path="chroma_persistent_storage": The directory where ChromaDB will store its data.
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=openrouter_ef
)

"""The openai Python library is built to interact with the OpenAI API. However, it's designed with a flexible parameter: base_url.
When you initialize client = OpenAI(api_key=your_key), by default, the base_url is set to OpenAI's official API endpoint (https://api.openai.com/v1).
However, you can override this base_url"""
"""By setting base_url to https://openrouter.ai/api/v1, (in OpenAI fn) you are effectively telling the openai library: 
"Hey, use this API key, but send all your requests (for chat completions, for embeddings, etc.) to OpenRouter's server instead of OpenAI's."""
client = OpenAI(
    api_key=openrouter_key,
    base_url=OPENROUTER_BASE_URL # Point the main client to OpenRouter's API base
)

# This commented-out section demonstrates a direct call to the OpenAI GPT-3.5-turbo model without any retrieval from your ChromaDB.
# make a request to OpenAI's chat completion API.
# resp = client.chat.completions.create(
#     model="deepseek/deepseek-r1-0528-qwen3-8b:free",#  Specify the LLM you want to use.
#     messages=[ # The "system" message sets the overall behavior or persona of the AI. It's like giving it initial instructions.
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user", # This is the user's actual question or prompt to the AI.
#             "content": "What is human life expectancy in the United States?",
#         },
#     ],
# )
# print(resp.choices[0].message.content)


# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"): # For each .txt file, it reads its entire content.
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()}) # It creates a list of dictionaries, where each dictionary has an id (the filename) and text (the file's content).
    return documents


# Function to split text into chunks
# This function breaks down large texts into smaller, manageable "chunks."
def split_text(text, chunk_size=1000, chunk_overlap=20):
    # chunk_size=1000: Each chunk will be approximately 1000 characters long.
    # chunk_overlap=20: A small overlap (20 characters) between consecutive chunks helps preserve context that might otherwise be split across chunk boundaries. This improves retrieval accuracy.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk}) # Each chunk gets a unique ID combining the original filename and its chunk number.

print(f"Split documents into {len(chunked_documents)} chunks")


# Upsert documents with embeddings into Chroma
# Chroma will automatically use the configured 'openrouter_ef' to generate embeddings for 'documents'.
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], 
        documents=[doc["text"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    # Chroma will automatically use the configured 'openrouter_ef' to embed the query_texts.
    results = collection.query(
        query_texts=question, # Pass the query text, Chroma will embed it
        n_results=n_results
    )

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a response from OpenRouter (LLM)
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks) #Combines all the retrieved chunks into a single string, separated by newlines, to form the "context."
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer


# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer.content) # Access the content attribute of the message object

