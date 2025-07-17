### Retrieval Augmented Generation (RAG)

This project implements a **Retrieval Augmented Generation (RAG)** system. RAG is a technique that enhances the capabilities of Large Language Models (LLMs) by allowing them to access and incorporate up-to-date and specific information from an external knowledge base.

* **Problem RAG Solves:** LLMs are powerful, but their knowledge is limited to their training data (which can be outdated or lack domain-specific information). They can also "hallucinate" (make up facts). RAG addresses this by giving the LLM a "memory" or a "knowledge base" to consult.
* **How it Works:** Instead of asking an LLM a question directly, a RAG system first *retrieves* relevant information from your custom documents. This retrieved information is then *augmented* (added) to the prompt given to the LLM. The LLM then *generates* an answer based on both its pre-trained knowledge and the provided context.

In this specific project, your custom knowledge base consists of text files (e.g., news articles) stored locally.

### Code Explanation (`app.py` Workflow)

The `app.py` script orchestrates the RAG pipeline through the following main steps:

1.  **Configuration & Initialization:**
    * It loads your **OpenRouter API key** from a `.env` file for security.
    * It sets the `OPENROUTER_BASE_URL` (`https://openrouter.ai/api/v1`) to direct all API calls (both for embeddings and chat completions) to OpenRouter's service.
    * It initializes a **ChromaDB client** in `PersistentClient` mode, meaning your database will be saved to a `chroma_persistent_storage` folder on your disk, so you don't lose your data between runs.
    * It defines a **ChromaDB Collection** (`document_qa_collection`) which is like a table where your data will live. Crucially, this collection is configured with an **embedding function**.

2.  **Document Ingestion (Building the Knowledge Base):**
    * **`load_documents_from_directory`:** Scans the `news_articles` directory, reads all `.txt` files, and extracts their content.
    * **`split_text`:** Breaks down potentially large text documents into smaller, manageable `chunks` (e.g., 1000 characters with a 20-character overlap). This is essential because LLMs have a limited "context window" (how much text they can process at once), and smaller chunks improve the relevance of search results.
    * **`collection.upsert()`:** Each of these text chunks is then `upsert`ed (inserted or updated) into your ChromaDB collection. When you `upsert` a document, because the `embedding_function` was already configured on the `collection`, ChromaDB automatically takes the text, generates its numerical **vector embedding** using the specified embedding model (`all-MiniLM-L6-v2` in your case), and stores both the text and its embedding.

3.  **Querying and Retrieval:**
    * **`query_documents`:** When a user asks a `question` (e.g., "tell me about databricks"), ChromaDB's `query` method is invoked.
    * Similar to ingestion, ChromaDB automatically uses the configured embedding function to convert your `question` into a vector embedding.
    * It then performs a **similarity search** in the vector space, finding the `n_results` (e.g., 2) document chunks whose embeddings are numerically closest to the query embedding. These are your `relevant_chunks`.

4.  **Response Generation (Augmenting the LLM):**
    * **`generate_response`:** This function takes the original `question` and the `relevant_chunks` retrieved from ChromaDB.
    * It constructs a detailed **`prompt`** for the LLM. This prompt includes:
        * Instructions on the LLM's role (e.g., "You are an assistant for question-answering tasks.").
        * A clear directive to *use the provided context* for the answer.
        * The actual `Context:` (the `relevant_chunks` joined together).
        * The original `Question:`.
    * **`client.chat.completions.create()`:** This makes an API call to the `deepseek/deepseek-r1-0528-qwen3-8b:free` LLM via OpenRouter, passing the crafted prompt.
    * The LLM then generates an `answer` based on this augmented context, providing an informed response that draws from your specific `news_articles`.

### Tools Used

1.  **Database for Embeddings:**
    * **ChromaDB**: This is your chosen **vector database**. It's used to store your document chunks and their corresponding numerical vector embeddings. It's designed for efficient similarity search, quickly finding chunks that are semantically related to a given query. Its `PersistentClient` ensures your data is saved to disk (`chroma_persistent_storage` folder) and reloaded automatically.

2.  **Tool for Embedding Generation:**
    * **`chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction` with `all-MiniLM-L6-v2`:**
        * This is the specific model used to convert your text documents (and queries) into numerical vector embeddings.
        * `SentenceTransformerEmbeddingFunction` is a utility from ChromaDB that allows you to use models from the popular `sentence-transformers` library.
        * `all-MiniLM-L6-v2` is a pre-trained **Sentence Transformer model** from Hugging Face. Crucially, this model is typically run **locally** on your machine. This means you do not incur API costs for generating embeddings with this specific model, and the embedding process is faster as it doesn't rely on network calls to an external service for every chunk.

3.  **Free API Keys / Service for LLM Interaction:**
    * **OpenRouter (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`):**
        * OpenRouter acts as a unified API gateway to various LLMs from different providers.
        * You use your **free OpenRouter API key** to authenticate your requests.
        * By setting the `base_url` of the `openai` Python client to `https://openrouter.ai/api/v1`, you instruct the client to send all its requests to OpenRouter's servers.
        * This allows you to access models hosted by OpenRouter using the familiar OpenAI API syntax, without directly dealing with multiple model providers' unique APIs.

4.  **Model to Generate Answers (LLM):**
    * **`deepseek/deepseek-r1-0528-qwen3-8b:free` (accessed via OpenRouter):**
        * This is the specific Large Language Model (LLM) that generates the final answers to your questions.
        * It's an open-source model made available through OpenRouter's service, and as indicated by `:free`, it likely falls under OpenRouter's free tier or available models. It performs the "Generation" part of RAG, formulating human-like responses based on the prompt and the retrieved context.