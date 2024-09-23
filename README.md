# RAG-Model-for-QA-Bot
### Retrieval-Augmented Generation (RAG) Model for Question Answering Bot

#### Objective
- The goal of this project is to create a QA bot that leverages a Retrieval-Augmented Generation (RAG) approach. This means combining the capabilities of both information retrieval (vector database - Pinecone) and generation (OpenAI GPT) to provide coherent and contextually relevant answers.

#### Architecture and Flow
1. **Data Loading and Preprocessing**: Load documents (I have used PDFs and can be modified easily for other data sources) and split them into chunks of text.
2. **Embeddings**: Each chunk of text is converted into embeddings using OpenAI's embedding model (`text-embedding-ada-002`).
3. **Vector Storage**: These embeddings are stored in Pinecone, a vector database optimized for similarity search.
4. **Retrieval**: When a user submits a query, relevant document chunks are retrieved from Pinecone based on similarity.
5. **Generative Model**: A generative model (GPT-3.5-turbo) uses the retrieved document chunks to generate a coherent response to the query.
6. **Question Answering**: The system responds with an answer that is grounded in the retrieved information.

#### Tools Used:
- **LangChain**: For managing prompts, chains, and interaction between components.
- **Pinecone**: As a vector database for efficient storage and retrieval of document embeddings.
- **OpenAI**: For embedding and generating responses.


## Part-1
- Documentation and explaination of task is clearly provided in notebook.

## Part-2
- Documentation and Explanation is clearly provided in the code.
- Access streamlit deployed app here (https://rag-model-for-app-bot-part-2.streamlit.app/)
  
  ![](https://github.com/bhargav0807/RAG-Model-for-QA-Bot/blob/main/UI.png)

- We can upload a PDF and query it and As i have already uploaded [budget_speech.pdf](https://github.com/bhargav0807/RAG-Model-for-QA-Bot/blob/main/budget_speech.pdf), Gen AI Engineer / ML Engineer Assignment Pdf are already in vectorDB, we can even query them.
