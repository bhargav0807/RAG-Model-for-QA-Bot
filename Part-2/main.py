# Importing Required Libraries
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from pinecone import Pinecone as PC
from pinecone import ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
from io import BytesIO
import time

# Load environment variables
load_dotenv()

# Initialize Streamlit App
st.title('Part-2: Interactive QA Bot Interface')

# PDF Upload Interface
uploaded_file = st.file_uploader("Upload a PDF file / Query on already uploaded pdfs", type="pdf")

# Function to read and process the PDF file
def read_doc(file):
    if isinstance(file, str):  # If the file is a string (file path), read from disk
        loader = PyPDFLoader(file)
    else:  # If it's an uploaded file (BytesIO object)
        pdf_data = BytesIO(file.read())  # Read the uploaded file into memory
        loader = PyPDFLoader(pdf_data)  # Pass the in-memory file to PyPDFLoader
    docs = loader.load()
    return docs

# Chunking the data
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

# If a file is uploaded, read and chunk the document
if uploaded_file is not None:
    with st.spinner('Processing PDF...'):
        doc = read_doc(uploaded_file)
        chunks = chunk_data(docs=doc)

        # Display the number of chunks created
        st.success(f"Document Processed and Chunked! ")

# Embedding model configuration
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to Pinecone and Create Index
pc = PC(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "docs-rag"

existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# Check if index already exists
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )
    # Wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Store document chunks in Pinecone
if uploaded_file is not None:
    with st.spinner('Storing embeddings in Pinecone...'):
        index = Pinecone.from_documents(chunks, embedding_model, index_name=index_name)
        st.success('Embeddings successfully stored!')
else:
    from langchain_pinecone import PineconeVectorStore
    index = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

# Chat model configuration
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2
)

# Function to augment the prompt with template
def augment_prompt_with_template(query, k=5):
    # Retrieve top 5 relevant chunks from the vectorstore
    results = index.similarity_search(query, k=k)
    source_knowledge = "\n".join([x.page_content for x in results])
    source = results[0].metadata
    
    # Define the prompt template with placeholders
    template = """
                Your task is to create an answer to the user's query using the information
                from the context provided. Follow these steps to generate the response:

                Step 1: Analyze the user-provided query: {query}
                Step 2: Review the relevant context provided: {contexts}
                Step 3: Generate a concise, clear, and informative response based on the context, 
                        ensuring it addresses the query and maintains accuracy.
                Step 4: If the query is not completly related to context and you dont know the answer 
                        please say that Query is not related to context.
    """
    
    # Initialize the LangChain PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["contexts", "query"],
        template=template
    )
    
    # Create the final augmented prompt by filling the template
    augmented_prompt = prompt_template.format(contexts=source_knowledge, query=query)
    
    return augmented_prompt, source

# Function to run the augmented prompt
def run_augmented_prompt(query):
    messages = [SystemMessage(content="You are a helpful assistant.")]
    augmented_prompt_text, source = augment_prompt_with_template(query)
    augmented_prompt = HumanMessage(content=augmented_prompt_text)
    messages.append(augmented_prompt)
    res = chat(messages)
    return res.content, source

# Chat Interface
query = st.text_input("Ask a question about any Uploaded PDF")
if st.button("Submit"):
      with st.spinner('Fetching answer...'):
          answer, source = run_augmented_prompt(query)
          st.success(f"Answer: {answer}")
          st.write(f"Source: {source}")
