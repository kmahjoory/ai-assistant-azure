import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import AzureSearch
from langchain.schema import Document

# Load environment variables from .env file.
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"

# Azure Cognitive Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-2"

# Data directory path
DATA_DIR = "data/"

def load_and_split_documents() -> list[Document]:
    """
    Load and split documents into smaller chunks for indexing.
    Returns a list of LangChain Document objects.
    """
    # Initialize the document loader to read markdown files from the directory.
    loader = DirectoryLoader(
        directory_path=DATA_DIR,
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    
    # Load documents from the directory
    documents = loader.load()

    # Initialize text splitter for markdown content
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=6000, chunk_overlap=100
    )
    
    # Split documents into smaller, manageable chunks
    split_documents = text_splitter.split_documents(documents)

    return split_documents

def initialize_vector_store():
    """
    Initialize Azure Cognitive Search vector store and upload data.
    """
    # Load and split the documents for indexing
    documents = load_and_split_documents()

    # Set up OpenAI embeddings for document processing
    embeddings = OpenAIEmbeddings(
        deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )

    # Initialize the Azure Search vector store with embedding function
    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
    )

    # Add documents to the Azure Cognitive Search index
    vector_store.add_documents(documents=documents)

def main():
    """
    Main function to initialize the vector store.
    """
    # Load environment variables and initialize the search index
    initialize_vector_store()

if __name__ == "__main__":
    main()
