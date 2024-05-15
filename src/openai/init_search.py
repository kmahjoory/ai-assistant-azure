import os
import openai
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswVectorSearchAlgorithmConfiguration,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
)
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Load environment variables from the .env file
load_dotenv()

# Azure Cognitive Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-1"

# Azure OpenAI Configuration
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Directory containing the data to be processed
DATA_DIR = "data/"

def load_and_split_documents() -> list[dict]:
    """
    Loads documents from the specified directory and splits them into smaller chunks.
    Returns a list of dictionaries for indexing.
    """
    # Load documents from directory
    loader = DirectoryLoader(
        DATA_DIR, loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    documents = loader.load()

    # Split documents using a markdown-specific text splitter
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=6000, chunk_overlap=100
    )
    split_docs = splitter.split_documents(documents)

    # Convert LangChain Document objects to dictionaries for indexing
    final_docs = [
        {
            "id": str(i),
            "content": doc.page_content,
            "sourcefile": os.path.basename(doc.metadata["source"]),
        }
        for i, doc in enumerate(split_docs)
    ]

    return final_docs

def create_search_index(name: str) -> SearchIndex:
    """
    Creates an Azure Cognitive Search index configuration.
    """
    # Define the fields to be indexed
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,  # Size for 'text-embedding-ada-002'
            vector_search_configuration="default",
        ),
    ]

    # Configure semantic settings, prioritizing the 'content' field
    semantic_settings = SemanticSettings(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=PrioritizedFields(
                    title_field=None,
                    prioritized_content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    # Set up vector search using HNSW (Hierarchical Navigable Small World) with cosine distance
    vector_search = VectorSearch(
        algorithm_configurations=[
            HnswVectorSearchAlgorithmConfiguration(
                name="default",
                kind="hnsw",
                parameters=HnswParameters(metric="cosine"),
            )
        ]
    )

    # Create and return the SearchIndex object
    return SearchIndex(
        name=name,
        fields=fields,
        semantic_settings=semantic_settings,
        vector_search=vector_search,
    )

def initialize_search_index(search_index_client: SearchIndexClient):
    """
    Initializes and uploads data to an Azure Cognitive Search index using vector search.
    """
    # Load and split documents
    docs = load_and_split_documents()

    # Generate embeddings for each document
    for doc in docs:
        embedding_response = openai.Embedding.create(
            engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=doc["content"]
        )
        doc["embedding"] = embedding_response["data"][0]["embedding"]

    # Create the search index
    index = create_search_index(AZURE_SEARCH_INDEX_NAME)
    search_index_client.create_or_update_index(index)

    # Upload documents to Azure Cognitive Search
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    search_client.upload_documents(documents=docs)

def delete_search_index(search_index_client: SearchIndexClient):
    """
    Deletes the specified Azure Cognitive Search index.
    """
    search_index_client.delete_index(AZURE_SEARCH_INDEX_NAME)

def main():
    # Set OpenAI API credentials
    openai.api_type = AZURE_OPENAI_API_TYPE
    openai.api_base = AZURE_OPENAI_API_BASE
    openai.api_version = AZURE_OPENAI_API_VERSION
    openai.api_key = AZURE_OPENAI_API_KEY

    # Initialize the SearchIndexClient
    search_index_client = SearchIndexClient(
        AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    # Initialize the search index with vector search
    initialize_search_index(search_index_client)
    # Uncomment the following line to delete the search index when needed
    # delete_search_index(search_index_client)

if __name__ == "__main__":
    main()
