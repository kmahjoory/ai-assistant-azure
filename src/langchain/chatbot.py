"""
Chatbot with context and memory, using updated LangChain and environment config.
"""
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.retrievers import AzureCognitiveSearchRetriever

# Load environment variables.
load_dotenv()

# Azure OpenAI Configuration.
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"

# Azure Cognitive Search Configuration.
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")


class Chatbot:
    """Chatbot that interacts with an LLM while retaining conversation history."""

    def __init__(self):
        # Initialize conversation memory
        self.memory = ConversationBufferMemory()

    def _infer_user_intent(self, query: str) -> str:
        """
        Infer the user's intent based on the conversation history and the latest query.
        """
        # Configure the LLM
        llm = AzureOpenAI(
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            openai_api_base=AZURE_OPENAI_API_BASE,
            openai_api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0.7,
        )

        # Build conversation history as a string
        conversation_history = "\n".join(
            f"{msg['type']}: {msg['content']}" for msg in self.memory.chat_memory.messages
        )

        # Create a system message template
        system_message = (
            "You are an AI assistant. Based on the following conversation history "
            "and user's query, determine the user's real intent.\n"
            "Chat history: {conversation_history}\n"
            "User query: {query}\n"
        )

        # Set up the prompt
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(system_message)]
        )

        # Create the LLM chain
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(conversation_history=conversation_history, query=query)
        
        return result.strip()

    def _fetch_relevant_context(self, user_intent: str) -> list:
        """
        Retrieve relevant context documents from Azure Cognitive Search based on the user intent.
        """
        # Initialize the retriever
        retriever = AzureCognitiveSearchRetriever(
            api_key=AZURE_SEARCH_KEY,
            service_name
