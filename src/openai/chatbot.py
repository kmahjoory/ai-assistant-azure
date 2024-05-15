
import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from dotenv import load_dotenv

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-1"

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Chat roles
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


class Chatbot:
    """Chat with an LLM using RAG. Keeps chat history in memory."""

    chat_history = []

    def __init__(self):
        load_dotenv()
        openai.api_type = AZURE_OPENAI_API_TYPE
        openai.api_base = AZURE_OPENAI_API_BASE
        openai.api_version = AZURE_OPENAI_API_VERSION
        openai.api_key = AZURE_OPENAI_API_KEY

    def _summarize_user_intent(self, query: str) -> str:
        """
        Creates a user message containing the user intent, by summarizing the chat
        history and user query.
        """
        chat_history_str = ""
        for entry in self.chat_history:
            chat_history_str += f"{entry['role']}: {entry['content']}\n"
        messages = [
            {
                "role": SYSTEM,
                "content": (
                    "You're an AI assistant reading the transcript of a conversation "
                    "between a user and an assistant. Given the chat history and "
                    "user's query, infer user real intent."
                    f"Chat history: ```{chat_history_str}```\n"
                    f"User's query: ```{query}```\n"
                ),
            }
        ]
        chat_intent_completion = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            n=1,
        )
        user_intent = chat_intent_completion.choices[0].message.content

        return user_intent

    def _get_context(self, user_intent: str) -> list[str]:
        """
        Gets the relevant documents from Azure Cognitive Search.
        """
        query_vector = Vector(
            value=openai.Embedding.create(
                engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=user_intent
            )["data"][0]["embedding"],
            fields="embedding",
        )

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY),
        )

        docs = search_client.search(search_text="", vectors=[query_vector], top=1)
        context_list = [doc["content"] for doc in docs]

        return context_list

    def _rag(self, context_list: list[str], query: str) -> str:
        """
        Asks the LLM to answer the user's query with the context provided.
        """
        user_message = {"role": USER, "content": query}
        self.chat_history.append(user_message)

        context = "\n\n".join(context_list)
        messages = [
            {
                "role": SYSTEM,
                "content": (
                    "You're a helpful assistant.\n"
                    "Please answer the user's question using only information you can "
                    "find in the context.\n"
                    "If the user's question is unrelated to the information in the "
                    "context, say you don't know.\n"
                    f"Context: ```{context}```\n"
                ),
            }
        ]
        messages = messages + self.chat_history

        chat_completion = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            n=1,
        )

        response = chat_completion.choices[0].message.content
        assistant_message = {"role": ASSISTANT, "content": response}
        self.chat_history.append(assistant_message)

        return response

    def ask(self, query: str) -> str:
        """
        Queries an LLM using RAG.
        """
        user_intent = self._summarize_user_intent(query)
        context_list = self._get_context(user_intent)
        response = self._rag(context_list, query)
        print(
            "*****\n"
            f"QUESTION:\n{query}\n"
            f"USER INTENT:\n{user_intent}\n"
            f"RESPONSE:\n{response}\n"
            "*****\n"
        )

        return response
import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from dotenv import load_dotenv

# Load environment variables
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
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Chat roles constants
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


class Chatbot:
    """Chatbot that interacts with an LLM using RAG (Retrieve and Generate) and maintains chat history."""

    def __init__(self):
        # Initialize OpenAI and chat history
        self.chat_history = []
        openai.api_type = AZURE_OPENAI_API_TYPE
        openai.api_base = AZURE_OPENAI_API_BASE
        openai.api_version = AZURE_OPENAI_API_VERSION
        openai.api_key = AZURE_OPENAI_API_KEY

    def _summarize_user_intent(self, query: str) -> str:
        """
        Summarize the user's intent based on chat history and current query.
        """
        # Construct chat history as a string
        chat_history_str = "\n".join(f"{entry['role']}: {entry['content']}" for entry in self.chat_history)
        
        # Create a system message to infer user intent
        messages = [
            {
                "role": SYSTEM,
                "content": (
                    "You're an AI assistant analyzing the conversation transcript. Based on the chat history "
                    "and user's query, infer the real user intent.\n"
                    f"Chat history: ```{chat_history_str}```\n"
                    f"User's query: ```{query}```\n"
                ),
            }
        ]
        
        # Request OpenAI to infer user intent
        response = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            n=1,
        )
        
        # Extract inferred intent from response
        user_intent = response.choices[0].message.content.strip()
        
        return user_intent

    def _get_context(self, user_intent: str) -> list[str]:
        """
        Retrieve relevant documents from Azure Cognitive Search based on user intent.
        """
        # Create a query vector for searching the index
        embedding = openai.Embedding.create(
            engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=user_intent
        )["data"][0]["embedding"]
        
        query_vector = Vector(
            value=embedding,
            fields="embedding"
        )
        
        # Initialize the Azure Search client
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY),
        )
        
        # Search for relevant documents
        search_results = search_client.search(search_text="", vectors=[query_vector], top=1)
        context_list = [doc["content"] for doc in search_results]
        
        return context_list

    def _generate_response(self, context_list: list[str], query: str) -> str:
        """
        Generate a response from the LLM based on the provided context and user query.
        """
        # Add user's query to chat history
        self.chat_history.append({"role": USER, "content": query})
        
        # Join the context list for use in the system message
        context = "\n\n".join(context_list)
        
        # Construct messages for OpenAI LLM to generate the response
        messages = [
            {
                "role": SYSTEM,
                "content": (
                    "You're a helpful assistant. Please answer the user's question using the context provided. "
                    "If the user's query is unrelated to the context, respond with 'I don't know'.\n"
                    f"Context: ```{context}```\n"
                ),
            }
        ] + self.chat_history
        
        # Request response from OpenAI LLM
        response = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            n=1,
        )
        
        # Extract the generated response
        assistant_response = response.choices[0].message.content.strip()
        
        # Add the assistant's response to chat history
        self.chat_history.append({"role": ASSISTANT, "content": assistant_response})
        
        return assistant_response

    def ask(self, query: str) -> str:
        """
        Process the user's query through RAG pipeline: infer user intent, retrieve context, and generate response.
        """
        # Step 1: Summarize the user intent
        user_intent = self._summarize_user_intent(query)
        
        # Step 2: Retrieve relevant context documents
        context_list = self._get_context(user_intent)
        
        # Step 3: Generate the final response
        response = self._generate_response(context_list, query)
        
        # Display conversation details for debugging
        print(f"*****\nQUESTION: {query}\nUSER INTENT: {user_intent}\nRESPONSE: {response}\n*****\n")
        
        return response
