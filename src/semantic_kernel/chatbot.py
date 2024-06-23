import os
import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)
from semantic_kernel.semantic_functions.chat_prompt_template import ChatPromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.semantic_functions.semantic_function_config import (
    SemanticFunctionConfig,
)

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE", "azure")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Azure Cognitive Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "products-index-3")

# Constants for roles and plugin name
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"
PLUGIN_NAME = "rag_plugin"


class Chatbot:
    """A Chatbot that interacts with an LLM using RAG (Retrieve and Generate) and maintains chat history."""

    def __init__(self):
        # Initialize the kernel and connect to Azure OpenAI services
        self.kernel = sk.Kernel()
        self.kernel.add_chat_service(
            service_name="azureopenai",
            service_connector=AzureChatCompletion(
                deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_API_BASE,
                api_key=AZURE_OPENAI_API_KEY,
            ),
        )

        # Initialize variables to store chat history and query context
        self.variables = sk.ContextVariables()
        self.variables["chat_history"] = ""

    async def _summarize_user_intent(self, query: str) -> str:
        """
        Summarizes the user's intent based on chat history and current query.
        """
        # Store the user's query in variables
        self.variables["query"] = query

        # Define the prompt template for summarizing user intent
        user_template = (
            "You're an AI assistant reading the transcript of a conversation "
            "between a user and an assistant. Based on the chat history and "
            "the user's query, infer the real user intent."
            "Chat history: ```{{$chat_history}}```\n"
            "User's query: ```{{$query}}```\n"
        )

        # Define the prompt configuration for the semantic function
        prompt_config_dict = {
            "type": "completion",
            "description": "An AI assistant that infers user intent.",
            "completion": {
                "temperature": 0.7,
                "top_p": 0.5,
                "max_tokens": 200,
                "number_of_responses": 1,
            },
            "input": {
                "parameters": [
                    {"name": "query", "description": "User's question."},
                    {"name": "chat_history", "description": "Conversation history."},
                ]
            },
        }

        # Create and configure the semantic function
        prompt_config = PromptTemplateConfig.from_dict(prompt_config_dict)
        prompt_template = ChatPromptTemplate(
            template=user_template,
            prompt_config=prompt_config,
            template_engine=self.kernel.prompt_template_engine,
        )
        user_intent_function_config = SemanticFunctionConfig(
            prompt_config, prompt_template
        )
        user_intent_function = self.kernel.register_semantic_function(
            skill_name=PLUGIN_NAME,
            function_name="user_intent_function",
            function_config=user_intent_function_config,
        )

        # Run the semantic function to infer user intent
        response = await self.kernel.run_async(
            user_intent_function, input_vars=self.variables
        )

        return str(response)

    async def _get_context(self, query: str) -> list[str]:
        """
        Retrieves relevant documents from Azure Cognitive Search based on the user's intent.
        """
        # Set up the kernel to use text embeddings and Azure Cognitive Search memory store
        kernel = sk.Kernel()
        kernel.add_text_embedding_generation_service(
            service_name="openai-embedding",
            service_connector=OpenAITextEmbedding(
                model_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                api_key=AZURE_OPENAI_API_KEY,
                endpoint=AZURE_OPENAI_API_BASE,
                api_type=AZURE_OPENAI_API_TYPE,
                api_version=AZURE_OPENAI_API_VERSION,
            ),
        )
        kernel.register_memory_store(
            memory_store=AzureCognitiveSearchMemoryStore(
                vector_size=1536,
                search_endpoint=AZURE_SEARCH_ENDPOINT,
                admin_key=AZURE_SEARCH_KEY,
            )
        )

        # Retrieve documents from Azure Cognitive Search
        docs = await kernel.memory.search_async(AZURE_SEARCH_INDEX_NAME, query, limit=1)
        return [doc.text for doc in docs]

    async def _generate_response(self, context_list: list[str], query: str) -> str:
        """
        Generates a response to the user's query using the provided context.
        """
        # Store the context and user's query
        context = "\n\n".join(context_list)
        self.variables["context"] = context
        self.variables["query"] = query

        # Define the system and user templates
        system_template = (
            "You're a helpful assistant.\n"
            "Answer the user's question using the information in the context.\n"
            "If the query is unrelated to the context, say 'I don't know'.\n"
            "Context: ```{{$context}}```\n"
        )
        user_template = "{{$chat_history}}" + f"{USER}: " + "{{$query}}\n"

        # Define the prompt configuration for generating a response
        prompt_config_dict = {
            "type": "completion",
            "description": "A helpful assistant for responding to user queries.",
            "completion": {
                "temperature": 0.7,
                "top_p": 0.5,
                "max_tokens": 200,
                "number_of_responses": 1,
                "chat_system_prompt": system_template,
            },
            "input": {
                "parameters": [
                    {"name": "query", "description": "User's question."},
                    {"name": "context", "description": "Context for the response."},
                    {"name": "chat_history", "description": "Conversation history."},
                ]
            },
        }

        # Create and configure the semantic function for RAG
        prompt_config = PromptTemplateConfig.from_dict(prompt_config_dict)
        prompt_template = ChatPromptTemplate(
            template=user_template,
            prompt_config=prompt_config,
            template_engine=self.kernel.prompt_template_engine,
        )
        rag_function_config = SemanticFunctionConfig(prompt_config, prompt_template)
        rag_function = self.kernel.register_semantic_function(
            skill_name=PLUGIN_NAME,
            function_name="rag_function",
            function_config=rag_function_config,
        )

        # Run the RAG function to generate the response
        response = await self.kernel.run_async(rag_function, input_vars=self.variables)

        # Update chat history
        self.variables["chat_history"] += f"{USER}: {query}\n{ASSISTANT}: {response}\n"

        return str(response)

    async def ask(self, query: str) -> str:
        """
        Processes the user's query by summarizing user intent, retrieving context, and generating a response.
        """
        user_intent = await self._summarize_user_intent(query)
        context_list = await self._get_context(user_intent)
        response = await self._generate_response(context_list, query)
        
        # Print conversation details for debugging
        print(f"*****\nQUESTION: {query}\nUSER INTENT: {user_intent}\nRESPONSE: {response}\n*****\n")
        
        return response
