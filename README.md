## Incorporating Custom Data into LLMs

This guide demonstrates how to build a chatbot that enhances ChatGPT with personalized data, using the Retrieval-Augmented Generation (RAG) pattern along with vector search. It presents three methods for solving this problem: working directly with OpenAI's APIs, utilizing LangChain, and applying Semantic Kernel.

| Update: For those wanting to use a more up-to-date version of OpenAI's API, refer to the modifications in [this pull request](https://github.com/bstollnitz/rag/pull/2).<br/> A big thanks to [@xcvil](https://github.com/xcvil) for making these improvements! |
|------|

## Requirements
- You must have an Azure subscription. If you don’t have one, you can obtain a [free Azure subscription](https://azure.microsoft.com/en-us/free).
- Set up a "Cognitive Search" service on Azure.
- Set up an "OpenAI" service on Azure, and within that service, deploy two models:
    - One deployment for the "text-embedding-ada-002" model.
    - One deployment for the "gpt-35-turbo" model.
- Add a ".env" file to the project with the necessary environment variables (you can use ".env-example" as a template):
    - **AZURE_OPENAI_API_BASE** - Access https://oai.azure.com/, go to "Chat Playground", "View code", and copy the API base from the code.
    - **AZURE_OPENAI_API_KEY** - In the same interface, grab the API key from the bottom section.
    - **AZURE_OPENAI_EMBEDDING_DEPLOYMENT** - Navigate to "Deployments" and get the deployment name for the "text-embedding-ada-002" model.
    - **AZURE_OPENAI_CHATGPT_DEPLOYMENT** - Similarly, locate the deployment name for the "gpt-35-turbo" model.
    - **AZURE_SEARCH_ENDPOINT** - Open https://portal.azure.com/, locate your "Cognitive Search" resource, and find the "Url".
    - **AZURE_SEARCH_KEY** - On the same resource page, navigate to "Settings", then "Keys", and copy the "Primary admin key".
    - **AZURE_SEARCH_SERVICE_NAME** - This is the name of your "Cognitive Search" resource in the Azure portal.

## Package Installation

Install the necessary packages from the `environment.yml` file:


## Running the Project

You can execute the same use case in one of three ways:
- Using OpenAI APIs directly:
    - Run `src/1_openai/init_search_1.py` (press F5 to initialize the Azure Cognitive Search index).
    - Run `src/1_openai/main_1.py` to perform queries using your data.
- Using LangChain:
    - Run `src/2_langchain/init_search_2.py`.
    - Run `src/2_langchain/main_2.py`.
- Using Semantic Kernel:
    - Run `src/3_semantic_kernel/init_search_3.py`.
    - Run `src/3_semantic_kernel/main_3.py`.
