## Incorporating Custom Data into LLMs

This project implements a Question answering AI-assistant which reponds questions only based on a source of knowledge, using the Retrieval-Augmented Generation (RAG) pattern along with vector search. It implements the assistant using three packages: OpenAI's APIs, utilizing LangChain, and Semantic Kernel.



## Requirements
- You must have an Azure subscription. 
- Set up a "AI Search" service on Azure.
- Set up an "OpenAI" service on Azure, and within that service, deploy two models:
    - One deployment for the "text-embedding-ada-002" model.
    - One deployment for the "gpt-35-turbo" model.
- Add a ".env" file to the project with the necessary environment variables (you can use ".env-example" as a template):

    - **AZURE_OPENAI_API_KEY** - In the same interface, grab the API key from the bottom section.
    - **AZURE_OPENAI_EMBEDDING_DEPLOYMENT** - Navigate to "Deployments" and get the deployment name for the "text-embedding-ada-002" model.
    - **AZURE_OPENAI_CHATGPT_DEPLOYMENT** - Similarly, locate the deployment name for the "gpt-35-turbo" model.
    - **AZURE_SEARCH_ENDPOINT** - Open https://portal.azure.com/, locate your "Cognitive Search" resource, and find the "Url".
    - **AZURE_SEARCH_KEY** - On the same resource page, navigate to "Settings", then "Keys", and copy the "Primary admin key".
    - **AZURE_SEARCH_SERVICE_NAME** - This is the name of your "Cognitive Search" resource in the Azure portal.


## Running the Project

You can execute the same use case in one of three ways:
- Using OpenAI APIs directly:
    - Run `src/openai/init_search.py`.
    - Run `src/openai/main.py` to perform queries using your data.
- Using LangChain:
    - Run `src/langchain/init_search.py`.
    - Run `src/langchain/main.py`.
- Using Semantic Kernel:
    - Run `src/semantic_kernel/init_search.py`.
    - Run `src/semantic_kernel/main.py`.
