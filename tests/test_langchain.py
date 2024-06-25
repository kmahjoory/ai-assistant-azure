import unittest
from unittest.mock import patch, MagicMock
from src.chatbot import Chatbot

class TestChatbot(unittest.TestCase):

    @patch('chatbot.AzureOpenAI')
    @patch('chatbot.AzureCognitiveSearchRetriever')
    def test_infer_user_intent(self, mock_azure_openai, mock_retriever):
        # Mock the LLM response for user intent inference
        mock_llm_instance = mock_azure_openai.return_value
        mock_llm_instance.run.return_value = "buy a backpack"

        chatbot = Chatbot()
        query = "I need a large backpack, any recommendations?"

        # Call the method we are testing
        result = chatbot._infer_user_intent(query)

        # Verify that the LLM was called with expected parameters
        mock_llm_instance.run.assert_called_once()

        # Assert that the result is correctly inferred
        self.assertEqual(result, "buy a backpack")

    @patch('chatbot.AzureCognitiveSearchRetriever')
    def test_fetch_relevant_context(self, mock_retriever):
        # Mock relevant documents retrieval
        mock_retriever_instance = mock_retriever.return_value
        mock_retriever_instance.get_relevant_documents.return_value = [
            MagicMock(page_content="Product details for backpack"),
            MagicMock(page_content="Another product description")
        ]

        chatbot = Chatbot()
        user_intent = "buy a backpack"

        # Call the method we are testing
        result = chatbot._fetch_relevant_context(user_intent)

        # Verify that the retriever was called with expected parameters
        mock_retriever_instance.get_relevant_documents.assert_called_once_with(user_intent)

        # Assert that the returned context is correct
        self.assertEqual(result, ["Product details for backpack", "Another product description"])

if __name__ == '__main__':
    unittest.main()
