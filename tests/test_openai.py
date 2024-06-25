import unittest
from unittest.mock import patch, MagicMock
from chatbot import Chatbot

class TestChatbot(unittest.TestCase):

    @patch('chatbot.openai.ChatCompletion.create')
    def test_summarize_user_intent(self, mock_chat_completion_create):
        # Mock the response from OpenAI's ChatCompletion API
        mock_chat_completion_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="buy a backpack"))]
        )

        chatbot = Chatbot()
        query = "I need a large backpack, any recommendations?"
        
        # Test the summarization of user intent
        result = chatbot._summarize_user_intent(query)

        # Check that OpenAI's API was called correctly
        mock_chat_completion_create.assert_called_once()

        # Assert that the correct user intent was inferred
        self.assertEqual(result, "buy a backpack")

    @patch('chatbot.openai.Embedding.create')
    @patch('chatbot.SearchClient')
    def test_get_context(self, mock_search_client, mock_embedding_create):
        # Mock the response from OpenAI's Embedding API
        mock_embedding_create.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        # Mock the search results from Azure Cognitive Search
        mock_search_client_instance = mock_search_client.return_value
        mock_search_client_instance.search.return_value = [
            {"content": "Product details for a large backpack"}
        ]

        chatbot = Chatbot()
        user_intent = "buy a backpack"

        # Test context retrieval based on user intent
        result = chatbot._get_context(user_intent)

        # Verify that the embedding was created correctly
        mock_embedding_create.assert_called_once_with(engine=chatbot.AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=user_intent)

        # Verify that the search client was called to search for relevant documents
        mock_search_client_instance.search.assert_called_once()

        # Assert the context list returned is correct
        self.assertEqual(result, ["Product details for a large backpack"])

    @patch('chatbot.openai.ChatCompletion.create')
    def test_generate_response(self, mock_chat_completion_create):
        # Mock the response from OpenAI's ChatCompletion API
        mock_chat_completion_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="The backpack costs $50"))]
        )

        chatbot = Chatbot()
        context_list = ["Product details for a large backpack"]
        query = "How much does the backpack cost?"

        # Test the response generation based on context
        result = chatbot._generate_response(context_list, query)

        # Check that OpenAI's API was called correctly
        mock_chat_completion_create.assert_called_once()

        # Assert that the generated response is correct
        self.assertEqual(result, "The backpack costs $50")

    @patch('chatbot.Chatbot._summarize_user_intent')
    @patch('chatbot.Chatbot._get_context')
    @patch('chatbot.Chatbot._generate_response')
    def test_ask(self, mock_generate_response, mock_get_context, mock_summarize_user_intent):
        # Mock the internal methods of the chatbot
        mock_summarize_user_intent.return_value = "buy a backpack"
        mock_get_context.return_value = ["Product details for a large backpack"]
        mock_generate_response.return_value = "The backpack costs $50"

        chatbot = Chatbot()
        query = "I need a large backpack, how much does it cost?"

        # Test the ask method, which orchestrates the entire process
        result = chatbot.ask(query)

        # Assert that all the steps were executed
        mock_summarize_user_intent.assert_called_once_with(query)
        mock_get_context.assert_called_once_with("buy a backpack")
        mock_generate_response.assert_called_once_with(["Product details for a large backpack"], query)

        # Assert the final response is as expected
        self.assertEqual(result, "The backpack costs $50")


if __name__ == '__main__':
    unittest.main()
