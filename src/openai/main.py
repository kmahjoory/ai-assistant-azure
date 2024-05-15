"""
Entry point for the chatbot.
"""
from openai.chatbot import Chatbot


def main():
    chatbot = Chatbot()
    chatbot.ask("I need Bright LED lights. What is your recommendation?")



if __name__ == "__main__":
    main()
