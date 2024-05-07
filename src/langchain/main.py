"""
Entry point for the chatbot application.
"""
from chatbot import Chatbot

def main():
    # Initialize the chatbot instance
    chatbot = Chatbot()
    
    # Example conversation with the chatbot
    response1 = chatbot.ask("I need Bright LED lights. What is your recommendation?")
    print(f"Chatbot Response: {response1}")
    

if __name__ == "__main__":
    main()
