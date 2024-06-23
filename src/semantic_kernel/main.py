"""
Entry point for the chatbot.
"""
import asyncio

from semantic_kernel.chatbot import Chatbot


async def main():
    chatbot = Chatbot()
    chatbot.ask("I need Bright LED lights. What is your recommendation?")


if __name__ == "__main__":
    asyncio.run(main())
