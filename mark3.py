"""
Chatbot Mark based on GPT-3
"""
import os
from dotenv import load_dotenv

from modules.chatbot3 import Chatbot3

# load all environment variables from .env files
load_dotenv()

# load dataset
with open("fisher_dataset.txt") as file:
    dataset = file.read()

# create chatbot instance
mark = Chatbot3(os.environ.get("OPENAI_API"), dataset)

while True:
    question = input()

    if question.lower() == "quit":
        break

    print(mark.ask(question, stop="\n\n"))
