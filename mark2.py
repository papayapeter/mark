"""
Chatbot Mark based on GPT-2
"""
import os

from modules.chatbot2 import Chatbot2

# load dataset
with open("fisher_dataset.txt") as file:
    dataset = file.read()

# create chatbot instance
mark = Chatbot2(dataset, "gpt2")

while True:
    question = input()

    if question.lower() == "quit":
        break

    print(mark.ask(question))
