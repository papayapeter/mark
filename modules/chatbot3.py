import openai
import os


class Chatbot3:
    """
    GPT-3 Implementation of a question answeringchatbot
    """

    def __init__(
        self,
        api_key,
        dataset,
        question_signifier="Q: ",
        answer_signifier="A: ",
        separator="\n\n",
    ):
        """
        api_key: key to acces the openai api
        dataset: string that, together with a question, serves as the prompt
        question_signifier: how a question starts in the 'path_to_dataset_txt' file
        answer_signifier: how an answer starts in the 'path_to_dataset_txt' file
        separator: what separates a set of question & answer
        """

        # load open ai api key
        openai.api_key = api_key

        # set instance variables
        self._dataset = dataset
        self._question_signifier = question_signifier
        self._answer_signifier = answer_signifier
        self._seperator = separator

        # correct the answer signifier (remove blank spaces)
        while len(self._answer_signifier) > 0 and self._answer_signifier[-1] == " ":
            self._answer_signifier = self._answer_signifier[:-1]

    def ask(self, question, max_tokens=100, stop=None):
        """
        generates an answer from a question

        question: the question as a string
        max_tokens: maximum length of the response in tokens
        stop: sequence where the api will stop generating further tokens
        """
        # generate prompt from dataset & question
        prompt = (
            self._dataset
            + self._seperator
            + self._question_signifier
            + question
            + "\n"
            + self._answer_signifier
        )

        # return the response from the openai api
        return (
            openai.Completion.create(
                engine="davinci", prompt=prompt, max_tokens=max_tokens, stop=stop
            )
            .choices[0]
            .text.strip()
        )
