from transformers import pipeline
import torch


class Chatbot2:
    """
    GPT-2 Implementation of a question answeringchatbot
    """

    def __init__(
        self,
        dataset,
        model,
        tokenizer=None,
        question_signifier="Q: ",
        answer_signifier="A: ",
        separator="\n\n",
    ):
        """
        dataset: string that, together with a question, serves as the prompt
        model: what gpt-2 model to use
        tokenizer: what gpt-2 tokenizer to use (only needs to be set if different from the model)
        question_signifier: how a question starts in the 'path_to_dataset_txt' file
        answer_signifier: how an answer starts in the 'path_to_dataset_txt' file
        separator: what separates a set of question & answer
        """
        # set instance variables
        self._dataset = dataset
        self._question_signifier = question_signifier
        self._answer_signifier = answer_signifier
        self._seperator = separator

        # model & tokenizer objects
        self._model = model
        if tokenizer is None:
            self._tokenizer = model
        else:
            self._tokenizer = tokenizer

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

        bot = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            config={"max_length": max_tokens},
        )

        answer = (
            bot(prompt)[0]["generated_text"]
            # bot(prompt)[0]["generated_text"].replace(f"{prompt} ", "").split("Q: ")[0]
        )

        print(answer)

        return answer
