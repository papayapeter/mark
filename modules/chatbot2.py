from transformers import GPT2LMHeadModel, GPT2Tokenizer
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

        # device to be used (cuda gpu or cpu)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model & tokenizer objects
        self._model = GPT2LMHeadModel.from_pretrained(model)
        if tokenizer is None:
            self._tokenizer = GPT2Tokenizer.from_pretrained(model)
        else:
            self._tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)

        self._model.to(self._device)

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

        # parameters for generation
        parameters = {
            "length": max_tokens,
            "temperature": 0.7,  # default: 1.0
            "top_k": 0,  # default: 0
            "top_p": 0.9,  # default: 0.9
            "repetition_penalty": 1.0,  # default: 1.0 (primarily useful for CTRL model)
            "num_return_sequences": 1,  # default: 1
            "stop_token": stop,  # character or None
            "seed": 0,
        }

        # generate prompt from dataset & question
        prompt = (
            self._dataset
            + self._seperator
            + self._question_signifier
            + question
            + "\n"
            + self._answer_signifier
        )

        # encode prompt to tokens
        encoded_prompt = self._tokenizer.encode(prompt, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self._device)

        # if the encoded prompt is empty
        # set the model input to None
        if encoded_prompt.size()[-1] == 0:
            input_tokens = None
        else:
            input_tokens = encoded_prompt

        # generate output sequence with parameters
        output_sequences = self._model.generate(
            input_ids=input_tokens,
            max_length=parameters["length"] + len(encoded_prompt[0]),
            temperature=parameters["temperature"],
            top_k=parameters["top_k"],
            top_p=parameters["top_p"],
            repetition_penalty=parameters["repetition_penalty"],
            do_sample=True,
            num_return_sequences=parameters["num_return_sequences"],
        )

        # decode sequence(s) from tokens to text
        generated_sequences = []

        for generated_sequence_id, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            # decode text
            text = self._tokenizer.decode(
                generated_sequence, clean_up_tokenization_spaces=True
            )

            # remove all text after the stop token
            text = text[
                : text.find(parameters["stop_token"])
                if not parameters["stop_token"] is None
                else None
            ]

            # add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt
                + text[
                    len(
                        self._tokenizer.decode(
                            encoded_prompt[0], clean_up_tokenization_spaces=True
                        )
                    ) :
                ]
            )

            generated_sequences.append(total_sequence)

        return generated_sequences[0].replace(prompt + " ", "").split("\n\n")[0]
