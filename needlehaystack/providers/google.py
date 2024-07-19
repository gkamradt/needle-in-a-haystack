import os
from operator import itemgetter
import pkg_resources
import requests
from typing import Optional

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import sentencepiece

from .model import ModelProvider


class Google(ModelProvider):
    """
    A wrapper class for interacting with Google's Gemini API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the Google model to use for evaluations and interactions.
        model: An instance of the Google Gemini client for API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """

    DEFAULT_MODEL_KWARGS: dict = dict(max_output_tokens=300,
                                      temperature=0)
    VOCAB_FILE_URL = "https://raw.githubusercontent.com/google/gemma_pytorch/33b652c465537c6158f9a472ea5700e5e770ad3f/tokenizer/tokenizer.model"

    def __init__(self,
                 model_name: str = "gemini-1.5-pro",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 vocab_file_url: str = VOCAB_FILE_URL):
        """
        Initializes the Google model provider with a specific model.

        Args:
            model_name (str): The name of the Google model to use. Defaults to 'gemini-1.5-pro'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
            vocab_file_url (str): Sentencepiece model file that defines tokenization vocabulary. Deafults to gemma
                tokenizer https://github.com/google/gemma_pytorch/blob/main/tokenizer/tokenizer.model

        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        local_vocab_file = 'tokenizer.model'
        if not os.path.exists(local_vocab_file):
            response = requests.get(vocab_file_url)  # Download Tokenizer Vocab File (4MB)
            response.raise_for_status()

            with open(local_vocab_file, 'wb') as f:
                for chunk in response.iter_content():
                    f.write(chunk)
        self.tokenizer = sentencepiece.SentencePieceProcessor(local_vocab_file)

        resource_path = pkg_resources.resource_filename('needlehaystack', 'providers/Anthropic_prompt.txt')

        # Generate the prompt structure for the model
        # Replace the following file with the appropriate prompt structure
        with open(resource_path, 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the Google model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        response = await self.model.generate_content_async(
            prompt,
            generation_config=self.model_kwargs
        )

        return response.text

    def generate_prompt(self, context: str, retrieval_question: str) -> str:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            str: The text prompt
        """
        return self.prompt_structure.format(
            retrieval_question=retrieval_question,
            context=context)

    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])

    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question,
        queries the Google model, and returns the model's response. This method leverages the LangChain
        library to build a sequence of operations: extracting input variables, generating a prompt,
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question.
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a
            dynamically provided question. The runnable encapsulates the entire process from prompt
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatGoogleGenerativeAI(temperature=0, model=self.model_name)
        chain = ({"context": lambda x: context,
                  "question": itemgetter("question")}
                 | prompt
                 | model
                 )
        return chain


