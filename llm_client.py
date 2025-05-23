# llm_client.py
# Defines a generic interface for any LLM provider.
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """
    Abstract base class for LLM clients. All clients must implement generate().
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response to the given prompt and return text.
        """
        pass

class OpenAIClient(LLMClient):
    """
    Client for OpenAI API (e.g., GPT-3.5-turbo).
    """
    def __init__(self, api_key: str):
        import openai
        openai.api_key = api_key

    def generate(self, prompt: str) -> str:
        # Sends the prompt to OpenAI and returns the assistant's reply
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ],
        )
        return resp.choices[0].message.content



