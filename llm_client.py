# llm_client.py
# Defines a generic interface for any LLM provider.
import logging
from abc import ABC, abstractmethod
from openai import OpenAI

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def generate_with_history(self, messages: list[dict]) -> str:
        """
        Generate a response given a full conversation history.
        Each message is a dict with 'role' and 'content' keys.
        """
        pass


class OpenAIClient(LLMClient):
    """
    Client for OpenAI API with configurable model and system prompt.
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", system_prompt: str = ""):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._call(messages)

    def generate_with_history(self, messages: list[dict]) -> str:
        full_messages = []
        if self._system_prompt:
            full_messages.append({"role": "system", "content": self._system_prompt})
        full_messages.extend(messages)
        return self._call(full_messages)

    def _call(self, messages: list[dict]) -> str:
        logger.info("Calling OpenAI model=%s with %d messages", self._model, len(messages))
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return resp.choices[0].message.content
