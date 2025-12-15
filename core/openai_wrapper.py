"""
OpenAI API wrapper with retry logic and circuit breaker pattern.
"""

import logging
from typing import Any, Dict, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors."""
    pass


class OpenAIWrapper:
    """
    Wrapper for OpenAI API calls with automatic retry and error handling.
    """

    def __init__(self, api_key: str, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        max_completion_tokens: Optional[int] = None,
        timeout: float = 30.0
    ) -> str:
        """
        Create a chat completion with automatic retry.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: OpenAI model to use
            temperature: Sampling temperature
            max_tokens: (Deprecated) Maximum tokens in response
            max_completion_tokens: Maximum completion tokens in response
            timeout: Request timeout in seconds

        Returns:
            Response content as string

        Raises:
            OpenAIServiceError: If the request fails after all retries
        """
        try:
            completion_limit = (
                max_completion_tokens
                if max_completion_tokens is not None
                else max_tokens
            )

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
            }

            if completion_limit is not None:
                request_kwargs["max_completion_tokens"] = completion_limit

            response = self.client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise OpenAIServiceError(f"Rate limit exceeded: {e}")

        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}")
            raise OpenAIServiceError(f"API timeout: {e}")

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIServiceError(f"API error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise OpenAIServiceError(f"Unexpected error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Create text embedding with automatic retry.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Dictionary containing:
                - 'embedding': Embedding vector as list of floats
                - 'model': Model name used
                - 'usage': Token usage information

        Raises:
            OpenAIServiceError: If the request fails after all retries
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )

            return {
                'embedding': response.data[0].embedding,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise OpenAIServiceError(f"Rate limit exceeded: {e}")

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIServiceError(f"API error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {e}")
            raise OpenAIServiceError(f"Unexpected error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def batch_create_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Create embeddings for multiple texts in a single API call.

        OpenAI API supports up to 2048 input texts per request for efficiency.

        Args:
            texts: List of texts to embed (max 2048)
            model: Embedding model to use

        Returns:
            Dictionary containing:
                - 'embeddings': List of embedding vectors (each is list of floats)
                - 'model': Model name used
                - 'usage': Token usage information

        Raises:
            OpenAIServiceError: If the request fails after all retries
            ValueError: If texts list is empty or exceeds 2048 items
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        if len(texts) > 2048:
            raise ValueError(f"Cannot process more than 2048 texts at once (got {len(texts)})")

        try:
            logger.info(f"Batch embedding request: {len(texts)} texts with model {model}")

            response = self.client.embeddings.create(
                input=texts,
                model=model
            )

            # Extract embeddings in order
            embeddings = [item.embedding for item in response.data]

            logger.info(f"Batch embedding completed: {len(embeddings)} embeddings generated")

            return {
                'embeddings': embeddings,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise OpenAIServiceError(f"Rate limit exceeded: {e}")

        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIServiceError(f"API error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI batch embeddings: {e}")
            raise OpenAIServiceError(f"Unexpected error: {e}")
