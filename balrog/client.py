import base64
import logging
import time
from collections import namedtuple
from io import BytesIO

from google import genai
from google.genai import types

from anthropic import Anthropic
from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
        "cache_creation_tokens",
        "cache_read_tokens",
        "extended_thinking",
    ],
    defaults=[0, 0, None],  # Default cache tokens to 0, extended_thinking to None
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """Base class for LLM client wrappers.

    Provides common functionality for interacting with different LLM APIs, including
    handling retries and common configuration settings. Subclasses should implement
    the `generate` method specific to their LLM API.
    """

    def __init__(self, client_config):
        """Initialize the LLM client wrapper with configuration settings.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.alternate_roles = client_config.alternate_roles

    def generate(self, messages):
        """Generate a response from the LLM given a list of messages.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)
        raise Exception(f"Failed to execute {func.__name__} after {self.max_retries} retries.")


def process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for OpenAI.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    """Process an image for Anthropic's Claude API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for Claude.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for Anthropic
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
    }


class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "nvidia" or self.client_name.lower() == "xai":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # For OpenAI, always use the standard API regardless of base_url
                self.client = OpenAI()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(process_image_openai(msg.attachment))
            if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == msg.role:
                converted_messages[-1]["content"].extend(new_content)
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with Google's Generative AI API."""

    def __init__(self, client_config):
        """Initialize the GoogleGenerativeAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Generative AI client if not already initialized."""
        if not self._initialized:
            self.client = genai.Client()
            self.model = None

            # Create kwargs dictionary for GenerationConfig
            client_kwargs = {
                "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                client_kwargs["temperature"] = temperature
                
            thinking_budget = self.client_kwargs.get("thinking_budget", -1)

            self.generation_config = genai.types.GenerateContentConfig(
                **client_kwargs, 
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            )
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the new Google GenAI SDK.

        Args:
            messages (list): A list of message objects.

        Returns:
            list[types.Content]: A list of Content objects formatted for the API.
        """
        converted_messages = []
        
        for msg in messages:
            parts = []
            
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
                
            if msg.content:
                parts.append(types.Part(text=msg.content))

            if msg.attachment is not None:
                parts.append(types.Part(image=msg.attachment))

            converted_messages.append(
                types.Content(role=role, parts=parts)
            )
        return converted_messages

    def extract_completion(self, response):
        """Extract the completion text from the API response.

        Args:
            response: The response object from the API.

        Returns:
            str: The extracted completion text.
            
        Raises:
            Exception: If response is None or missing expected fields.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")

        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")
            
        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        text = getattr(content_parts[0], "text", None)
        if text is None:
            raise Exception("No text found in the content parts.")
            
        return text.strip()

    def generate(self, messages):
        """Generate a response from the Generative AI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Generative AI API.
        """
        self._initialize_client()

        converted_messages = self.convert_messages(messages)

        def api_call():
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=converted_messages,
                config=self.generation_config,
            )
            # Attempt to extract completion immediately after API call
            completion = self.extract_completion(response)
            # Return both response and completion if successful
            return response, completion

        try:
            # Execute the API call and extraction together with retries
            response, completion = self.execute_with_retries(api_call)

            # Check if the successful response contains an empty completion
            if not completion or completion.strip() == "":
                logger.warning(f"Gemini returned an empty completion for model {self.model_id}. Returning default empty response.")
                return LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    stop_reason="empty_response",
                    input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    reasoning=None,
                )
            else:
                # If completion is not empty, return the normal response
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", "unknown")
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=(
                        getattr(response.usage_metadata, "prompt_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    output_tokens=(
                        getattr(response.usage_metadata, "candidates_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    reasoning=None,
                )
        except Exception as e:
            logger.error(f"API call failed after {self.max_retries} retries: {e}. Returning empty completion.")
            # Return a default response indicating failure
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                stop_reason="error_max_retries",
                input_tokens=0, # Assuming 0 tokens consumed if call failed
                output_tokens=0,
                reasoning=None,
            )


class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API with prompt caching."""

    def __init__(self, client_config):
        """Initialize the ClaudeWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Claude client if not already initialized."""
        if not self._initialized:
            self.client = Anthropic()
            self._initialized = True

    def _get_extra_headers(self) -> dict[str, str]:
        """Get extra headers for prompt caching."""
        return {"anthropic-beta": "prompt-caching-2024-07-31"}

    # Minimum tokens required for caching (conservative: use Haiku's 2048)
    MIN_CACHE_TOKENS = 2048

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(text) // 4

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Claude API with caching.

        Extracts system messages for the system parameter and adds cache_control
        markers to optimize prompt caching. Only adds cache_control when content
        exceeds the minimum token threshold (2048 for Haiku, 1024 for Sonnet).

        Args:
            messages (list): A list of message objects.

        Returns:
            tuple: (system_content, converted_messages) where system_content is
                   the system prompt with cache_control, and converted_messages
                   is the list of user/assistant messages.
        """
        system_content = None
        system_text = ""
        converted_messages = []

        for msg in messages:
            content_block = {"type": "text", "text": msg.content}

            if msg.role == "system":
                system_text = msg.content
                system_content = [content_block]  # Don't add cache_control yet
                continue

            converted_messages.append({"role": msg.role, "content": [content_block]})

            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_claude(msg.attachment))

        # Calculate cumulative token counts to find optimal cache breakpoint
        # Strategy: cache at second-to-last user message if cumulative >= MIN_CACHE_TOKENS
        cumulative_tokens = self._estimate_tokens(system_text)

        # Build cumulative token count per message
        msg_cumulative = []
        for msg in converted_messages:
            msg_tokens = sum(self._estimate_tokens(c.get("text", "")) for c in msg["content"] if c.get("type") == "text")
            cumulative_tokens += msg_tokens
            msg_cumulative.append(cumulative_tokens)

        # Find user message indices
        user_indices = [i for i, m in enumerate(converted_messages) if m["role"] == "user"]

        # Place cache_control on second-to-last user message if cumulative there >= MIN_CACHE_TOKENS
        breakpoint_idx = None
        if len(user_indices) >= 2:
            second_to_last = user_indices[-2]
            if msg_cumulative[second_to_last] >= self.MIN_CACHE_TOKENS:
                breakpoint_idx = second_to_last

        if breakpoint_idx is not None:
            last_content = converted_messages[breakpoint_idx]["content"][-1]
            last_content["cache_control"] = {"type": "ephemeral"}
            logger.info(f"Cache breakpoint at msg {breakpoint_idx}, ~{msg_cumulative[breakpoint_idx]} tokens (min: {self.MIN_CACHE_TOKENS})")
        elif system_content and self._estimate_tokens(system_text) >= self.MIN_CACHE_TOKENS:
            # System prompt alone is long enough
            system_content[0]["cache_control"] = {"type": "ephemeral"}
            logger.info(f"Cache breakpoint at system prompt, ~{self._estimate_tokens(system_text)} tokens")
        else:
            # Log why caching isn't active yet
            total_tokens = msg_cumulative[-1] if msg_cumulative else self._estimate_tokens(system_text)
            logger.info(
                f"Cache not active: {len(user_indices)} user msgs, ~{total_tokens} tokens "
                f"(need 2+ msgs and {self.MIN_CACHE_TOKENS}+ tokens at second-to-last)"
            )

        return system_content, converted_messages

    def generate(self, messages):
        """Generate a response from the Claude API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Claude API.
        """
        self._initialize_client()
        system_content, converted_messages = self.convert_messages(messages)

        # Check for extended thinking budget
        thinking_budget = self.client_kwargs.get("thinking_budget", 0)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Add system prompt with caching if present
            if system_content:
                api_kwargs["system"] = system_content

            # Enable extended thinking if budget specified
            if thinking_budget > 0:
                api_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                # Extended thinking requires higher max_tokens
                api_kwargs["max_tokens"] = max(api_kwargs["max_tokens"], thinking_budget + 1024)

            # Only include temperature if it's not None (not compatible with thinking)
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None and thinking_budget == 0:
                api_kwargs["temperature"] = temperature

            return self.client.messages.create(**api_kwargs, extra_headers=self._get_extra_headers())

        response = self.execute_with_retries(api_call)

        # Extract cache metrics from response
        usage = response.usage
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Extract thinking and text content from response blocks
        extended_thinking = None
        completion_text = ""
        for block in response.content:
            if block.type == "thinking":
                extended_thinking = block.thinking
            elif block.type == "text":
                completion_text = block.text.strip()

        # Debug: log cache info and thinking usage
        thinking_tokens = getattr(usage, "thinking_tokens", 0) or 0
        logger.info(
            f"Claude usage: in={usage.input_tokens}, out={usage.output_tokens}, "
            f"thinking={thinking_tokens}, cache_create={cache_creation}, cache_read={cache_read}"
        )

        return LLMResponse(
            model_id=self.model_id,
            completion=completion_text,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            extended_thinking=extended_thinking,
        )


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the client name.

    Args:
        client_config: Configuration object containing client-specific settings.

    Returns:
        callable: A factory function that returns an instance of the appropriate LLM client.
    """

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower or "xai" in client_name_lower:
            # NVIDIA and XAI use OpenAI-compatible API, so we use the OpenAI wrapper
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_name_lower:
            return ClaudeWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
