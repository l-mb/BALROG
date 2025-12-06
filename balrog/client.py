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

    # Minimum tokens for caching by model family (per Anthropic docs)
    # Opus 4.5 and Haiku 4.5: 4096 tokens
    # Haiku 3.5/3: 2048 tokens
    # Sonnet (all), Opus 4.1/4: 1024 tokens
    MIN_CACHE_TOKENS_BY_MODEL = {
        "haiku-4-5": 4096,
        "opus-4-5": 4096,
        "haiku-3-5": 2048,
        "haiku-3": 2048,
    }
    MIN_CACHE_TOKENS_DEFAULT = 1024  # Sonnet models, Opus 4.1/4

    def _get_min_cache_tokens(self) -> int:
        """Get minimum cacheable tokens for current model."""
        model_lower = self.model_id.lower()
        for pattern, min_tokens in self.MIN_CACHE_TOKENS_BY_MODEL.items():
            if pattern in model_lower:
                return min_tokens
        return self.MIN_CACHE_TOKENS_DEFAULT

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(text) // 4

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Claude API with caching.

        Extracts system messages for the system parameter and adds cache_control
        markers to optimize prompt caching. Minimum token thresholds vary by model:
        - Opus 4.5, Haiku 4.5: 4096 tokens
        - Haiku 3.5/3: 2048 tokens
        - Sonnet (all), Opus 4.1/4: 1024 tokens

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

        # Cache the system prompt if it meets the model's minimum token threshold
        # (System prompt is stable; user messages change as history slides, breaking cache prefix)
        if system_content:
            min_tokens = self._get_min_cache_tokens()
            system_tokens = self._estimate_tokens(system_text)

            if system_tokens >= min_tokens:
                system_content[0]["cache_control"] = {"type": "ephemeral"}
                logger.debug(f"Cache breakpoint at system prompt: ~{system_tokens} tokens (min: {min_tokens})")
            else:
                logger.debug(f"System prompt below cache threshold: ~{system_tokens} tokens (need {min_tokens})")

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

            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        # Extract cache metrics from response (try nested object first, then flat attrs)
        usage = response.usage
        cache_creation_obj = getattr(usage, "cache_creation", None)
        if cache_creation_obj:
            cache_creation = (getattr(cache_creation_obj, "ephemeral_5m_input_tokens", 0) or 0) + \
                             (getattr(cache_creation_obj, "ephemeral_1h_input_tokens", 0) or 0)
        else:
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

        # Log usage metrics
        logger.info(
            f"Claude usage: in={usage.input_tokens}, out={usage.output_tokens}, "
            f"cache_create={cache_creation}, cache_read={cache_read}"
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


class ClaudeSDKWrapper(LLMClientWrapper):
    """Wrapper using Claude Agent SDK for session-based interactions.

    Maintains a persistent session across generate() calls within an episode.
    Only sends new observations each turn - SDK maintains conversation history.
    Use client_name='claude-sdk' to enable.
    """

    def __init__(self, client_config):
        """Initialize the ClaudeSDKWrapper with the given configuration."""
        super().__init__(client_config)
        import asyncio
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None  # ClaudeSDKClient instance
        self._llm_call_count = 0  # Actual LLM calls (not agent steps)
        self._system_prompt: str | None = None
        # Re-inject system prompt every N LLM calls (0 = never refresh)
        self._system_refresh_interval = self.client_kwargs.get("system_refresh_interval", 100)
        # Track incremental messages for monitoring
        self._last_sent: str | None = None
        self._last_received: str | None = None
        self._conversation_history: list[dict[str, str]] = []  # [{role, content}, ...]

    def _ensure_loop(self):
        """Create event loop if needed."""
        import asyncio
        if self._loop is None:
            self._loop = asyncio.new_event_loop()

    def _initialize_session(self, system_prompt: str):
        """Start new SDK session with system prompt."""
        self._ensure_loop()
        self._system_prompt = system_prompt
        self._llm_call_count = 0

        # Get thinking budget from client kwargs
        thinking_budget = self.client_kwargs.get("thinking_budget", 0)

        async def _connect():
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

            options_kwargs = {
                "system_prompt": system_prompt,
                "model": self.model_id,
                "max_turns": None,  # We control turn-by-turn externally
            }

            # Add thinking tokens if specified
            if thinking_budget and thinking_budget > 0:
                options_kwargs["max_thinking_tokens"] = thinking_budget

            options = ClaudeAgentOptions(**options_kwargs)
            self._client = ClaudeSDKClient(options)
            await self._client.connect()
            logger.info(f"ClaudeSDK connected: model={self.model_id}, thinking={thinking_budget}")

        try:
            self._loop.run_until_complete(_connect())
            logger.info(f"ClaudeSDK session initialized for model {self.model_id}")
        except Exception as e:
            logger.error(f"ClaudeSDK session init failed: {type(e).__name__}: {e}")
            raise

    def close_session(self):
        """Close current session (call at episode end)."""
        if self._client and self._loop:
            async def _disconnect():
                await self._client.disconnect()
            try:
                self._loop.run_until_complete(_disconnect())
            except Exception as e:
                logger.warning(f"Error closing SDK session: {e}")
            self._client = None
        self._llm_call_count = 0
        self._conversation_history = []
        self._last_sent = None
        self._last_received = None
        logger.info("ClaudeSDK session closed")

    def get_incremental_history(self) -> list[dict[str, str]]:
        """Return conversation history for monitoring (newest first)."""
        return list(reversed(self._conversation_history))

    def generate(self, messages) -> LLMResponse:
        """Generate response using SDK session.

        Only sends the latest user message - SDK maintains conversation history.
        System prompt is sent at session start and refreshed periodically.
        """
        # Extract system prompt from first message if present
        system_prompt = None
        user_messages = messages
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            user_messages = messages[1:]

        # Initialize session if needed
        if self._client is None:
            if system_prompt:
                self._initialize_session(system_prompt)
            else:
                # No system prompt - initialize with empty
                self._initialize_session("")

        # Periodic system prompt refresh (0 = disabled)
        self._llm_call_count += 1
        needs_refresh = (
            self._system_refresh_interval > 0
            and self._llm_call_count % self._system_refresh_interval == 0
        )

        # Get only the latest user message (new observation)
        # SDK maintains conversation history
        latest_content = user_messages[-1].content if user_messages else ""

        # Use current system_prompt (includes current memory state) for refresh
        if needs_refresh and system_prompt:
            latest_content = f"[CONTEXT REFRESH]\n{system_prompt}\n\n[Current observation]\n{latest_content}"
            logger.debug(f"Refreshing system prompt at LLM call {self._llm_call_count}")

        self._ensure_loop()

        async def _send_and_receive(content: str):
            """Send query and receive full response."""
            from claude_agent_sdk.types import AssistantMessage, ResultMessage

            response_text = ""
            input_tokens = 0
            output_tokens = 0

            # First, send the query
            await self._client.query(content)

            # Then, receive the response
            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            response_text += block.text
                elif isinstance(msg, ResultMessage):
                    # Extract usage from result
                    if msg.usage:
                        input_tokens = msg.usage.get("input_tokens", 0) or 0
                        output_tokens = msg.usage.get("output_tokens", 0) or 0

            return response_text, input_tokens, output_tokens

        def api_call():
            # Create fresh coroutine each retry attempt
            try:
                return self._loop.run_until_complete(_send_and_receive(latest_content))
            except Exception as e:
                logger.error(f"ClaudeSDK error: {type(e).__name__}: {e}")
                raise

        completion, in_tok, out_tok = self.execute_with_retries(api_call)

        # Track for monitoring
        self._last_sent = latest_content
        self._last_received = completion.strip()
        self._conversation_history.append({"role": "user", "content": latest_content})
        self._conversation_history.append({"role": "assistant", "content": self._last_received})

        logger.info(f"ClaudeSDK usage: in={in_tok}, out={out_tok}, call={self._llm_call_count}")

        return LLMResponse(
            model_id=self.model_id,
            completion=self._last_received,
            stop_reason="end_turn",
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning=None,
        )


class ClaudeToolWrapper(LLMClientWrapper):
    """Wrapper using Claude Agent SDK with MCP tools for BRAID.

    Extends ClaudeSDKWrapper with tool support via in-process MCP server.
    Tools execute automatically during response generation.
    Use client_name='claude-tools' to enable.
    """

    def __init__(self, client_config):
        """Initialize the ClaudeToolWrapper with the given configuration."""
        super().__init__(client_config)
        import asyncio

        self._loop: asyncio.AbstractEventLoop | None = None
        self._client = None
        self._llm_call_count = 0
        self._system_prompt: str | None = None
        self._system_refresh_interval = self.client_kwargs.get("system_refresh_interval", 100)
        self._last_sent: str | None = None
        self._last_received: str | None = None
        self._conversation_history: list[dict[str, str]] = []
        # MCP server for tools
        self._mcp_server = None
        # Tool call tracking for logging
        self._tool_calls: list[dict] = []

    def set_mcp_server(self, mcp_server) -> None:
        """Set the MCP server with BRAID tools."""
        self._mcp_server = mcp_server

    def _ensure_loop(self):
        """Create event loop if needed."""
        import asyncio

        if self._loop is None:
            self._loop = asyncio.new_event_loop()

    def _initialize_session(self, system_prompt: str):
        """Start new SDK session with system prompt and tools."""
        self._ensure_loop()
        self._system_prompt = system_prompt
        self._llm_call_count = 0

        thinking_budget = self.client_kwargs.get("thinking_budget", 0)

        async def _connect():
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            options_kwargs = {
                "system_prompt": system_prompt,
                "model": self.model_id,
                # max_turns must be None or high enough for tool use cycle
                # Tool use requires: tool_use → tool_result → response (multiple turns)
            }

            if thinking_budget and thinking_budget > 0:
                options_kwargs["max_thinking_tokens"] = thinking_budget

            # Add MCP server with tools if configured
            if self._mcp_server is not None:
                options_kwargs["mcp_servers"] = {"braid": self._mcp_server}
                # Allow BRAID tools + Claude's built-in TodoWrite for task tracking
                options_kwargs["allowed_tools"] = ["mcp__braid__*", "TodoWrite"]
                # Bypass permissions for headless operation - our tools are safe
                options_kwargs["permission_mode"] = "bypassPermissions"
                logger.info(f"ClaudeTools: MCP server configured: {self._mcp_server}")
            else:
                logger.warning("ClaudeTools: NO MCP server configured - tools will not be available!")

            options = ClaudeAgentOptions(**options_kwargs)
            self._client = ClaudeSDKClient(options)
            await self._client.connect()
            logger.info(f"ClaudeTools connected: model={self.model_id}, thinking={thinking_budget}")

        try:
            self._loop.run_until_complete(_connect())
            logger.info(f"ClaudeTools session initialized for model {self.model_id}")
        except Exception as e:
            logger.error(f"ClaudeTools session init failed: {type(e).__name__}: {e}")
            raise

    def close_session(self):
        """Close current session (call at episode end)."""
        if self._client and self._loop:

            async def _disconnect():
                await self._client.disconnect()

            try:
                self._loop.run_until_complete(_disconnect())
            except Exception as e:
                logger.warning(f"Error closing ClaudeTools session: {e}")
            self._client = None
        self._llm_call_count = 0
        self._conversation_history = []
        self._last_sent = None
        self._last_received = None
        self._tool_calls = []
        logger.info("ClaudeTools session closed")

    def get_incremental_history(self) -> list[dict[str, str]]:
        """Return conversation history for monitoring (newest first)."""
        return list(reversed(self._conversation_history))

    def get_tool_calls(self) -> list[dict]:
        """Return tool calls from last generate() for logging."""
        return self._tool_calls

    def generate(self, messages) -> LLMResponse:
        """Generate response using SDK session with tool execution.

        Tools execute automatically via the SDK tool runner.
        """
        # Extract system prompt
        system_prompt = None
        user_messages = messages
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            user_messages = messages[1:]

        # Initialize session if needed
        if self._client is None:
            self._initialize_session(system_prompt or "")

        # Periodic system prompt refresh
        self._llm_call_count += 1
        needs_refresh = self._system_refresh_interval > 0 and self._llm_call_count % self._system_refresh_interval == 0

        latest_content = user_messages[-1].content if user_messages else ""

        if needs_refresh and system_prompt:
            latest_content = f"[CONTEXT REFRESH]\n{system_prompt}\n\n[Current observation]\n{latest_content}"
            logger.debug(f"Refreshing system prompt at LLM call {self._llm_call_count}")

        self._ensure_loop()
        self._tool_calls = []  # Reset for this call

        async def _send_and_receive(content: str):
            """Send query and receive response, tools execute automatically."""
            from claude_agent_sdk.types import AssistantMessage, ResultMessage

            response_text = ""
            input_tokens = 0
            output_tokens = 0
            seen_tool_ids: set[str] = set()

            await self._client.query(content)

            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            response_text += block.text
                        # Track tool uses - check class name since SDK uses ToolUseBlock
                        block_type = type(block).__name__
                        if block_type == "ToolUseBlock":
                            # Use tool ID to deduplicate (SDK may return same tool multiple times)
                            tool_id = getattr(block, "id", None)
                            if tool_id and tool_id in seen_tool_ids:
                                continue
                            if tool_id:
                                seen_tool_ids.add(tool_id)
                            tool_call = {
                                "name": getattr(block, "name", "unknown"),
                                "input": getattr(block, "input", {}),
                            }
                            self._tool_calls.append(tool_call)
                elif isinstance(msg, ResultMessage):
                    if msg.usage:
                        input_tokens = msg.usage.get("input_tokens", 0) or 0
                        output_tokens = msg.usage.get("output_tokens", 0) or 0

            return response_text, input_tokens, output_tokens

        def api_call():
            try:
                return self._loop.run_until_complete(_send_and_receive(latest_content))
            except Exception as e:
                logger.error(f"ClaudeTools error: {type(e).__name__}: {e}")
                raise

        completion, in_tok, out_tok = self.execute_with_retries(api_call)

        self._last_sent = latest_content
        self._last_received = completion.strip()
        self._conversation_history.append({"role": "user", "content": latest_content})
        self._conversation_history.append({"role": "assistant", "content": self._last_received})

        logger.info(f"ClaudeTools usage: in={in_tok}, out={out_tok}, call={self._llm_call_count}, tools={len(self._tool_calls)}")

        return LLMResponse(
            model_id=self.model_id,
            completion=self._last_received,
            stop_reason="end_turn",
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning=None,
        )


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the client name.

    Args:
        client_config: Configuration object containing client-specific settings.

    Returns:
        callable: A factory function that returns an instance of the appropriate LLM client.

    Supported client names:
        - openai, vllm, nvidia, xai: OpenAI-compatible API
        - gemini: Google Generative AI
        - claude: Direct Anthropic Messages API (stateless, uses prompt caching)
        - claude-sdk: Claude Agent SDK (session-based, maintains context)
        - claude-tools: Claude Agent SDK with MCP tools (for BRAID agent)
    """

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower or "xai" in client_name_lower:
            # NVIDIA and XAI use OpenAI-compatible API, so we use the OpenAI wrapper
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif client_name_lower == "claude-tools":
            # SDK client with MCP tools - must match exactly
            return ClaudeToolWrapper(client_config)
        elif client_name_lower == "claude-sdk":
            # Session-based SDK client - must match exactly
            return ClaudeSDKWrapper(client_config)
        elif "claude" in client_name_lower:
            # Direct API client (default for "claude", "claude-haiku", etc.)
            return ClaudeWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
