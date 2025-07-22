"""
AI client management for the Virtual Literature Companion system.

This module provides centralized client initialization and management for LLM providers,
supporting both Anthropic and OpenAI with automatic fallback logic. Anthropic is
preferred when available, with OpenAI as a backup option.

The module handles:
- Environment variable loading from .env file
- Client initialization with proper error handling
- Model selection based on available API keys
- Fallback logic when preferred providers are unavailable
- Unified interface for LLM interactions
"""

import logging
from enum import Enum
import os
import time
import random
import asyncio
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

from virtual_literature_companion.config import LLMConfig


# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Provider availability flags
_anthropic_available = False
_openai_available = False
_anthropic_client = None
_openai_client = None
_openrouter_client = None

_anthropic_async_client = None
_openai_async_client = None
_openrouter_async_client = None

try:
    import anthropic
    _anthropic_available = True
except ImportError:
    logger.warning("Anthropic package not available. Install with: pip install anthropic")

try:
    import openai
    _openai_available = True
except ImportError:
    logger.warning("OpenAI package not available. Install with: pip install openai")


class Providers(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


def get_client(config: LLMConfig) -> Optional[object]:
    """
    Create and return a client.
    
    Returns:
        Optional[object]: Client or None if unavailable
    """
    if config.provider == Providers.ANTHROPIC:
        return create_anthropic_client()
    elif config.provider == Providers.OPENAI:
        return create_openai_client()
    elif config.provider == Providers.OPENROUTER:
        return create_openrouter_client()
    else:
        logger.error(f"Unsupported provider: {config.provider}")
        return None


def get_client_async(config: LLMConfig) -> Optional[object]:
    """
    Create and return an async client.
    
    Returns:
        Optional[object]: Async client or None if unavailable
    """
    if config.provider == Providers.ANTHROPIC:
        return create_anthropic_client_async()
    elif config.provider == Providers.OPENAI:
        return create_openai_client_async()
    elif config.provider == Providers.OPENROUTER:
        return create_openrouter_client_async()
    else:
        logger.error(f"Unsupported provider: {config.provider}")
        return None


def create_anthropic_client() -> Optional[object]:
    """
    Create and return an Anthropic sync client.
    
    Returns:
        Optional[anthropic.Anthropic]: Anthropic sync client or None if unavailable
    """
    global _anthropic_client
    
    if _anthropic_client is not None:
        return _anthropic_client
    
    if not _anthropic_available:
        logger.error("Anthropic package not installed")
        return None
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        return None
    
    try:
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
        logger.info("Successfully initialized Anthropic sync client")
        return _anthropic_client
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic sync client: {e}")
        return None


def create_anthropic_client_async() -> Optional[object]:
    """
    Create and return an Anthropic async client.
    
    Returns:
        Optional[anthropic.AsyncAnthropic]: Anthropic async client or None if unavailable
    """
    global _anthropic_async_client
    
    if _anthropic_async_client is not None:
        return _anthropic_async_client
    
    if not _anthropic_available:
        logger.error("Anthropic package not installed")
        return None
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        return None
    
    try:
        _anthropic_async_client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info("Successfully initialized Anthropic async client")
        return _anthropic_async_client
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic async client: {e}")
        return None


def create_openai_client() -> Optional[object]:
    """
    Create and return an OpenAI sync client.
    
    Returns:
        Optional[openai.OpenAI]: OpenAI sync client or None if unavailable
    """
    global _openai_client
    
    if _openai_client is not None:
        return _openai_client
    
    if not _openai_available:
        logger.error("OpenAI package not installed")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return None
    
    try:
        _openai_client = openai.OpenAI(api_key=api_key)
        logger.info("Successfully initialized OpenAI sync client")
        return _openai_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI sync client: {e}")
        return None


def create_openai_client_async() -> Optional[object]:
    """
    Create and return an OpenAI async client.
    
    Returns:
        Optional[openai.AsyncOpenAI]: OpenAI async client or None if unavailable
    """
    global _openai_async_client
    
    if _openai_async_client is not None:
        return _openai_async_client
    
    if not _openai_available:
        logger.error("OpenAI package not installed")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return None
    
    try:
        _openai_async_client = openai.AsyncOpenAI(api_key=api_key)
        logger.info("Successfully initialized OpenAI async client")
        return _openai_async_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI async client: {e}")
        return None
    
def create_openrouter_client() -> Optional[object]:
    """
    Create and return an OpenRouter client.
    
    Returns:
        Optional[openrouter.OpenRouter]: OpenRouter client or None if unavailable
    """
    global _openrouter_client

    if _openrouter_client is not None:
        return _openrouter_client
    
    if not _openai_available:
        logger.error("OpenAI package not installed, which is required for OpenRouter")
        return None
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        return None
    
    try:
        _openrouter_client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        logger.info("Successfully initialized OpenRouter client")
        return _openrouter_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter client: {e}")
        return None
    

def create_openrouter_client_async() -> Optional[object]:
    """
    Create and return an OpenRouter async client.
    
    Returns:
        Optional[openai.AsyncOpenAI]: OpenRouter async client or None if unavailable
    """
    global _openrouter_async_client

    if _openrouter_async_client is not None:
        return _openrouter_async_client
    
    if not _openai_available:
        logger.error("OpenAI package not installed, which is required for OpenRouter")
        return None
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment")
        return None
    
    try:
        _openrouter_async_client = openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        logger.info("Successfully initialized OpenRouter async client")
        return _openrouter_async_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter async client: {e}")
        return None


def make_llm_request(
    messages: list,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    config: LLMConfig = None
) -> Optional[str]:
    """
    Make a request to the LLM using the best available provider with exponential backoff.
    
    This function provides a unified interface for LLM requests, handling
    the differences between Anthropic and OpenAI APIs automatically. It includes
    exponential backoff retry logic to handle rate limiting and temporary failures.
    
    Args:
        messages (list): List of message dictionaries
        max_tokens (int): Maximum tokens to generate
        system_message (Optional[str]): Optional system message
        max_retries (int): Maximum number of retry attempts (default: 3)
        base_delay (float): Initial delay in seconds between retries (default: 1.0)
        max_delay (float): Maximum delay in seconds between retries (default: 60.0)
        backoff_factor (float): Multiplier for exponential delay growth (default: 2.0)
        config: LLMConfig
        
    Returns:
        Optional[str]: LLM response content or None if all attempts failed
    """
    client = get_client(config)
    
    if client is None:
        logger.error("No LLM client available for request")
        return None
    
    model = config.model
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if config.provider == Providers.ANTHROPIC:
                return _make_anthropic_request(
                    client, messages, max_tokens, system_message, config
                )
            elif config.provider in [Providers.OPENAI, Providers.OPENROUTER]:
                return _make_openai_request(
                    client, messages, max_tokens, system_message, config
                )
            else:
                logger.error(f"Unsupported provider: {config.provider}")
                return None
                
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable error
            if not _is_retryable_error(e, config):
                logger.error(f"Non-retryable error with {config.provider}: {e}")
                return None
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                jitter = delay * 0.1 * random.random()  # Add up to 10% jitter
                total_delay = delay + jitter
                
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}) "
                    f"with {config.provider}: {e}. Retrying in {total_delay:.2f}s"
                )
                time.sleep(total_delay)
            else:
                logger.error(
                    f"All retry attempts exhausted for {config.provider}. "
                    f"Final error: {last_exception}"
                )
    
    return None


async def make_llm_request_async(
    messages: list,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    config: LLMConfig = None
) -> Optional[str]:
    """
    Make a request to the LLM using the best available provider with exponential backoff.
    
    This function provides a unified interface for LLM requests, handling
    the differences between Anthropic and OpenAI APIs automatically. It includes
    exponential backoff retry logic to handle rate limiting and temporary failures.
    
    Args:
        messages (list): List of message dictionaries
        max_tokens (int): Maximum tokens to generate
        system_message (Optional[str]): Optional system message
        max_retries (int): Maximum number of retry attempts (default: 3)
        base_delay (float): Initial delay in seconds between retries (default: 1.0)
        max_delay (float): Maximum delay in seconds between retries (default: 60.0)
        backoff_factor (float): Multiplier for exponential delay growth (default: 2.0)
        
    Returns:
        Optional[str]: LLM response content or None if all attempts failed
    """
    client = get_client_async(config)
    
    if client is None:
        logger.error("No LLM client available for request")
        return None
    
    model = config.model
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if config.provider == Providers.ANTHROPIC:
                return await _make_anthropic_request_async(
                    client, messages, max_tokens, system_message, config
                )
            elif config.provider in [Providers.OPENAI, Providers.OPENROUTER]:
                return await _make_openai_request_async(
                    client, messages, max_tokens, system_message, config
                )
            else:
                logger.error(f"Unsupported provider: {config.provider}")
                return None
                
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable error
            if not _is_retryable_error(e, config):
                logger.error(f"Non-retryable error with {config.provider}: {e}")
                return None
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                jitter = delay * 0.1 * random.random()  # Add up to 10% jitter
                total_delay = delay + jitter
                
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}) "
                    f"with {config.provider}: {e}. Retrying in {total_delay:.2f}s"
                )
                await asyncio.sleep(total_delay)
            else:
                logger.error(
                    f"All retry attempts exhausted for {config.provider}. "
                    f"Final error: {last_exception}"
                )
    
    return None


def _is_retryable_error(error: Exception, config: LLMConfig) -> bool:
    """
    Determine if an error is retryable based on the error type and provider.
    
    Args:
        error (Exception): The exception that occurred
        config (LLMConfig): The LLM configuration, containing the provider.
        
    Returns:
        bool: True if the error should be retried, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Common retryable error patterns
    retryable_patterns = [
        "rate limit", "rate_limit", "ratelimit",
        "timeout", "timed out",
        "connection", "network",
        "503", "502", "500", "429",
        "service unavailable", "bad gateway",
        "internal server error", "too many requests"
    ]
    
    # Check for retryable error patterns in the error message
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True
    
    # Provider-specific error handling
    if config.provider == Providers.ANTHROPIC:
        # Anthropic-specific retryable errors
        if "anthropic" in error_str and any(x in error_str for x in ["overloaded", "busy"]):
            return True
    elif config.provider in [Providers.OPENAI, Providers.OPENROUTER]:
        # OpenAI-specific retryable errors
        if hasattr(error, 'status_code'):
            # OpenAI client typically raises errors with status codes
            return getattr(error, 'status_code') in [429, 500, 502, 503, 504]
    
    # Don't retry authentication errors, invalid requests, etc.
    non_retryable_patterns = [
        "invalid", "unauthorized", "forbidden", "authentication",
        "api key", "apikey", "permission", "access denied", "401", "403"
    ]
    
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False
    
    # Default to retrying for unknown errors (conservative approach)
    return True


def _make_anthropic_request(
    client: object,
    messages: list,
    max_tokens: Optional[int],
    system_message: Optional[str],
    config: LLMConfig
) -> Optional[str]:
    """Make a request to Anthropic's API."""
    try:
        # Prepare messages for Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] != "system":  # Anthropic handles system messages separately
                formatted_messages.append(msg)
        
        kwargs = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": float(config.temperature),
            "messages": formatted_messages
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = client.messages.create(**kwargs)
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Anthropic API request failed: {e}")
        return None


async def _make_anthropic_request_async(
    client: object,
    messages: list,
    max_tokens: Optional[int],
    system_message: Optional[str],
    config: LLMConfig
) -> Optional[str]:
    """Make a request to Anthropic's API."""
    try:
        # Prepare messages for Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] != "system":  # Anthropic handles system messages separately
                formatted_messages.append(msg)
        
        kwargs = {
            "model": config.model,
            "max_tokens": max_tokens,
            "temperature": float(config.temperature),
            "messages": formatted_messages
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = await client.messages.create(**kwargs)
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Anthropic API request failed: {e}")
        return None


def _make_openai_request(
    client: object,
    messages: list,
    max_tokens: Optional[int],
    system_message: Optional[str],
    config: LLMConfig
) -> Optional[str]:
    """Make a request to OpenAI's API."""
    try:
        # Prepare messages for OpenAI format
        formatted_messages = []
        
        if system_message:
            formatted_messages.append({"role": "system", "content": system_message})
        
        formatted_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=config.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=float(config.temperature)
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        return None


async def _make_openai_request_async(
    client: object,
    messages: list,
    max_tokens: Optional[int],
    system_message: Optional[str],
    config: LLMConfig
) -> Optional[str]:
    """Make a request to OpenAI's API."""
    try:
        # Prepare messages for OpenAI format
        formatted_messages = []
        
        if system_message:
            formatted_messages.append({"role": "system", "content": system_message})
        
        formatted_messages.extend(messages)
        
        response = await client.chat.completions.create(
            model=config.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=config.temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API request failed: {e}")
        return None


def get_ai_status() -> Dict[str, Any]:
    """
    Get comprehensive status information about AI providers.
    
    Returns:
        Dict[str, Any]: Status information including availability and configuration
    """
    status = {
        "providers": {
            "anthropic": _anthropic_available,
            "openai": _openai_available,
            "openrouter": _openai_available
        },
        "codegen_model": {
            "provider": os.getenv("CODEGEN_LLM_PROVIDER"),
            "model": os.getenv("CODEGEN_LLM")
        },
        "text_clean_model": {
            "provider": os.getenv("TEXT_CLEAN_LLM_PROVIDER"),
            "model": os.getenv("TEXT_CLEAN_LLM")
        },
        "packages_installed": {
            "anthropic": _anthropic_available,
            "openai": _openai_available
        },
        "api_keys_configured": {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY"))
        }
    }
    
    return status
