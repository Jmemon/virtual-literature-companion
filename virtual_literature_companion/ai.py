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
import os
import time
import random
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Provider availability flags
_anthropic_available = False
_openai_available = False
_anthropic_client = None
_openai_client = None

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


class AIProvider:
    """Enumeration of supported AI providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    NONE = "none"


def get_available_providers() -> Dict[str, bool]:
    """
    Get information about available AI providers.
    
    Returns:
        Dict[str, bool]: Dictionary mapping provider names to availability
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    return {
        AIProvider.ANTHROPIC: _anthropic_available and bool(anthropic_key),
        AIProvider.OPENAI: _openai_available and bool(openai_key),
    }


def get_preferred_provider() -> str:
    """
    Get the preferred AI provider based on availability and configuration.
    
    Priority order:
    1. Anthropic (if API key available)
    2. OpenAI (if API key available)
    3. None (no providers available)
    
    Returns:
        str: The preferred provider name or "none" if none available
    """
    available = get_available_providers()
    
    if available[AIProvider.ANTHROPIC]:
        return AIProvider.ANTHROPIC
    elif available[AIProvider.OPENAI]:
        return AIProvider.OPENAI
    else:
        return AIProvider.NONE


def create_anthropic_client() -> Optional[object]:
    """
    Create and return an Anthropic client.
    
    Returns:
        Optional[anthropic.Anthropic]: Anthropic client or None if unavailable
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
        logger.info("Successfully initialized Anthropic client")
        return _anthropic_client
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        return None


def create_openai_client() -> Optional[object]:
    """
    Create and return an OpenAI client.
    
    Returns:
        Optional[openai.OpenAI]: OpenAI client or None if unavailable
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
        logger.info("Successfully initialized OpenAI client")
        return _openai_client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


def get_llm_client() -> tuple[Optional[object], str]:
    """
    Get the best available LLM client based on provider priority.
    
    This is the main function that should be used throughout the codebase
    to get an LLM client. It automatically selects the best available provider.
    
    Returns:
        tuple[Optional[object], str]: (client, provider_name) or (None, "none")
    """
    provider = get_preferred_provider()
    
    if provider == AIProvider.ANTHROPIC:
        client = create_anthropic_client()
        if client is not None:
            return client, AIProvider.ANTHROPIC
        
        # Fallback to OpenAI if Anthropic fails
        logger.warning("Anthropic client failed, falling back to OpenAI")
        provider = AIProvider.OPENAI
    
    if provider == AIProvider.OPENAI:
        client = create_openai_client()
        if client is not None:
            return client, AIProvider.OPENAI
    
    logger.warning("No LLM providers available")
    return None, AIProvider.NONE


def get_model_name(provider: str) -> str:
    """
    Get the appropriate model name for the given provider.
    
    Args:
        provider (str): Provider name (anthropic, openai, or none)
        
    Returns:
        str: Model name for the provider
    """
    if provider == AIProvider.ANTHROPIC:
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    elif provider == AIProvider.OPENAI:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        return "none"


def make_llm_request(
    messages: list,
    max_tokens: int = 200,
    temperature: float = 0.1,
    system_message: Optional[str] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Optional[str]:
    """
    Make a request to the LLM using the best available provider with exponential backoff.
    
    This function provides a unified interface for LLM requests, handling
    the differences between Anthropic and OpenAI APIs automatically. It includes
    exponential backoff retry logic to handle rate limiting and temporary failures.
    
    Args:
        messages (list): List of message dictionaries
        max_tokens (int): Maximum tokens to generate
        temperature (float): Temperature for generation
        system_message (Optional[str]): Optional system message
        max_retries (int): Maximum number of retry attempts (default: 3)
        base_delay (float): Initial delay in seconds between retries (default: 1.0)
        max_delay (float): Maximum delay in seconds between retries (default: 60.0)
        backoff_factor (float): Multiplier for exponential delay growth (default: 2.0)
        
    Returns:
        Optional[str]: LLM response content or None if all attempts failed
    """
    client, provider = get_llm_client()
    
    if client is None:
        logger.error("No LLM client available for request")
        return None
    
    model = get_model_name(provider)
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            if provider == AIProvider.ANTHROPIC:
                return _make_anthropic_request(
                    client, messages, max_tokens, temperature, system_message, model
                )
            elif provider == AIProvider.OPENAI:
                return _make_openai_request(
                    client, messages, max_tokens, temperature, system_message, model
                )
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None
                
        except Exception as e:
            last_exception = e
            
            # Check if this is a retryable error
            if not _is_retryable_error(e, provider):
                logger.error(f"Non-retryable error with {provider}: {e}")
                return None
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                jitter = delay * 0.1 * random.random()  # Add up to 10% jitter
                total_delay = delay + jitter
                
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}) "
                    f"with {provider}: {e}. Retrying in {total_delay:.2f}s"
                )
                time.sleep(total_delay)
            else:
                logger.error(
                    f"All retry attempts exhausted for {provider}. "
                    f"Final error: {last_exception}"
                )
    
    return None


def _is_retryable_error(error: Exception, provider: str) -> bool:
    """
    Determine if an error is retryable based on the error type and provider.
    
    Args:
        error (Exception): The exception that occurred
        provider (str): The provider name (anthropic or openai)
        
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
    if provider == AIProvider.ANTHROPIC:
        # Anthropic-specific retryable errors
        if "anthropic" in error_str and any(x in error_str for x in ["overloaded", "busy"]):
            return True
    elif provider == AIProvider.OPENAI:
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
    max_tokens: int,
    temperature: float,
    system_message: Optional[str],
    model: str
) -> Optional[str]:
    """Make a request to Anthropic's API."""
    try:
        # Prepare messages for Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] != "system":  # Anthropic handles system messages separately
                formatted_messages.append(msg)
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = client.messages.create(**kwargs)
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Anthropic API request failed: {e}")
        return None


def _make_openai_request(
    client: object,
    messages: list,
    max_tokens: int,
    temperature: float,
    system_message: Optional[str],
    model: str
) -> Optional[str]:
    """Make a request to OpenAI's API."""
    try:
        # Prepare messages for OpenAI format
        formatted_messages = []
        
        if system_message:
            formatted_messages.append({"role": "system", "content": system_message})
        
        formatted_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature
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
    available = get_available_providers()
    preferred = get_preferred_provider()
    
    status = {
        "providers": available,
        "preferred_provider": preferred,
        "models": {
            "anthropic": get_model_name(AIProvider.ANTHROPIC),
            "openai": get_model_name(AIProvider.OPENAI)
        },
        "packages_installed": {
            "anthropic": _anthropic_available,
            "openai": _openai_available
        },
        "api_keys_configured": {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY"))
        }
    }
    
    return status


# Backward compatibility functions for existing code
def get_openai_client() -> Optional[object]:
    """
    Backward compatibility function for existing OpenAI-specific code.
    
    Returns:
        Optional[object]: OpenAI client or None
    """
    return create_openai_client()


def get_default_model() -> str:
    """
    Get the default model for the preferred provider.
    
    Returns:
        str: Default model name
    """
    provider = get_preferred_provider()
    return get_model_name(provider) 