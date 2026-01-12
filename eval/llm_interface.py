"""
LLM (Language Model) Interface for PPT Evaluation

Supports text-only language model backends:
- OpenAI GPT-4/GPT-3.5
- Anthropic Claude
- Google Gemini
- Local models via Ollama

Author: PPT Evaluation System
"""

import os
import json
import logging
import time
import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM_GOOGLE = "custom-google"
    OLLAMA = "ollama"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM Interface."""
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0
    max_retries: int = 10
    retry_delay: float = 2.0
    timeout: int = 240
    fallback_models: Optional[List[str]] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Make an LLM call with text prompt only."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make an OpenAI LLM call."""
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=0.95
        )
        
        return response.choices[0].message.content


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make an Anthropic Claude LLM call."""
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text


class GoogleClient(BaseLLMClient):
    """Google Gemini client (official SDK)."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not provided")
        
        try:
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make a Google Gemini LLM call."""
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature
            )
        )
        
        return response.text


class CustomGoogleClient(BaseLLMClient):
    """Custom Google Gemini client with HTTP API and Bearer token authentication."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = os.getenv("CUSTOM_GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not provided")
        
        # Set base URL - support custom endpoints
        self.base_url = os.getenv("GOOGLE_BASE_URL", "https://gemini.visioncoder.cn").rstrip('/')
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make a Custom Google Gemini LLM call via HTTP API."""
        import requests
        
        # Construct API URL
        api_url = f"{self.base_url}/v1beta/models/{self.config.model_name}:generateContent"
        
        # Prepare request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        # Add generation config if temperature is set
        if self.config.temperature is not None:
            payload["generationConfig"] = {
                "temperature": self.config.temperature
            }
        
        # Make request with Bearer token authentication
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]
        
        raise ValueError(f"Unexpected response format from Gemini API: {result}")


class OllamaClient(BaseLLMClient):
    """Ollama local model client."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make an Ollama LLM call."""
        import requests
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        return response.json()["response"]


class MockClient(BaseLLMClient):
    """Mock client for testing."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def call(self, prompt: str, **kwargs) -> str:
        """Return mock responses based on prompt type."""
        logger.info(f"[MOCK] Processing text-only prompt...")
        
        # Return mock JSON response
        return json.dumps({
            "Content": {
                "score": random.randint(6, 9),
                "reason": "Mock evaluation result",
                "sub_scores": {
                    "Accuracy": random.randint(6, 9),
                    "Completeness": random.randint(5, 8),
                    "Logical_Flow": random.randint(6, 9),
                    "Clarity": random.randint(7, 9)
                }
            }
        })


class LLMInterface:
    """
    Unified interface for Language Model interactions (text-only).
    
    Supports multiple backends: OpenAI, Anthropic, Google, Ollama, and Mock.
    
    Usage:
        # Using OpenAI (default)
        llm = LLMInterface()
        result = llm.call_llm(prompt)
        
        # Using Anthropic Claude
        llm = LLMInterface(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
        result = llm.call_llm(prompt)
    """
    
    # Fallback models for each provider
    FALLBACK_MODELS = {
        "openai": ["qwen-vl-max", "qwen-vl-max", "qwen-vl-max"],
        "anthropic": ["claude-3-haiku-20240307", "claude-3-opus-20240229"],
        "google": ["gemini-3-flash-preview"],
        "custom-google": ["gemini-3-flash-preview", "gemini-3-pro-preview"],
    }
    
    # Error codes that indicate overload/rate limiting
    OVERLOAD_ERROR_CODES = [503, 529, 500, 502, 504]
    RATE_LIMIT_ERROR_CODES = [429]
    
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        max_retries: int = 10,
        fallback_models: List[str] = None,
        **kwargs
    ):
        """
        Initialize LLM Interface.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "google", "ollama", "mock")
            model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key (defaults to environment variable)
            base_url: Custom API base URL
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Number of retry attempts on failure
            fallback_models: List of fallback model names when primary model fails
        """
        # Determine provider from environment or default
        provider = provider or os.getenv("LLM_PROVIDER", "openai")
        
        # Map string to enum
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "google": LLMProvider.GOOGLE,
            "custom-google": LLMProvider.CUSTOM_GOOGLE,
            "ollama": LLMProvider.OLLAMA,
            "mock": LLMProvider.MOCK
        }
        
        provider_enum = provider_map.get(provider.lower(), LLMProvider.OPENAI)
        
        # Set default model based on provider
        default_models = {
            LLMProvider.OPENAI: "qwen-turbo",
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            LLMProvider.GOOGLE: "gemini-3-flash-preview",
            LLMProvider.CUSTOM_GOOGLE: "gemini-2.5-flash",
            LLMProvider.OLLAMA: "llama2",
            LLMProvider.MOCK: "mock"
        }
        
        model_name = model_name or os.getenv("LLM_MODEL") or default_models[provider_enum]
        
        # Set fallback models
        if fallback_models:
            self.fallback_models = fallback_models
        elif os.getenv("LLM_FALLBACK_MODELS"):
            self.fallback_models = os.getenv("LLM_FALLBACK_MODELS").split(",")
        else:
            self.fallback_models = self.FALLBACK_MODELS.get(provider, [])
        
        # Create config
        self.config = LLMConfig(
            provider=provider_enum,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            fallback_models=self.fallback_models
        )
        
        # Store original model name for fallback recovery
        self._original_model = model_name
        self._current_fallback_index = -1
        
        # Initialize client
        self.client = self._create_client()
        
        logger.info(f"LLM Interface initialized: provider={provider_enum.value}, model={model_name}")
        if self.fallback_models:
            logger.info(f"Fallback models: {self.fallback_models}")
    
    def _create_client(self) -> BaseLLMClient:
        """Create the appropriate LLM client based on provider."""
        clients = {
            LLMProvider.OPENAI: OpenAIClient,
            LLMProvider.ANTHROPIC: AnthropicClient,
            LLMProvider.GOOGLE: GoogleClient,
            LLMProvider.CUSTOM_GOOGLE: CustomGoogleClient,
            LLMProvider.OLLAMA: OllamaClient,
            LLMProvider.MOCK: MockClient
        }
        
        client_class = clients.get(self.config.provider, MockClient)
        return client_class(self.config)
    
    def call_llm(
        self,
        prompt: str,
        parse_json: bool = True,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Call the LLM with a text prompt.
        
        Args:
            prompt: Text prompt for the LLM
            parse_json: Whether to parse the response as JSON
            **kwargs: Additional arguments passed to the client
        
        Returns:
            Response string or parsed JSON dict
        """
        logger.info(f"Calling LLM ({self.config.model_name})...")
        
        last_error = None
        models_to_try = [self.config.model_name] + self.fallback_models
        
        for model_idx, current_model in enumerate(models_to_try):
            self.config.model_name = current_model
            self.client = self._create_client()
            
            for attempt in range(self.config.max_retries):
                try:
                    response = self.client.call(prompt, **kwargs)
                    
                    if parse_json:
                        return self._extract_json(response)
                    return response
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"LLM call failed (model={current_model}, attempt {attempt+1}/{self.config.max_retries}): {e}")
                    
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
        
        # Restore original model for future calls
        self.config.model_name = self._original_model
        self.client = self._create_client()
        
        raise RuntimeError(f"LLM call failed after trying all models ({models_to_try}). Last error: {last_error}")
    
    def _extract_json(self, response: str) -> Union[str, Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        import re
        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue
        
        # Try parsing the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return response
    
    @property
    def provider(self) -> str:
        """Get current provider name."""
        return self.config.provider.value
    
    @property
    def model(self) -> str:
        """Get current model name."""
        return self.config.model_name


# Convenience functions
def create_llm(provider: str = "openai", **kwargs) -> LLMInterface:
    """Factory function to create LLM interface."""
    return LLMInterface(provider=provider, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available LLM providers."""
    return [p.value for p in LLMProvider]
