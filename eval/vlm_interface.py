"""
VLM (Vision-Language Model) Interface for PPT Evaluation

Supports multiple VLM backends:
- OpenAI GPT-4V/GPT-4o
- Anthropic Claude 3.5 Sonnet/Claude 3 Opus
- Google Gemini Pro Vision
- Local models via Ollama

Author: PPT Evaluation System
"""

import base64
import json
import os
import re
import random
import time
import logging
import io
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
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

# Image optimization settings
MAX_IMAGE_DIMENSION = 2048  # Max width/height for images
JPEG_QUALITY = 85  # Quality for JPEG compression
MAX_IMAGE_SIZE_KB = 500  # Max size per image in KB

# Image concatenation settings
# Grid layout options - choose based on quality vs efficiency tradeoff:
#   - 4x2 (8 slides): More coverage, lower per-slide resolution
#   - 2x2 (4 slides): Better resolution, less coverage per grid
#   - 3x2 (6 slides): Balanced option
CONCAT_GRID_COLS = 2  # Number of columns in concatenated grid
CONCAT_GRID_ROWS = 2  # Number of rows in concatenated grid
CONCAT_SLIDES_PER_IMAGE = CONCAT_GRID_COLS * CONCAT_GRID_ROWS  # slides per concatenated image

# Cell dimensions for each slide in the grid
# Higher resolution = better text readability, but larger file size
# Recommended: 768x432 (good balance) or 1024x576 (high quality)
CONCAT_CELL_WIDTH = 1024   # Width of each cell in pixels
CONCAT_CELL_HEIGHT = 576  # Height of each cell in pixels (16:9 aspect ratio)

CONCAT_OUTPUT_DIR = None  # Set to a path to save concatenated images for inspection


class VLMProvider(Enum):
    """Supported VLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM_GOOGLE = "custom-google"
    OLLAMA = "ollama"
    MOCK = "mock"


@dataclass
class VLMConfig:
    """Configuration for VLM Interface."""
    provider: VLMProvider = VLMProvider.OPENAI
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0
    max_retries: int = 10  # Increased from 3
    retry_delay: float = 2.0  # Increased from 2.0
    timeout: int = 240  # Increased from 120
    fallback_models: Optional[List[str]] = None  # Fallback models when primary fails


def concatenate_images_to_grid(
    image_paths: List[str],
    cols: int = CONCAT_GRID_COLS,
    rows: int = CONCAT_GRID_ROWS,
    cell_width: int = CONCAT_CELL_WIDTH,
    cell_height: int = CONCAT_CELL_HEIGHT,
    add_labels: bool = True,
    label_prefix: str = "",
    start_index: int = 1,
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Concatenate multiple images into a single grid image.
    
    Args:
        image_paths: List of paths to images to concatenate
        cols: Number of columns in the grid
        rows: Number of rows in the grid
        cell_width: Width of each cell in pixels
        cell_height: Height of each cell in pixels
        add_labels: Whether to add slide number labels
        label_prefix: Prefix for labels (e.g., "A_" or "B_")
        start_index: Starting slide number for labels (e.g., 1 for first grid, 9 for second grid)
        save_path: Optional path to save the concatenated image
        
    Returns:
        PIL Image object of the concatenated grid, or None if failed
    """
    try:
        import PIL.Image
        import PIL.ImageDraw
        import PIL.ImageFont
    except ImportError:
        logger.error("PIL not available, cannot concatenate images")
        return None
    
    if not image_paths:
        return None
    
    max_images = cols * rows
    images_to_use = image_paths[:max_images]
    
    # Create blank canvas
    grid_width = cols * cell_width
    grid_height = rows * cell_height
    grid_image = PIL.Image.new('RGB', (grid_width, grid_height), color='white')
    
    draw = PIL.ImageDraw.Draw(grid_image)
    
    # Try to load a font for labels
    font = None
    if add_labels:
        try:
            # Try to use a system font
            font = PIL.ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            try:
                font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = PIL.ImageFont.load_default()
    
    for idx, img_path in enumerate(images_to_use):
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue
        
        row = idx // cols
        col = idx % cols
        x_offset = col * cell_width
        y_offset = row * cell_height
        
        try:
            with PIL.Image.open(img_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                
                # Resize to fit cell while maintaining aspect ratio
                img.thumbnail((cell_width - 4, cell_height - 4), PIL.Image.Resampling.LANCZOS)
                
                # Center the image in the cell
                paste_x = x_offset + (cell_width - img.width) // 2
                paste_y = y_offset + (cell_height - img.height) // 2
                
                grid_image.paste(img, (paste_x, paste_y))
                
                # Add label - use start_index to get correct slide number
                if add_labels:
                    slide_num = start_index + idx
                    label = f"{label_prefix}slide_{slide_num}"
                    # Draw label background
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    label_x = x_offset + 5
                    label_y = y_offset + 5
                    
                    # Draw semi-transparent background for label
                    draw.rectangle(
                        [label_x - 2, label_y - 2, label_x + text_width + 4, label_y + text_height + 4],
                        fill=(0, 0, 0, 180)
                    )
                    draw.text((label_x, label_y), label, fill='white', font=font)
                    
                # Draw cell border
                draw.rectangle(
                    [x_offset, y_offset, x_offset + cell_width - 1, y_offset + cell_height - 1],
                    outline='gray',
                    width=1
                )
                
        except Exception as e:
            logger.warning(f"Failed to process image {img_path}: {e}")
            # Draw placeholder for failed image
            draw.rectangle(
                [x_offset + 2, y_offset + 2, x_offset + cell_width - 3, y_offset + cell_height - 3],
                fill='lightgray'
            )
            draw.text(
                (x_offset + 10, y_offset + cell_height // 2),
                f"Error: {os.path.basename(img_path)}",
                fill='red',
                font=font
            )
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grid_image.save(save_path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
        logger.info(f"Saved concatenated image to: {save_path}")
    
    return grid_image


def encode_concatenated_image(
    image_paths: List[str],
    cols: int = CONCAT_GRID_COLS,
    rows: int = CONCAT_GRID_ROWS,
    label_prefix: str = "",
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Concatenate images and encode to base64.
    
    Args:
        image_paths: List of image paths
        cols: Grid columns
        rows: Grid rows
        label_prefix: Label prefix for slides
        save_path: Optional path to save the grid image
        
    Returns:
        Base64 encoded string or None if failed
    """
    grid_image = concatenate_images_to_grid(
        image_paths,
        cols=cols,
        rows=rows,
        add_labels=True,
        label_prefix=label_prefix,
        save_path=save_path
    )
    
    if grid_image is None:
        return None
    
    # Encode to base64
    buffer = io.BytesIO()
    grid_image.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_slide_grids(
    image_paths: List[str],
    slides_per_grid: int = CONCAT_SLIDES_PER_IMAGE,
    cols: int = CONCAT_GRID_COLS,
    rows: int = CONCAT_GRID_ROWS,
    label_prefix: str = "",
    output_dir: Optional[str] = None,
    identifier: str = ""
) -> List[str]:
    """
    Create multiple grid images from a list of slide images.
    
    This function takes all slides and groups them into grids.
    For example, 16 slides with 8 slides per grid = 2 grid images.
    
    Args:
        image_paths: List of all slide image paths
        slides_per_grid: Number of slides per grid image
        cols: Number of columns per grid
        rows: Number of rows per grid
        label_prefix: Prefix for slide labels (e.g., "A_")
        output_dir: Directory to save grid images (if None, returns base64 strings)
        identifier: Identifier for saved files
        
    Returns:
        List of base64 encoded grid images (or file paths if output_dir is set)
    """
    if not image_paths:
        return []
    
    grid_results = []
    num_grids = (len(image_paths) + slides_per_grid - 1) // slides_per_grid
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * slides_per_grid
        end_idx = min(start_idx + slides_per_grid, len(image_paths))
        grid_images = image_paths[start_idx:end_idx]
        
        # Determine save path if output_dir is set
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(
                output_dir,
                f"{identifier}_{label_prefix}grid_{grid_idx + 1}_slides_{start_idx + 1}_to_{end_idx}.jpg"
            )
        
        # Encode with adjusted label numbers
        # We need to temporarily rename labels to reflect actual slide numbers
        grid_image = concatenate_images_to_grid(
            grid_images,
            cols=cols,
            rows=rows,
            add_labels=True,
            label_prefix=f"{label_prefix}",
            start_index=start_idx + 1,  # Pass correct starting slide number
            save_path=save_path
        )
        
        if grid_image is None:
            continue
        
        if output_dir and save_path:
            grid_results.append(save_path)
        else:
            # Encode to base64
            buffer = io.BytesIO()
            grid_image.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            buffer.seek(0)
            grid_results.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    
    logger.info(f"Created {len(grid_results)} grid images from {len(image_paths)} slides ({label_prefix})")
    
    return grid_results


class BaseVLMClient(ABC):
    """Abstract base class for VLM clients."""
    
    @abstractmethod
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Make a VLM call with text and images."""
        pass
    
    @staticmethod
    def encode_image(image_path: str, optimize: bool = True) -> str:
        """
        Encode an image file to base64, with optional optimization.
        
        Args:
            image_path: Path to the image file
            optimize: Whether to optimize/compress the image
            
        Returns:
            Base64 encoded string
        """
        if not optimize:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        try:
            import PIL.Image
            
            with PIL.Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                
                # Resize if too large
                original_size = img.size
                if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), PIL.Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image from {original_size} to {img.size}")
                
                # Save to bytes with compression
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                buffer.seek(0)
                
                # Check size and reduce quality if still too large
                size_kb = len(buffer.getvalue()) / 1024
                if size_kb > MAX_IMAGE_SIZE_KB:
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=90, optimize=True)
                    buffer.seek(0)
                    logger.debug(f"Reduced image quality due to size: {size_kb:.1f}KB -> {len(buffer.getvalue())/1024:.1f}KB")
                
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except ImportError:
            logger.warning("PIL not available, using raw image encoding")
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Image optimization failed for {image_path}: {e}, using raw encoding")
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def encode_image_bytes(image_path: str, optimize: bool = True) -> bytes:
        """
        Encode an image file to raw bytes, with optional optimization (returns JPEG bytes when optimized).

        Args:
            image_path: Path to the image file
            optimize: Whether to optimize/compress the image

        Returns:
            Raw image bytes
        """
        if not optimize:
            with open(image_path, "rb") as image_file:
                return image_file.read()

        try:
            import PIL.Image

            with PIL.Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')

                # Resize if too large
                original_size = img.size
                if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), PIL.Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image from {original_size} to {img.size}")

                # Save to bytes with compression (JPEG)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                buffer.seek(0)

                # Check size and reduce quality if still too large
                size_kb = len(buffer.getvalue()) / 1024
                if size_kb > MAX_IMAGE_SIZE_KB:
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=90, optimize=True)
                    buffer.seek(0)
                    logger.debug(f"Reduced image quality due to size: {size_kb:.1f}KB -> {len(buffer.getvalue())/1024:.1f}KB")

                return buffer.getvalue()

        except ImportError:
            logger.warning("PIL not available, returning raw image bytes")
            with open(image_path, "rb") as image_file:
                return image_file.read()
        except Exception as e:
            logger.warning(f"Image optimization failed for {image_path}: {e}, returning raw bytes")
            with open(image_path, "rb") as image_file:
                return image_file.read()
    
    @staticmethod
    def get_image_mime_type(image_path: str) -> str:
        """Get MIME type based on file extension."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(ext, 'image/jpeg')


class OpenAIClient(BaseVLMClient):
    """OpenAI GPT-4V/GPT-4o client."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            import httpx
            
            # Create client with retry-capable HTTP settings
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=config.base_url
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Make an OpenAI VLM call."""
        optimize_images = kwargs.get("optimize_images", True)
        content = [{"type": "text", "text": prompt}]
        
        for i, image_path in enumerate(images):
            if os.path.exists(image_path):
                # Use optimized encoding to reduce payload size
                base64_image = self.encode_image(image_path, optimize=optimize_images)
                # Always use JPEG mime type when optimizing (since we convert to JPEG)
                mime_type = "image/jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": kwargs.get("detail", "auto")
                    }
                })
            else:
                logger.warning(f"Image not found: {image_path}")
        
        messages = [{"role": "user", "content": content}]
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=0.95
        )
        
        return response.choices[0].message.content


class AnthropicClient(BaseVLMClient):
    """Anthropic Claude client."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Make an Anthropic Claude VLM call."""
        content = []
        
        for i, image_path in enumerate(images):
            if os.path.exists(image_path):
                base64_image = self.encode_image(image_path)
                mime_type = self.get_image_mime_type(image_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image
                    }
                })
            else:
                logger.warning(f"Image not found: {image_path}")
        
        content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": content}]
        )
        
        return response.content[0].text


class GoogleClient(BaseVLMClient):
    """Google Gemini client (official SDK)."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")
        
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Make a Google Gemini VLM call."""
        from google.genai import types
        
        content = []
        
        # Add images first as Part objects
        for image_path in images:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                content.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ))
            else:
                logger.warning(f"Image not found: {image_path}")
        
        # Add text prompt last
        content.append(prompt)
        
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=content,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature
            )
        )
        
        return response.text


class CustomGoogleClient(BaseVLMClient):
    """Custom Google Gemini client with HTTP API and Bearer token authentication."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.api_key = os.getenv("CUSTOM_GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not provided. Set CUSTOM_GOOGLE_API_KEY environment variable.")
        
        # Set base URL - support custom endpoints

        self.base_url = os.getenv("GOOGLE_BASE_URL", "https://gemini.visioncoder.cn").rstrip('/')
    
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Make a Custom Google Gemini VLM call via HTTP API."""
        import requests
        
        parts = []
        
        # Add images first as inline data
        for image_path in images:
            if os.path.exists(image_path):
                # Encode image to base64
                image_data = self.encode_image(image_path, optimize=True)
                mime_type = self.get_image_mime_type(image_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })
            else:
                logger.warning(f"Image not found: {image_path}")
        
        # Add text prompt
        parts.append({"text": prompt})
        
        # Construct API URL
        api_url = f"{self.base_url}/v1beta/models/{self.config.model_name}:generateContent"
        
        # Prepare request payload
        payload = {
            "contents": [
                {"parts": parts}
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


class OllamaClient(BaseVLMClient):
    """Ollama local model client."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
    
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Make an Ollama VLM call."""
        import requests
        
        image_data = []
        for image_path in images:
            if os.path.exists(image_path):
                image_data.append(self.encode_image(image_path))
            else:
                logger.warning(f"Image not found: {image_path}")
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "images": image_data,
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


class MockClient(BaseVLMClient):
    """Mock client for testing."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
    
    def call(self, prompt: str, images: List[str], **kwargs) -> str:
        """Return mock responses based on prompt type."""
        logger.info(f"[MOCK] Processing {len(images)} images...")
        
        # Detect prompt type and return appropriate mock response
        if "EVALUATION CRITERIA" in prompt and "Sub-criteria" in prompt:
            return self._mock_detailed_quantitative_response()
        elif "head-to-head comparison" in prompt.lower():
            return self._mock_arena_response()
        elif "supreme judge" in prompt.lower() or "inconsistencies" in prompt.lower():
            return self._mock_judge_response()
        elif "single slide" in prompt.lower():
            return self._mock_single_slide_response()
        
        return self._mock_quantitative_response()
    
    def _mock_detailed_quantitative_response(self) -> str:
        """Generate detailed mock response with sub-scores."""
        return json.dumps({
            "Content": {
                "score": random.randint(6, 9),
                "reason": "The presentation covers main topics from the source document with good accuracy.",
                "sub_scores": {
                    "Accuracy": random.randint(6, 9),
                    "Completeness": random.randint(5, 8),
                    "Logical_Flow": random.randint(6, 9),
                    "Clarity": random.randint(7, 9)
                }
            },
            "Visual_Design": {
                "score": random.randint(6, 9),
                "reason": "Visual design is cohesive with appropriate color scheme.",
                "sub_scores": {
                    "Color_Scheme": random.randint(6, 9),
                    "Typography": random.randint(6, 8),
                    "Visual_Consistency": random.randint(7, 9),
                    "Image_Quality": random.randint(5, 8),
                    "Theme_Appropriateness": random.randint(6, 9)
                }
            },
            "Layout": {
                "score": random.randint(6, 9),
                "reason": "Layout is well-organized with proper spacing.",
                "sub_scores": {
                    "Spatial_Balance": random.randint(6, 9),
                    "Element_Alignment": random.randint(7, 9),
                    "No_Overlapping": random.randint(7, 10),
                    "Readability": random.randint(6, 9)
                }
            },
            "Complexity": {
                "score": random.randint(5, 8),
                "reason": "Presentation includes some advanced visual elements.",
                "sub_scores": {
                    "Charts_and_Data": random.randint(4, 8),
                    "Visual_Elements": random.randint(5, 8),
                    "Advanced_Design": random.randint(4, 7),
                    "Layout_Variety": random.randint(5, 8)
                }
            },
            "Professionalism": {
                "score": random.randint(7, 9),
                "reason": "Presentation appears polished and professional.",
                "sub_scores": {
                    "Grammar_Spelling": random.randint(7, 10),
                    "Slide_Count_Appropriateness": random.randint(6, 9),
                    "Overall_Polish": random.randint(7, 9)
                }
            },
            "Weighted_Total": round(random.uniform(60, 80), 2),
            "Overall_Feedback": "This is a well-crafted presentation with good content coverage and visual design.",
            "Top_Strengths": [
                "Clear and logical content organization",
                "Consistent visual theme throughout",
                "Good use of whitespace"
            ],
            "Areas_for_Improvement": [
                "Could include more data visualizations",
                "Some images could be higher quality",
                "Consider adding more visual variety"
            ]
        })
    
    def _mock_quantitative_response(self) -> str:
        """Generate simple mock quantitative response."""
        scores = {
            "Content": random.randint(6, 10),
            "Style": random.randint(6, 10),
            "Layout": random.randint(6, 10),
            "Complexity": random.randint(5, 9)
        }
        total = sum(scores.values())
        return json.dumps({
            "Content": {"score": scores["Content"], "reason": "Good content coverage."},
            "Style": {"score": scores["Style"], "reason": "Consistent style."},
            "Layout": {"score": scores["Layout"], "reason": "Clean layout."},
            "Complexity": {"score": scores["Complexity"], "reason": "Moderate complexity."},
            "Total_Score": total
        })
    
    def _mock_arena_response(self) -> str:
        """Generate mock arena comparison response."""
        outcomes = ["A", "B", "Tie"]
        winner = random.choice(outcomes)
        return json.dumps({
            "Content": {
                "winner": random.choice(outcomes),
                "score_difference": random.randint(1, 3),
                "reason": "Better coverage of key topics and clearer organization."
            },
            "Visual_Design": {
                "winner": random.choice(outcomes),
                "score_difference": random.randint(1, 3),
                "reason": "More appealing color scheme and better typography."
            },
            "Layout": {
                "winner": random.choice(outcomes),
                "score_difference": random.randint(1, 2),
                "reason": "Cleaner layout with better use of whitespace."
            },
            "Complexity": {
                "winner": random.choice(outcomes),
                "score_difference": random.randint(1, 2),
                "reason": "More sophisticated use of charts and visual elements."
            },
            "Professionalism": {
                "winner": random.choice(outcomes),
                "score_difference": random.randint(1, 2),
                "reason": "More polished overall appearance."
            },
            "Overall_Winner": winner,
            "Confidence": random.randint(3, 5),
            "Overall_Reason": f"Presentation {winner} demonstrates better overall execution with stronger visual design and content organization.",
            "Key_Differences": [
                "Visual consistency across slides",
                "Depth of content coverage",
                "Use of supporting graphics"
            ]
        })
    
    def _mock_judge_response(self) -> str:
        """Generate mock judge agent response."""
        winner = random.choice(["A", "B", "Tie"])
        return json.dumps({
            "Final_Winner": winner,
            "Confidence": random.randint(3, 5),
            "Corrected_Scores": {
                "A": {
                    "Content": random.randint(6, 9),
                    "Visual_Design": random.randint(6, 9),
                    "Layout": random.randint(7, 9),
                    "Complexity": random.randint(5, 8),
                    "Professionalism": random.randint(7, 9),
                    "Total": random.randint(35, 42)
                },
                "B": {
                    "Content": random.randint(6, 9),
                    "Visual_Design": random.randint(6, 9),
                    "Layout": random.randint(7, 9),
                    "Complexity": random.randint(5, 8),
                    "Professionalism": random.randint(7, 9),
                    "Total": random.randint(35, 42)
                }
            },
            "Inconsistency_Analysis": "The arena comparison focused more on visual appeal while quantitative scoring weighted content more heavily.",
            "Reason": f"After careful re-examination, Presentation {winner} shows superior overall quality when considering all criteria equally.",
            "Recommendations": [
                "Consider adjusting criterion weights for future evaluations",
                "Add per-slide comparison for more granular assessment"
            ]
        })
    
    def _mock_single_slide_response(self) -> str:
        """Generate mock single slide analysis response."""
        return json.dumps({
            "slide_number": 1,
            "content_score": random.randint(6, 9),
            "design_score": random.randint(6, 9),
            "layout_score": random.randint(7, 9),
            "issues": ["Text slightly small", "Image could be higher resolution"],
            "strengths": ["Clear title", "Good color contrast"],
            "suggestions": ["Increase font size", "Use vector graphics"]
        })


class VLMInterface:
    """
    Unified interface for Vision-Language Model interactions.
    
    Supports multiple backends: OpenAI, Anthropic, Google, Ollama, and Mock.
    
    Usage:
        # Using OpenAI (default)
        vlm = VLMInterface()
        result = vlm.call_vlm(prompt, image_paths)
        
        # Using Anthropic Claude
        vlm = VLMInterface(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
        result = vlm.call_vlm(prompt, image_paths)
        
        # Using mock for testing
        vlm = VLMInterface(provider="mock")
        result = vlm.call_vlm(prompt, image_paths)
    """
    
    # Fallback models for each provider when primary model is overloaded
    FALLBACK_MODELS = {
        "openai": ["qwen-vl-max", "qwen-vl-max", "qwen-vl-max"],
        "anthropic": ["claude-3-haiku-20240307", "claude-3-opus-20240229"],
        "google": ["gemini-3-flash-preview"],
        "custom-google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    }
    
    # Error codes that indicate overload/rate limiting (should use longer delays or fallback)
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
        Initialize VLM Interface.
        
        Args:
            provider: VLM provider ("openai", "anthropic", "google", "ollama", "mock")
            model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            api_key: API key (defaults to environment variable)
            base_url: Custom API base URL
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Number of retry attempts on failure
            fallback_models: List of fallback model names when primary model fails
        """
        # Determine provider from environment or default
        provider = provider or os.getenv("VLM_PROVIDER", "openai")
        
        # Map string to enum
        provider_map = {
            "openai": VLMProvider.OPENAI,
            "anthropic": VLMProvider.ANTHROPIC,
            "google": VLMProvider.GOOGLE,
            "custom-google": VLMProvider.CUSTOM_GOOGLE,
            "ollama": VLMProvider.OLLAMA,
            "mock": VLMProvider.MOCK
        }
        
        provider_enum = provider_map.get(provider.lower(), VLMProvider.OPENAI)
        
        # Set default model based on provider
        default_models = {
            VLMProvider.OPENAI: "qwen-vl-max", # qwen-vl-max
            VLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            VLMProvider.GOOGLE: "gemini-3-flash-preview",
            VLMProvider.CUSTOM_GOOGLE: "gemini-2.5-flash",
            VLMProvider.OLLAMA: "llava",
            VLMProvider.MOCK: "mock"
        }
        
        model_name = model_name or os.getenv("VLM_MODEL") or default_models[provider_enum]
        
        # Set fallback models from parameter, environment, or defaults
        if fallback_models:
            self.fallback_models = fallback_models
        elif os.getenv("VLM_FALLBACK_MODELS"):
            self.fallback_models = os.getenv("VLM_FALLBACK_MODELS").split(",")
        else:
            self.fallback_models = self.FALLBACK_MODELS.get(provider.lower(), [])
        
        # Create config
        self.config = VLMConfig(
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
        
        logger.info(f"VLM Interface initialized: provider={provider_enum.value}, model={model_name}")
        if self.fallback_models:
            logger.info(f"Fallback models configured: {self.fallback_models}")
    
    def _create_client(self) -> BaseVLMClient:
        """Create the appropriate VLM client based on provider."""
        clients = {
            VLMProvider.OPENAI: OpenAIClient,
            VLMProvider.ANTHROPIC: AnthropicClient,
            VLMProvider.GOOGLE: GoogleClient,
            VLMProvider.CUSTOM_GOOGLE: CustomGoogleClient,
            VLMProvider.OLLAMA: OllamaClient,
            VLMProvider.MOCK: MockClient
        }
        
        client_class = clients.get(self.config.provider, MockClient)
        return client_class(self.config)
    
    def call_vlm(
        self,
        prompt: str,
        image_paths: List[str],
        parse_json: bool = True,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Call the VLM with a prompt and images.
        
        Args:
            prompt: Text prompt for the VLM
            image_paths: List of paths to image files
            parse_json: Whether to parse the response as JSON
            **kwargs: Additional arguments passed to the client
        
        Returns:
            Response string or parsed JSON dict
        """
        # Filter valid images
        valid_images = [p for p in image_paths if os.path.exists(p)]
        
        if not valid_images:
            logger.warning("No valid images provided!")
            if len(image_paths) > 0:
                logger.warning(f"Missing images: {image_paths}")
        
        logger.info(f"Calling VLM ({self.config.model_name}) with {len(valid_images)} images...")
        
        last_error = None
        models_to_try = [self.config.model_name] + self.fallback_models
        print("Models to try:", models_to_try)
        
        for model_idx, current_model in enumerate(models_to_try):
            # Update model if using fallback
            if model_idx > 0:
                logger.info(f"Switching to fallback model: {current_model}")
                self.config.model_name = current_model
                self.client = self._create_client()

            for attempt in range(self.config.max_retries):
                try:
                    response = self.client.call(prompt, valid_images, **kwargs)
                    # Reset to original model on success for next call
                    if model_idx > 0:
                        logger.info(f"Fallback model {current_model} succeeded. Keeping for current session.")
                    if parse_json:
                        return self._extract_json(response)
                    return response
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # Catch inappropriate content errors and re-raise with explicit message
                    if (
                        "inappropriate content" in error_str.lower() or
                        "data_inspection_failed" in error_str.lower() or
                        ("Error code: 400" in error_str and "inappropriate content" in error_str)
                    ):
                        logger.error(f"VLM call failed due to inappropriate content: {error_str}")
                        raise RuntimeError(f"VLM inappropriate content: {error_str}") from e
                    # Check error types
                    is_overload = self._is_overload_error(e)
                    is_rate_limit = self._is_rate_limit_error(e)
                    is_connection = self._is_connection_error(e)
                    is_retryable = is_overload or is_rate_limit or is_connection
                    logger.warning(f"VLM call attempt {attempt + 1}/{self.config.max_retries} "
                                   f"(model: {current_model}) failed: {type(e).__name__}: {e}")
                    if attempt < self.config.max_retries - 1 and is_retryable:
                        # Calculate delay based on error type
                        if is_overload:
                            base_delay = self.config.retry_delay * 3
                            sleep_time = base_delay * (1.5 ** attempt) + random.uniform(1, 5)
                            logger.info(f"Server overloaded. Waiting {sleep_time:.1f} seconds before retry...")
                        elif is_rate_limit:
                            sleep_time = self._extract_retry_after(e) or (self.config.retry_delay * (2 ** attempt))
                            logger.info(f"Rate limited. Waiting {sleep_time:.1f} seconds...")
                        elif is_connection:
                            base_delay = self.config.retry_delay * 2
                            sleep_time = base_delay * (1.5 ** attempt) + random.uniform(0.5, 3)
                            logger.info(f"Connection error. Waiting {sleep_time:.1f} seconds before retry...")
                        else:
                            sleep_time = self.config.retry_delay * (2 ** attempt)
                            logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    elif attempt < self.config.max_retries - 1 and not is_retryable:
                        sleep_time = self.config.retry_delay
                        logger.info(f"Unexpected error. Retrying in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        if (is_overload or is_connection) and model_idx < len(models_to_try) - 1:
                            logger.info(f"Model {current_model} unavailable after {self.config.max_retries} attempts. "
                                        f"Trying fallback model...")
                            break  # Break inner loop to try next model
        # Restore original model for future calls
        self.config.model_name = self._original_model
        self.client = self._create_client()
        raise RuntimeError(f"VLM call failed after trying all models ({models_to_try}). Last error: {last_error}")
    
    def _is_overload_error(self, error: Exception) -> bool:
        """Check if error indicates server overload or connection issues."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check for common overload indicators
        overload_keywords = [
            'overloaded', 'engine_overloaded', 'no channel available',
            'service unavailable', 'capacity', 'temporarily unavailable',
            'server busy', 'too many requests'
        ]
        
        for keyword in overload_keywords:
            if keyword in error_str:
                return True
        
        # Check for HTTP status codes
        for code in self.OVERLOAD_ERROR_CODES:
            if str(code) in error_str:
                return True
        
        return False
    
    def _is_connection_error(self, error: Exception) -> bool:
        """Check if error is a connection/network error that should be retried."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Connection error keywords
        connection_keywords = [
            'connection', 'timeout', 'timed out', 'reset', 'refused',
            'network', 'unreachable', 'eof', 'broken pipe', 'ssl',
            'certificate', 'handshake', 'socket', 'dns', 'resolve',
            'connecttimeout', 'readtimeout', 'connectionerror', 'remotedisconnected'
        ]
        
        for keyword in connection_keywords:
            if keyword in error_str or keyword in error_type:
                return True
        
        # Check for specific exception types
        connection_exception_types = [
            'connectionerror', 'timeout', 'timeouterror', 'connecttimeout',
            'readtimeout', 'sslerror', 'httperror', 'remotedisconnected',
            'connectionreseterror', 'apiconnectionerror'
        ]
        
        for exc_type in connection_exception_types:
            if exc_type in error_type:
                return True
        
        return False
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error indicates rate limiting."""
        error_str = str(error).lower()
        
        rate_limit_keywords = ['rate limit', 'rate_limit', 'too many requests', 'quota exceeded']
        
        for keyword in rate_limit_keywords:
            if keyword in error_str:
                return True
        
        for code in self.RATE_LIMIT_ERROR_CODES:
            if str(code) in error_str:
                return True
        
        return False
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Try to extract retry-after value from error."""
        error_str = str(error)
        
        # Try to find retry-after in various formats
        import re
        patterns = [
            r'retry.?after[:\s]+(\d+)',
            r'wait[:\s]+(\d+)',
            r'(\d+)\s*seconds?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _extract_json(self, response: str) -> Union[str, Dict[str, Any]]:
        """Extract JSON from VLM response, handling markdown code blocks."""
        # Try to find JSON in code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        
        # Try parsing the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Could not parse response as JSON, returning raw string")
            return response
    
    def call_vlm_with_labeled_images(
        self,
        prompt: str,
        image_groups: Dict[str, List[str]],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Call VLM with labeled groups of images (useful for A/B comparisons).
        
        Args:
            prompt: Text prompt (should reference labels like "A_slide_1", "B_slide_2")
            image_groups: Dict mapping labels to image path lists
                         e.g., {"A": ["a1.jpg", "a2.jpg"], "B": ["b1.jpg", "b2.jpg"]}
            **kwargs: Additional arguments
        
        Returns:
            Response from VLM
        """
        # Flatten images while maintaining label context
        # Note: Most VLMs don't support explicit image labeling, so we rely on
        # the prompt to describe the image order
        all_images = []
        image_info = []
        
        for label, paths in image_groups.items():
            for i, path in enumerate(paths):
                all_images.append(path)
                image_info.append(f"{label}_slide_{i+1}")
        
        # Add image ordering info to prompt if not already present
        if "Image order:" not in prompt:
            order_info = "\n**Image Order:** " + ", ".join(image_info)
            prompt = prompt + order_info
        # print(prompt)
        # print(all_images)
        # input()
        return self.call_vlm(prompt, all_images, **kwargs)
    
    def call_vlm_with_concatenated_images(
        self,
        prompt: str,
        image_groups: Dict[str, List[str]],
        slides_per_grid: int = CONCAT_SLIDES_PER_IMAGE,
        cols: int = CONCAT_GRID_COLS,
        rows: int = CONCAT_GRID_ROWS,
        save_grids_dir: Optional[str] = None,
        identifier: str = "",
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Call VLM with concatenated grid images for efficient comparison.
        
        Instead of sending individual slide images, this method concatenates
        multiple slides into grid images (e.g., 2x4 = 8 slides per image).
        This allows including more slides while reducing the number of images
        sent to the VLM.
        
        Args:
            prompt: Text prompt
            image_groups: Dict mapping labels to image path lists
                         e.g., {"A": ["a1.jpg", ...], "B": ["b1.jpg", ...]}
            slides_per_grid: Number of slides per grid image
            cols: Number of columns in each grid
            rows: Number of rows in each grid
            save_grids_dir: Optional directory to save grid images for inspection
            identifier: Identifier for saved grid files (e.g., topic name)
            **kwargs: Additional arguments passed to call_vlm
        
        Returns:
            Response from VLM
        """
        all_grid_images = []
        grid_info = []
        
        for label, paths in image_groups.items():
            if not paths:
                continue
            
            # Create grid images for this group
            num_slides = len(paths)
            num_grids = (num_slides + slides_per_grid - 1) // slides_per_grid
            
            for grid_idx in range(num_grids):
                start_idx = grid_idx * slides_per_grid
                end_idx = min(start_idx + slides_per_grid, num_slides)
                grid_slides = paths[start_idx:end_idx]
                
                # Determine save path
                save_path = None
                if save_grids_dir:
                    os.makedirs(save_grids_dir, exist_ok=True)
                    safe_identifier = identifier.replace("/", "_").replace(" ", "_")
                    save_path = os.path.join(
                        save_grids_dir,
                        f"{safe_identifier}_{label}_grid{grid_idx + 1}_slides{start_idx + 1}-{end_idx}.jpg"
                    )
                
                # Create the grid image with correct start_index for labels
                # start_idx is 0-based, but slide labels should be 1-based
                grid_image = concatenate_images_to_grid(
                    grid_slides,
                    cols=cols,
                    rows=rows,
                    add_labels=True,
                    label_prefix=f"{label}_",
                    start_index=start_idx + 1,  # Convert to 1-based slide numbering
                    save_path=save_path
                )
                
                if grid_image is None:
                    logger.warning(f"Failed to create grid for {label} slides {start_idx + 1}-{end_idx}")
                    continue
                
                # Encode to base64 and store
                buffer = io.BytesIO()
                grid_image.save(buffer, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                buffer.seek(0)
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                all_grid_images.append(base64_image)
                
                # Build info string for this grid - use correct 1-based slide numbers
                slides_in_grid = [f"{label}_slide_{i}" for i in range(start_idx + 1, end_idx + 1)]
                grid_label = f"{label}_grid_{grid_idx + 1} (contains: {', '.join(slides_in_grid)})"
                grid_info.append(grid_label)
                
                if save_path:
                    logger.info(f"Saved grid image: {save_path}")
        
        if not all_grid_images:
            logger.error("No grid images created")
            return {"error": "Failed to create grid images"}
        
        # Update prompt with grid information
        grid_order_info = "\n\n**Image Layout:** Each image is a grid containing multiple slides. " \
                         f"Grid layout: {rows} rows x {cols} columns.\n" \
                         f"**Grid Images:** " + ", ".join(grid_info)
        
        # Check if prompt already has image order info, replace it
        if "**Image Order:**" in prompt:
            prompt = prompt.split("**Image Order:**")[0].strip()
        
        prompt = prompt + grid_order_info
        
        # Build content for VLM call with pre-encoded base64 images
        content = [{"type": "text", "text": prompt}]
        
        for base64_image in all_grid_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": kwargs.get("detail", "high")  # Use high detail for grids
                }
            })
        
        # Call VLM directly with pre-built content
        parse_json = kwargs.get("parse_json", False)
        
        try:
            response = self.client.call_with_content(content, **kwargs)
            
            if parse_json:
                return self._extract_json(response)
            return response
            
        except AttributeError:
            # Fallback: client doesn't have call_with_content, use standard call
            # This requires saving grid images temporarily
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_paths = []
            
            try:
                for i, base64_image in enumerate(all_grid_images):
                    temp_path = os.path.join(temp_dir, f"grid_{i}.jpg")
                    with open(temp_path, 'wb') as f:
                        f.write(base64.b64decode(base64_image))
                    temp_paths.append(temp_path)
                
                return self.call_vlm(prompt, temp_paths, **kwargs)
                
            finally:
                # Cleanup temp files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self.config.model_name
    
    @property
    def provider(self) -> str:
        """Get current provider name."""
        return self.config.provider.value


# Convenience functions
def create_vlm(provider: str = "openai", **kwargs) -> VLMInterface:
    """Factory function to create VLM interface."""
    return VLMInterface(provider=provider, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available VLM providers."""
    return [p.value for p in VLMProvider]
