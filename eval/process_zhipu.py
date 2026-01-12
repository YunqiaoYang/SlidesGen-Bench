#!/usr/bin/env python3
"""
Zhipu (ChatGLM) PPT URL Processor

This script extracts slide images from Zhipu/ChatGLM share URLs.
It uses Playwright to navigate the share page and capture each slide image.

Usage:
    python process_zhipu.py --url <url> --output <output_dir>
    python process_zhipu.py --product-dir <zhipu_product_dir> [--force]
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a PPT."""
    source_path: str
    output_dir: str
    product: str
    difficulty: str = ""
    topic: str = ""
    ppt_id: str = ""
    num_slides: int = 0
    image_paths: List[str] = field(default_factory=list)
    text_paths: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None


class ZhipuURLProcessor:
    """
    Process Zhipu (ChatGLM) share URLs to extract PPT slides.
    
    ChatGLM share pages render PPT slides as images in the conversation.
    """
    
    def __init__(self, image_format: str = "png"):
        self.image_format = image_format
    
    def _extract_url(self, text: str) -> str:
        """
        Extract actual URL from text that may contain description.
        
        Handles formats like:
        - "Generate a PowerPoi... →点击查看智谱清言的回答https://chatglm.cn/share/MF0qZXU8"
        - "https://chatglm.cn/share/MF0qZXU8"
        
        Returns:
            Cleaned URL string
        """
        import re
        
        # Try to find URL pattern in the text
        url_pattern = r'https?://[^\s\u4e00-\u9fff]+'
        match = re.search(url_pattern, text)
        
        if match:
            url = match.group(0)
            # Clean up any trailing punctuation or special characters
            url = url.rstrip('.,;:!?）】》')
            return url
        
        # If no URL pattern found, return original text (might already be clean URL)
        return text.strip()
    
    def process_url(
        self,
        url: str,
        output_dir: str,
        force_reprocess: bool = False
    ) -> ProcessingResult:
        """
        Process a Zhipu ChatGLM share URL.
        
        Args:
            url: ChatGLM share URL (can be raw text or actual URL)
            output_dir: Directory to save slide images
            force_reprocess: Force reprocess even if images exist
            
        Returns:
            ProcessingResult with paths and status
        """
        
        # Extract actual URL if text contains description
        # Format: "Generate a PowerPoi... →点击查看智谱清言的回答https://chatglm.cn/share/MF0qZXU8"
        url = self._extract_url(url)
        
        logger.info(f"Processing Zhipu URL: {url}")
        
        images_dir = os.path.join(output_dir, "slide_images")
        texts_dir = os.path.join(output_dir, "slide_texts")
        
        # Check if already processed
        if os.path.exists(images_dir) and not force_reprocess:
            existing_images = sorted([
                os.path.join(images_dir, f)
                for f in os.listdir(images_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if existing_images:
                logger.info(f"Already processed: {output_dir} ({len(existing_images)} images)")
                return ProcessingResult(
                    source_path=url,
                    output_dir=output_dir,
                    product="Zhipu",
                    num_slides=len(existing_images),
                    image_paths=existing_images,
                    success=True
                )
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(texts_dir, exist_ok=True)
        
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                
                # Navigate to URL
                logger.info(f"Navigating to {url}")
                page.goto(url, wait_until='networkidle', timeout=60000)
                time.sleep(5)  # Wait for dynamic content to fully load
                
                # Extract slides
                image_paths, text_paths = self._extract_slides(page, images_dir, texts_dir)
                
                browser.close()
                
                if image_paths:
                    logger.info(f"Extracted {len(image_paths)} slides from Zhipu URL")
                    return ProcessingResult(
                        source_path=url,
                        output_dir=output_dir,
                        product="Zhipu",
                        num_slides=len(image_paths),
                        image_paths=image_paths,
                        text_paths=text_paths,
                        success=True
                    )
                else:
                    return ProcessingResult(
                        source_path=url,
                        output_dir=output_dir,
                        product="Zhipu",
                        success=False,
                        error="No slides extracted from URL"
                    )
                
        except ImportError:
            logger.error("playwright not installed. Install with: pip install playwright && playwright install chromium")
            return ProcessingResult(
                source_path=url,
                output_dir=output_dir,
                product="Zhipu",
                success=False,
                error="playwright not installed"
            )
        except Exception as e:
            logger.error(f"Error processing Zhipu URL: {e}")
            import traceback
            traceback.print_exc()
            return ProcessingResult(
                source_path=url,
                output_dir=output_dir,
                product="Zhipu",
                success=False,
                error=str(e)
            )
    
    def _extract_slides(self, page, images_dir: str, texts_dir: str) -> tuple:
        """
        Extract slides from ChatGLM share page.
        
        Interaction flow:
        1. Click the play button (SVG with path containing slide play icon) to enter slides mode
        2. Get the page index from element like "1 / 18"
        3. Screenshot each slide and use keyboard "right" to navigate to next
        """
        image_paths = []
        text_paths = []
        
        try:
            # Wait for the page to fully load
            time.sleep(3)
            
            # Step 1: Click the play button to enter slides mode
            # The button has an SVG with a specific path for the play icon
            play_button_selectors = [
                # Look for the action icon container that has the play SVG
                "svg.action-icon",
                "[class*='action-icon']",
                "svg[viewBox='0 0 160 160']",
                # Also try parent elements
                "button:has(svg.action-icon)",
                "[class*='action']:has(svg)",
                # Generic play button patterns
                "[class*='play']",
                "[aria-label*='play']",
                "[title*='播放']",
                "[title*='演示']",
            ]
            
            play_button_clicked = False
            for selector in play_button_selectors:
                try:
                    buttons = page.locator(selector).all()
                    for btn in buttons:
                        if btn.is_visible():
                            # Check if this looks like the play button
                            try:
                                # Try to click it
                                logger.info(f"Trying to click play button with selector: {selector}")
                                btn.click()
                                time.sleep(2)  # Wait for slides mode to activate
                                
                                # Check if we're now in slides mode by looking for page index
                                page_index = page.locator(".page-index, [class*='page-index'], [class*='pageIndex']").first
                                if page_index.is_visible():
                                    play_button_clicked = True
                                    logger.info("Successfully entered slides mode!")
                                    break
                            except Exception as e:
                                logger.debug(f"Click failed: {e}")
                                continue
                    
                    if play_button_clicked:
                        break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not play_button_clicked:
                logger.warning("Could not find or click play button, trying alternative methods...")
                # Try clicking on any SVG that might be the play button
                svgs = page.locator("svg").all()
                for svg in svgs:
                    try:
                        if svg.is_visible():
                            box = svg.bounding_box()
                            if box and 15 < box['width'] < 50 and 15 < box['height'] < 50:
                                svg.click()
                                time.sleep(1)
                                # Check for slides mode
                                page_index = page.locator(".page-index, [class*='page-index']").first
                                if page_index.is_visible():
                                    play_button_clicked = True
                                    logger.info("Entered slides mode via SVG click!")
                                    break
                    except:
                        continue
            
            # Step 2: Get the page index to know total number of slides
            total_slides = 1
            try:
                page_index_elem = page.locator(".page-index, [class*='page-index'], [class*='pageIndex'], p:has-text('/')").first
                if page_index_elem.is_visible():
                    index_text = page_index_elem.inner_text()
                    logger.info(f"Page index text: {index_text}")
                    # Parse "1 / 18" format
                    if '/' in index_text:
                        parts = index_text.split('/')
                        total_slides = int(parts[1].strip())
                        logger.info(f"Total slides: {total_slides}")
            except Exception as e:
                logger.warning(f"Could not get page index: {e}")
                # Try alternative: look for any text matching pattern
                try:
                    all_text = page.inner_text("body")
                    import re
                    match = re.search(r'(\d+)\s*/\s*(\d+)', all_text)
                    if match:
                        total_slides = int(match.group(2))
                        logger.info(f"Found total slides from text: {total_slides}")
                except:
                    pass
            
            # Step 3: Take screenshots and navigate through slides
            logger.info(f"Starting to capture {total_slides} slides...")
            
            for slide_num in range(1, total_slides + 1):
                try:
                    # Wait for slide to render
                    time.sleep(0.5)
                    
                    # Take screenshot of the current slide
                    # Try to find the slide container first
                    slide_container_selectors = [
                        "[class*='slide-container']",
                        "[class*='ppt-container']",
                        "[class*='presentation']",
                        "[class*='fullscreen']",
                        "[class*='viewer']",
                        ".slide",
                    ]
                    
                    screenshot_taken = False
                    for container_selector in slide_container_selectors:
                        try:
                            container = page.locator(container_selector).first
                            if container.is_visible():
                                box = container.bounding_box()
                                if box and box['width'] > 500 and box['height'] > 300:
                                    img_path = os.path.join(images_dir, f"slide_{slide_num:04d}.{self.image_format}")
                                    container.screenshot(path=img_path)
                                    image_paths.append(img_path)
                                    screenshot_taken = True
                                    logger.info(f"Captured slide {slide_num}/{total_slides} using container {container_selector}")
                                    break
                        except:
                            continue
                    
                    # Fallback: full page screenshot
                    if not screenshot_taken:
                        img_path = os.path.join(images_dir, f"slide_{slide_num:04d}.{self.image_format}")
                        page.screenshot(path=img_path)
                        image_paths.append(img_path)
                        logger.info(f"Captured slide {slide_num}/{total_slides} (full page)")
                    
                    # Navigate to next slide using keyboard right arrow
                    if slide_num < total_slides:
                        page.keyboard.press("ArrowRight")
                        time.sleep(0.8)  # Wait for transition
                        
                except Exception as e:
                    logger.warning(f"Error capturing slide {slide_num}: {e}")
                    continue
            
            # If we didn't enter slides mode and got no slides, try fallback methods
            if not image_paths:
                logger.warning("No slides captured in slides mode, trying fallback...")
                return self._extract_slides_fallback(page, images_dir, texts_dir)
            
        except Exception as e:
            logger.error(f"Error extracting slides: {e}")
            import traceback
            traceback.print_exc()
        
        return image_paths, text_paths
    
    def _extract_slides_fallback(self, page, images_dir: str, texts_dir: str) -> tuple:
        """
        Fallback method to extract slides by finding images directly.
        """
        image_paths = []
        text_paths = []
        
        try:
            # Find all large images that could be slides
            all_images = page.locator("img").all()
            logger.info(f"Fallback: Found {len(all_images)} total images on page")
            
            slide_images = []
            for img in all_images:
                try:
                    if img.is_visible():
                        box = img.bounding_box()
                        if box and box['width'] > 400 and box['height'] > 200:
                            src = img.get_attribute('src') or ''
                            if 'avatar' not in src.lower() and 'icon' not in src.lower():
                                slide_images.append({
                                    'element': img,
                                    'y': box['y'],
                                    'width': box['width'],
                                    'height': box['height']
                                })
                except:
                    continue
            
            # Sort by vertical position
            slide_images.sort(key=lambda x: x['y'])
            logger.info(f"Fallback: Found {len(slide_images)} potential slide images")
            
            for i, slide_info in enumerate(slide_images):
                try:
                    img = slide_info['element']
                    img.scroll_into_view_if_needed()
                    time.sleep(0.5)
                    
                    img_path = os.path.join(images_dir, f"slide_{i+1:04d}.{self.image_format}")
                    img.screenshot(path=img_path)
                    image_paths.append(img_path)
                    logger.info(f"Fallback: Captured slide {i+1}")
                except Exception as e:
                    logger.warning(f"Fallback: Error capturing slide {i+1}: {e}")
            
            # If still no images, take full page screenshot
            if not image_paths:
                logger.warning("Fallback: Taking full page screenshot")
                img_path = os.path.join(images_dir, "slide_0001.png")
                page.screenshot(path=img_path, full_page=True)
                image_paths.append(img_path)
                
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
        
        return image_paths, text_paths


def process_product_directory(
    product_dir: str,
    force_reprocess: bool = False
) -> List[ProcessingResult]:
    """
    Process all Zhipu URLs in a product directory structure.
    
    Expected structure:
    product_dir/
        low/
            topic1/
                url.txt
            topic2/
                url.txt
        medium/
            ...
        high/
            ...
    """
    results = []
    processor = ZhipuURLProcessor()
    
    if not os.path.exists(product_dir):
        logger.error(f"Product directory not found: {product_dir}")
        return results
    
    logger.info(f"Processing Zhipu product directory: {product_dir}")
    
    # Iterate through difficulties
    for difficulty in ['low', 'medium', 'high']:
        diff_path = os.path.join(product_dir, difficulty)
        if not os.path.exists(diff_path):
            continue
        
        # Iterate through topics
        for topic in os.listdir(diff_path):
            topic_path = os.path.join(diff_path, topic)
            if not os.path.isdir(topic_path) or topic.startswith('.'):
                continue
            
            # Look for url.txt
            url_file = os.path.join(topic_path, "url.txt")
            if os.path.exists(url_file):
                try:
                    with open(url_file, 'r', encoding='utf-8') as f:
                        url = f.read().strip()
                    
                    if url:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"Processing: {difficulty}/{topic}")
                        logger.info(f"URL: {url}")
                        logger.info(f"{'='*60}")
                        
                        result = processor.process_url(url, topic_path, force_reprocess)
                        result.difficulty = difficulty
                        result.topic = topic
                        results.append(result)
                        
                        if result.success:
                            logger.info(f"SUCCESS: {result.num_slides} slides extracted")
                        else:
                            logger.error(f"FAILED: {result.error}")
                            
                except Exception as e:
                    logger.error(f"Error processing {topic_path}: {e}")
            else:
                # Check for subdirectories that might contain url.txt
                for item in os.listdir(topic_path):
                    item_path = os.path.join(topic_path, item)
                    if os.path.isdir(item_path):
                        sub_url_file = os.path.join(item_path, "url.txt")
                        if os.path.exists(sub_url_file):
                            try:
                                with open(sub_url_file, 'r', encoding='utf-8') as f:
                                    url = f.read().strip()
                                
                                if url:
                                    logger.info(f"\n{'='*60}")
                                    logger.info(f"Processing: {difficulty}/{topic}/{item}")
                                    logger.info(f"URL: {url}")
                                    logger.info(f"{'='*60}")
                                    
                                    result = processor.process_url(url, item_path, force_reprocess)
                                    result.difficulty = difficulty
                                    result.topic = topic
                                    result.ppt_id = item
                                    results.append(result)
                                    
                                    if result.success:
                                        logger.info(f"SUCCESS: {result.num_slides} slides extracted")
                                    else:
                                        logger.error(f"FAILED: {result.error}")
                                        
                            except Exception as e:
                                logger.error(f"Error processing {item_path}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process Zhipu (ChatGLM) PPT URLs to extract slide images"
    )
    
    parser.add_argument(
        "--url",
        help="Single ChatGLM share URL to process"
    )
    parser.add_argument(
        "--output",
        help="Output directory for single URL processing"
    )
    parser.add_argument(
        "--product-dir",
        required=True,
        help="Zhipu product directory to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess even if slide_images exists"
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.url:
        # Process single URL
        output_dir = args.output or "./zhipu_output"
        processor = ZhipuURLProcessor()
        result = processor.process_url(args.url, output_dir, args.force)
        results.append(result)
    else:
        # Process product directory
        results = process_product_directory(args.product_dir, args.force)
    
    # Print summary
    print("\n" + "="*60)
    print("ZHIPU PROCESSING COMPLETE")
    print("="*60)
    
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)
    total_slides = sum(r.num_slides for r in results if r.success)
    
    print(f"Total: {len(results)} items")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total slides: {total_slides}")
    
    if failed_count > 0:
        print("\nFailed items:")
        for r in results:
            if not r.success:
                print(f"  - {r.source_path}: {r.error}")
    
    print("="*60)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
