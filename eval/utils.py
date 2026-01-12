"""
Utility functions for PPT Evaluation System.

Common functionality shared across evaluation modules.

Author: PPT Evaluation System
"""

import os
import json
import logging
from typing import Dict, List, Optional, Callable, Set

from eval_config import (
    EvalConfig,
    get_all_products,
    get_product_path,
    get_product_config
)

logger = logging.getLogger(__name__)


def discover_ppts(
    ppt_gen_root: str,
    products: List[str] = None,
    difficulties: List[str] = None,
    topics: List[str] = None,
    limit: int = None,
    group_by_topic: bool = False,
    require_multiple_products: bool = False,
    cache_check_fn: Callable[[str, str, str, str], bool] = None,
    quiz_json_path: str = None
) -> Dict[str, List[Dict]]:
    """
    Discover PPTs in the generation directory.
    
    Handles different product directory structures:
    - Standard: {product}/difficulty/topic/id
    - Kimi: Kimi/{Standard,Smart,Banana}/difficulty/topic/name
    - NotebookLM: NotebookLM/difficulty/topic (no subdirectory)
    
    Args:
        ppt_gen_root: Root directory for generated PPTs
        products: List of products to include (None for all)
        difficulties: List of difficulties to include (None for all, e.g. ["low", "medium", "high"])
        topics: List of topics to include (None for all)
        limit: Maximum number of PPTs to discover **per product** (None for no limit)
        group_by_topic: If True, return dict grouped by topic; if False, return flat list in 'all' key
        require_multiple_products: If True and group_by_topic, only return topics with 2+ products
        cache_check_fn: Optional function(product, difficulty, topic, ppt_id) -> bool 
                       to check if PPT is cached. If provided, results are split into 
                       'to_evaluate' and 'cached' keys.
        quiz_json_path: Optional path to quiz JSON file for filtering valid PPTs
    
    Returns:
        Dict with discovered PPTs. Format depends on parameters:
        - If group_by_topic=True: {topic: [ppt_info, ...], ...}
        - If group_by_topic=False: {'to_evaluate': [...], 'cached': [...]}
          (cached list is empty if cache_check_fn is None)
        
        Each ppt_info dict contains:
        {
            'product': str,
            'difficulty': str, 
            'topic': str,
            'ppt_id': str,  # or 'id' for group_by_topic mode
            'ppt_path': str  # or 'path' for group_by_topic mode
        }
    """
    if not os.path.exists(ppt_gen_root):
        logger.error(f"PPT generation root not found: {ppt_gen_root}")
        if group_by_topic:
            return {}
        elif cache_check_fn:
            return {'to_evaluate': [], 'cached': []}
        else:
            return {'all': []}
    
    # Load valid PPT set from quiz JSON if provided (load once at the beginning)
    valid_ppt_set: Optional[Set[str]] = None
    if quiz_json_path and os.path.exists(quiz_json_path):
        logger.info(f"Loading valid PPT set from: {quiz_json_path}")
        try:
            with open(quiz_json_path, "r", encoding="utf-8") as f:
                quiz_data = json.load(f)
            valid_ppt_set = set()
            for doc in quiz_data.get("documents", []):
                meta = doc.get("meta_info", {})
                # Use folder name from source_path as the primary identifier
                # source_path format: ./domain/folder_name/filename.pdf
                source_path = meta.get("source_path", "")
                if source_path:
                    parts = source_path.split("/")
                    if len(parts) >= 3:
                        folder_name = parts[2]  # The folder containing the PDF
                        valid_ppt_set.add(folder_name)
                # Also add filename (without extension) as fallback for compatibility
                if "filename" in meta:
                    name = meta["filename"].rsplit(".", 1)[0]
                    valid_ppt_set.add(name)
            logger.info(f"Loaded {len(valid_ppt_set)} valid PPT names from quiz JSON")
        except Exception as e:
            logger.warning(f"Failed to load quiz JSON: {e}. Proceeding without filtering.")
            valid_ppt_set = None
    
    # Use configured products
    all_products = products if products else get_all_products()
    
    # Debug logging
    if products:
        logger.info(f"Filtering to products: {products}")
    else:
        logger.info(f"Using all available products: {all_products}")
    
    # Storage based on mode (initialize all variables regardless of mode)
    ppts_by_topic = {}
    to_evaluate = []
    cached = []
    
    total_count = 0
    
    for product in all_products:
        product_config = get_product_config(product)
        product_path = get_product_path(product, ppt_gen_root)
        
        if not os.path.exists(product_path):
            logger.warning(f"Product directory not found: {product_path}")
            continue
        
        logger.info(f"Discovering PPTs for product: {product}")
        
        # Reset count for each product
        product_count = 0
        
        # Get difficulties
        all_diffs = [d for d in os.listdir(product_path) 
                    if os.path.isdir(os.path.join(product_path, d)) 
                    and not d.startswith('.')]
        
        # # Skip old difficulty names that have been converted to 'topic_introduction'
        # skip_difficulties = {'high', 'low', 'medium', 'lite'}
        # all_diffs = [d for d in all_diffs if d not in skip_difficulties]
        
        if difficulties:
            all_diffs = [d for d in all_diffs if d in difficulties]
        
        for diff in all_diffs:
            diff_path = os.path.join(product_path, diff)
            if not os.path.exists(diff_path):
                logger.warning(f"  Difficulty directory not found: {diff_path}")
                continue
            
            # Get topics
            all_topics = [d for d in os.listdir(diff_path) 
                         if os.path.isdir(os.path.join(diff_path, d)) 
                         and not d.startswith('.')]
            logger.debug(f"  Difficulty '{diff}': found {len(all_topics)} topics")
            
            if topics:
                all_topics = [t for t in all_topics if t in topics]
            
            for topic in all_topics:
                topic_path = os.path.join(diff_path, topic)
                
                # Check limit for this product
                if limit and product_count >= limit:
                    break
                
                # Case 1: Files directly in topic folder (e.g., NotebookLM)
                images_dir = os.path.join(topic_path, "slide_images")
                if os.path.exists(images_dir):
                    ppt_id = topic
                    
                    # Apply filtering if valid_ppt_set is provided
                    if valid_ppt_set is not None and ppt_id not in valid_ppt_set:
                        logger.debug(f"  Skipping PPT not in valid set: {ppt_id}")
                        continue
                    
                    ppt_info = _create_ppt_info(
                        product, diff, topic, ppt_id, topic_path, group_by_topic
                    )
                    
                    _add_ppt_to_results(
                        ppt_info, group_by_topic, ppts_by_topic, 
                        to_evaluate, cached, cache_check_fn, topic,
                        product, diff, ppt_id
                    )
                    
                    product_count += 1
                    total_count += 1
                    continue
                
                # Case 2: Handle products with subdirectories (Standard structure)
                ppt_entries = [d for d in os.listdir(topic_path)
                              if os.path.isdir(os.path.join(topic_path, d))
                              and not d.startswith('.')
                              and os.path.exists(os.path.join(topic_path, d, "slide_images"))]
                
                if not ppt_entries:
                    logger.debug(f"  No PPT entries found in topic: {topic_path}")
                    continue
                
                for ppt_id in ppt_entries:
                    # Check limit for this product
                    if limit and product_count >= limit:
                        break
                    
                    ppt_path = os.path.join(topic_path, ppt_id)
                    
                    # Apply filtering if valid_ppt_set is provided
                    if valid_ppt_set is not None and ppt_id not in valid_ppt_set:
                        logger.debug(f"  Skipping PPT not in valid set: {ppt_id}")
                        continue
                    
                    ppt_info = _create_ppt_info(
                        product, diff, topic, ppt_id, ppt_path, group_by_topic
                    )
                    
                    _add_ppt_to_results(
                        ppt_info, group_by_topic, ppts_by_topic,
                        to_evaluate, cached, cache_check_fn, topic,
                        product, diff, ppt_id
                    )
                    
                    product_count += 1
                    total_count += 1
                
                # Break out of topic loop if limit reached
                if limit and product_count >= limit:
                    break
            
            # Break out of difficulty loop if limit reached
            if limit and product_count >= limit:
                break
        
        if limit and product_count >= limit:
            logger.info(f"  Reached limit of {limit} PPTs for product {product}")
    
    # Return based on mode
    if group_by_topic:
        # Log discovered PPTs
        for topic, ppts in ppts_by_topic.items():
            products_in_topic = set(p["product"] for p in ppts)
            logger.info(f"  Topic '{topic}': {len(ppts)} PPTs from {products_in_topic}")
        
        # Filter to topics with multiple products if requested
        if require_multiple_products:
            filtered = {}
            for topic, ppts in ppts_by_topic.items():
                products_in_topic = set(p["product"] for p in ppts)
                if len(products_in_topic) >= 2:
                    filtered[topic] = ppts
            logger.info(f"Found {len(filtered)} topics with PPTs from multiple products")
            return filtered
        
        return ppts_by_topic
    else:
        # Always return 'to_evaluate' and 'cached' format for consistency
        # When no cache_check_fn, all items go to 'to_evaluate'
        print(f"Total PPTs to evaluate: {len(to_evaluate)}, cached: {len(cached)}")
        return {'to_evaluate': to_evaluate, 'cached': cached}


def _add_ppt_to_results(
    ppt_info: Dict,
    group_by_topic: bool,
    ppts_by_topic: Dict,
    to_evaluate: List,
    cached: List,
    cache_check_fn: Optional[Callable],
    topic: str,
    product: str,
    difficulty: str,
    ppt_id: str
) -> None:
    """
    Helper function to add PPT info to the appropriate result collection.
    
    Args:
        ppt_info: PPT metadata dictionary
        group_by_topic: Whether grouping by topic
        ppts_by_topic: Dict for topic-grouped results
        to_evaluate: List for PPTs to evaluate
        cached: List for cached PPTs
        cache_check_fn: Optional cache checking function
        topic: Topic name
        product: Product name
        difficulty: Difficulty level
        ppt_id: PPT identifier
    """
    if group_by_topic:
        if topic not in ppts_by_topic:
            ppts_by_topic[topic] = []
        ppts_by_topic[topic].append(ppt_info)
    else:
        if cache_check_fn and cache_check_fn(product, difficulty, topic, ppt_id):
            cached.append(ppt_info)
        else:
            to_evaluate.append(ppt_info)


def _create_ppt_info(
    product: str,
    difficulty: str,
    topic: str,
    ppt_id: str,
    ppt_path: str,
    use_arena_format: bool = False
) -> Dict:
    """
    Create a PPT info dictionary.
    
    Args:
        product: Product name
        difficulty: Difficulty level
        topic: Topic name
        ppt_id: PPT identifier
        ppt_path: Full path to PPT directory
        use_arena_format: If True, use 'id' and 'path' keys (arena format);
                         if False, use 'ppt_id' and 'ppt_path' keys (quantitative format)
    
    Returns:
        Dict with PPT metadata
    """
    if use_arena_format:
        return {
            'product': product,
            'difficulty': difficulty,
            'topic': topic,
            'id': ppt_id,
            'path': ppt_path
        }
    else:
        return {
            'product': product,
            'difficulty': difficulty,
            'topic': topic,
            'ppt_id': ppt_id,
            'ppt_path': ppt_path
        }
