"""
Arena Evaluation Module for PPT Head-to-Head Comparison

This module performs pairwise comparisons between presentations from different
products/tools, using an ELO-based ranking system to determine overall quality.

Features:
- Head-to-head VLM-based comparisons
- ELO rating system for ranking
- Supports multiple comparison strategies
- Detailed per-criterion comparison results

Author: PPT Evaluation System
"""

import os
import json
import glob
import logging
import argparse
import itertools
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from tqdm import tqdm

from eval_config import (
    ARENA_COMPARISON_PROMPT_TEMPLATE,
    EvalConfig,
    get_all_products,
    get_product_path,
    get_product_config,
    PRODUCTS
)
from utils import discover_ppts

# Max images per PPT in arena comparison (half of VLM call limit for A/B comparison)
MAX_IMAGES_PER_PPT_ARENA = 1

from vlm_interface import VLMInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Data class for storing a single match result."""
    topic: str
    ppt_a_product: str
    ppt_a_id: str
    ppt_a_path: str
    ppt_b_product: str
    ppt_b_id: str
    ppt_b_path: str
    
    # Results by criterion (Visual Design and Layout only)
    visual_winner: str = ""
    visual_reason: str = ""
    visual_score_diff: int = 0
    
    layout_winner: str = ""
    layout_reason: str = ""
    layout_score_diff: int = 0
    
    # Overall result
    overall_winner: str = ""
    overall_reason: str = ""
    confidence: int = 0
    key_differences: List[str] = field(default_factory=list)
    
    # Metadata
    evaluation_time: str = ""
    vlm_model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ELOSystem:
    """
    ELO Rating System for PPT comparisons.
    
    Based on the chess ELO system, adapted for pairwise comparisons.
    """
    
    def __init__(
        self,
        initial_rating: float = 1000,
        k_factor: float = 32,
        min_rating: float = 100,
        max_rating: float = 3000
    ):
        """
        Initialize ELO system.
        
        Args:
            initial_rating: Starting rating for new players
            k_factor: K-factor for rating adjustments
            min_rating: Minimum allowed rating
            max_rating: Maximum allowed rating
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.ratings: Dict[str, float] = {}
        self.match_history: List[Dict] = []
    
    def get_rating(self, player: str) -> float:
        """Get rating for a player, initializing if needed."""
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
        return self.ratings[player]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(
        self,
        player_a: str,
        player_b: str,
        result: str,
        confidence: int = 3
    ) -> Tuple[float, float]:
        """
        Update ratings based on match result.
        
        Args:
            player_a: Name of player A
            player_b: Name of player B
            result: "A", "B", or "Tie"
            confidence: Match confidence (1-5), affects K-factor
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        
        # Actual scores
        if result == "A":
            actual_a, actual_b = 1.0, 0.0
        elif result == "B":
            actual_a, actual_b = 0.0, 1.0
        else:  # Tie
            actual_a, actual_b = 0.5, 0.5
        
        # Expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Adjust K-factor based on confidence
        adjusted_k = self.k_factor * (confidence / 3)
        
        # Calculate new ratings
        new_rating_a = rating_a + adjusted_k * (actual_a - expected_a)
        new_rating_b = rating_b + adjusted_k * (actual_b - expected_b)
        
        # Clamp ratings
        new_rating_a = max(self.min_rating, min(self.max_rating, new_rating_a))
        new_rating_b = max(self.min_rating, min(self.max_rating, new_rating_b))
        
        # Update stored ratings
        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b
        
        # Record match
        self.match_history.append({
            "player_a": player_a,
            "player_b": player_b,
            "result": result,
            "rating_a_before": rating_a,
            "rating_b_before": rating_b,
            "rating_a_after": new_rating_a,
            "rating_b_after": new_rating_b
        })
        
        return new_rating_a, new_rating_b
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get all players sorted by rating."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        if not self.ratings:
            return {}
        
        ratings_list = list(self.ratings.values())
        return {
            "num_players": len(self.ratings),
            "num_matches": len(self.match_history),
            "avg_rating": sum(ratings_list) / len(ratings_list),
            "max_rating": max(ratings_list),
            "min_rating": min(ratings_list),
            "rating_spread": max(ratings_list) - min(ratings_list)
        }


class ArenaEvaluator:
    """
    Arena evaluator for head-to-head PPT comparisons.
    
    Compares presentations pairwise and maintains ELO rankings.
    """
    
    def __init__(
        self,
        vlm_provider: str = "openai",
        vlm_model: str = None,
        ppt_gen_root: str = None,
        output_dir: str = None,
        max_images_per_ppt: int = None,
        quantitative_results_path: str = None,
        source_mode: str = "selected",
        num_workers: int = 1,
        use_grids: bool = False
    ):
        """
        Initialize arena evaluator.
        
        Args:
            vlm_provider: VLM provider to use
            vlm_model: Specific model name
            ppt_gen_root: Root directory for generated PPTs
            output_dir: Directory to save results
            max_images_per_ppt: Max images per presentation
            quantitative_results_path: Path to quantitative results for metadata
            source_mode: Source content mode - 'selected', 'lite', or 'all'
            num_workers: Number of parallel workers for VLM calls (default: 1 for sequential)
            use_grids: Whether to use pre-generated grid images from slide_images_grid folder
        """
        # Store VLM config for creating per-thread instances
        self.vlm_provider = vlm_provider
        self.vlm_model = vlm_model
        self.num_workers = max(1, num_workers)
        self.use_grids = use_grids
        
        # Create main VLM instance (used for sequential mode or main thread)
        self.vlm = VLMInterface(provider=vlm_provider, model_name=vlm_model)
        
        # Thread-local storage for VLM instances in parallel mode
        self._thread_local = threading.local()
        
        self.ppt_gen_root = ppt_gen_root or EvalConfig.PPT_GEN_ROOT
        self.output_dir = output_dir or EvalConfig.OUTPUT_DIR
        self.max_images = max_images_per_ppt or EvalConfig.MAX_IMAGES_PER_PPT
        self.source_mode = source_mode
        
        # Load quantitative results if available
        self.quant_results = {}
        if quantitative_results_path and os.path.exists(quantitative_results_path):
            self._load_quantitative_results(quantitative_results_path)
        
        # Initialize ELO system
        self.elo = ELOSystem(
            initial_rating=EvalConfig.ELO_INITIAL_RATING,
            k_factor=EvalConfig.ELO_K_FACTOR
        )
        
        # Lock for thread-safe ELO updates
        self._elo_lock = threading.Lock()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"ArenaEvaluator initialized:")
        logger.info(f"  VLM: {self.vlm.provider}/{self.vlm.model_name}")
        logger.info(f"  Source Mode: {self.source_mode}")
        logger.info(f"  PPT Gen Root: {self.ppt_gen_root}")
        logger.info(f"  Parallel workers: {self.num_workers}")
        logger.info(f"  Use grids: {self.use_grids}")
        
        # Cached results for incremental evaluation
        self.cached_matches: List[Dict] = []
        self.cached_products: set = set()
        
        # Incremental saving settings
        self.incremental_output_path: Optional[str] = None
        self.auto_save: bool = True
        self.save_interval: int = 1  # Save after every N matches
    
    def load_cached_arena_results(self, cache_path: str) -> bool:
        """
        Load cached arena results for incremental evaluation.
        
        This allows running arena evaluation for new products while
        reusing existing match results for previously evaluated products.
        
        Args:
            cache_path: Path to cached arena_results.json
            
        Returns:
            True if cache was loaded successfully
        """
        if not os.path.exists(cache_path):
            logger.warning(f"Cache file not found: {cache_path}")
            return False
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            
            # Load cached matches
            self.cached_matches = cached_data.get("matches", [])
            
            # Extract products from cached matches
            for match in self.cached_matches:
                self.cached_products.add(match.get("ppt_a_product", ""))
                self.cached_products.add(match.get("ppt_b_product", ""))
            
            # Restore ELO ratings from cache
            cached_ratings = cached_data.get("elo_ratings", {})
            for product, rating in cached_ratings.items():
                self.elo.ratings[product] = rating
            
            # Restore match history
            cached_history = cached_data.get("match_history", [])
            self.elo.match_history = cached_history.copy()
            
            logger.info(f"Loaded cached arena results:")
            logger.info(f"  Cached matches: {len(self.cached_matches)}")
            logger.info(f"  Cached products: {sorted(self.cached_products)}")
            logger.info(f"  Restored ELO ratings: {len(cached_ratings)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading cached arena results: {e}")
            return False
    
    def _is_match_cached(self, product_a: str, product_b: str, topic: str) -> bool:
        """
        Check if a match between two products on a topic is already cached.
        
        Args:
            product_a: First product name
            product_b: Second product name  
            topic: Topic name
            
        Returns:
            True if match exists in cache
        """
        for match in self.cached_matches:
            match_products = {match.get("ppt_a_product"), match.get("ppt_b_product")}
            if match_products == {product_a, product_b} and match.get("topic") == topic:
                return True
        return False
    
    def _save_incremental(self, new_results: List[MatchResult]) -> None:
        """
        Save results incrementally to allow resuming.
        
        Args:
            new_results: List of new MatchResult objects to save
        """
        if not self.incremental_output_path:
            return
        
        try:
            # Combine cached matches with new results
            all_matches = self.cached_matches.copy()
            for result in new_results:
                all_matches.append(result.to_dict())
            
            # Compile output data
            output_data = {
                "metadata": {
                    "evaluation_time": datetime.now().isoformat(),
                    "vlm_model": self.vlm.model_name,
                    "vlm_provider": self.vlm.provider,
                    "num_matches": len(all_matches),
                    "is_incremental": True
                },
                "matches": all_matches,
                "elo_ratings": dict(self.elo.ratings),
                "elo_rankings": [
                    {"rank": i+1, "product": prod, "rating": rating}
                    for i, (prod, rating) in enumerate(self.elo.get_rankings())
                ],
                "elo_statistics": self.elo.get_statistics(),
                "match_history": self.elo.match_history
            }
            
            # Write to temp file first, then rename for atomic write
            temp_path = self.incremental_output_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            os.replace(temp_path, self.incremental_output_path)
            
            logger.debug(f"Incremental save: {len(all_matches)} matches saved to {self.incremental_output_path}")
            
        except Exception as e:
            logger.error(f"Error during incremental save: {e}")
    
    def _needs_evaluation(self, product_a: str, product_b: str, topic: str, new_products: set = None) -> bool:
        """
        Determine if a match needs evaluation.
        
        A match needs evaluation if:
        1. It's not in cache, OR
        2. At least one product is in the new_products set
        
        Args:
            product_a: First product name
            product_b: Second product name
            topic: Topic name
            new_products: Set of new products that always need evaluation
            
        Returns:
            True if match should be evaluated
        """
        # If new_products specified and at least one product is new, evaluate
        if new_products:
            if product_a in new_products or product_b in new_products:
                return True
        
        # If not cached, needs evaluation
        if not self._is_match_cached(product_a, product_b, topic):
            return True
        
        return False
    
    def _get_thread_vlm(self) -> VLMInterface:
        """
        Get or create a VLM instance for the current thread.
        
        In parallel mode, each thread needs its own VLM client to avoid
        race conditions and connection issues.
        
        Returns:
            VLMInterface instance for the current thread
        """
        if not hasattr(self._thread_local, 'vlm'):
            self._thread_local.vlm = VLMInterface(
                provider=self.vlm_provider,
                model_name=self.vlm_model
            )
        return self._thread_local.vlm
    
    def _load_quantitative_results(self, path: str):
        """Load quantitative results for PPT metadata."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results = data.get("results", data) if isinstance(data, dict) else data
            
            for res in results:
                key = f"{res.get('product', res.get('Product'))}_{res.get('topic', res.get('Topic'))}_{res.get('ppt_id', res.get('ID'))}"
                self.quant_results[key] = res
            
            logger.info(f"Loaded {len(self.quant_results)} quantitative results")
        except Exception as e:
            logger.warning(f"Could not load quantitative results: {e}")
    
    def get_image_paths(self, ppt_path: str) -> List[str]:
        """Get sorted image paths for a PPT."""
        slide_images_dir = os.path.join(ppt_path, "slide_images")
        if not os.path.exists(slide_images_dir):
            return []
        
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(os.path.join(slide_images_dir, pattern)))
        
        images = sorted(images, key=lambda x: os.path.basename(x))
        
        if len(images) > self.max_images:
            step = len(images) / self.max_images
            indices = [int(i * step) for i in range(self.max_images)]
            images = [images[i] for i in indices]
        
        return images
    
    def get_grid_image_paths(self, ppt_path: str) -> List[str]:
        """Get pre-generated grid images from slide_images_grid folder."""
        grid_dir = os.path.join(ppt_path, "slide_images_grid")
        if not os.path.exists(grid_dir):
            logger.warning(f"Grid folder not found: {grid_dir}, falling back to individual images")
            return []
        
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        grids = []
        for pattern in patterns:
            grids.extend(glob.glob(os.path.join(grid_dir, pattern)))
        
        grids = sorted(grids, key=lambda x: os.path.basename(x))
        
        if grids:
            logger.debug(f"Found {len(grids)} grid images in {grid_dir}")
        
        return grids
    
    def get_source_content(self, topic: str, difficulty: str = None) -> Optional[str]:
        """
        Get source content for a topic based on source mode.
        
        Supports three modes:
        - "selected": Full test - sources in Selected/{topic}/{topic}.md
        - "lite": Quick test - sources in Lite/{difficulty}/{topic}/{topic}.md
        - "all": Combined test - searches both Selected and Scene folders
        
        Args:
            topic: Topic name
            difficulty: Difficulty level (required for 'lite' and 'all' modes when topic is in Scene)
            
        Returns:
            Source content string or None
        """
        source_root = EvalConfig.PPT_EVAL_ROOT
        possible_paths = []
        
        if self.source_mode == "selected":
            # Selected mode: Selected/{topic}/{topic}.md
            possible_paths.append(os.path.join(source_root, EvalConfig.SOURCE_SELECTED_DIR, topic, f"{topic}.md"))
        elif self.source_mode == "lite":
            # Lite mode: Lite/{difficulty}/{topic}/{topic}.md
            if not difficulty:
                logger.warning(f"Lite mode requires difficulty, skipping source for {topic}")
                return None
            possible_paths.append(os.path.join(source_root, EvalConfig.SOURCE_LITE_DIR, difficulty, topic, f"{topic}.md"))
        elif self.source_mode == "all":
            # All mode: Search both Selected and Scene folders
            # First try Selected folder
            possible_paths.append(os.path.join(source_root, EvalConfig.SOURCE_SELECTED_DIR, topic, f"{topic}.md"))
            
            # Then try Scene folders (if difficulty is provided and is a Scene category)
            scene_categories = ["work_report", "business_plan", "brand_promote", 
                               "personal_statement", "product_launch", "course_preparation"]
            if difficulty and difficulty in scene_categories:
                # Scene/{difficulty}/{topic}/{topic}.md
                possible_paths.extend([
                    os.path.join(source_root, "Scene", difficulty, topic, f"{topic}.md"),
                    os.path.join(source_root, "Scene", difficulty, topic, "content.md"),
                    os.path.join(source_root, "Scene", difficulty, topic, f"{topic}.txt"),
                    os.path.join(source_root, "Scene", difficulty, topic, "paragraph.txt"),
                    os.path.join(source_root, "Scene", difficulty, topic, "one_sentence.txt"),
                ])
        
        # Try each path in order
        for source_path in possible_paths:
            if os.path.exists(source_path):
                try:
                    with open(source_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Truncate if too long
                    max_chars = EvalConfig.SOURCE_CONTENT_MAX_CHARS
                    if len(content) > max_chars:
                        content = content[:max_chars] + "\n\n[Content truncated...]"
                    return content
                except Exception as e:
                    logger.warning(f"Could not read source content from {source_path}: {e}")
                    continue
        
        logger.debug(f"Source content not found for topic '{topic}' in any expected location")
        return None

    def build_comparison_prompt(
        self,
        topic: str,
        num_slides_a: int,
        num_slides_b: int
    ) -> str:
        """Build comparison prompt."""
        return ARENA_COMPARISON_PROMPT_TEMPLATE.format(
            topic=topic,
            num_slides_a=num_slides_a,
            num_slides_b=num_slides_b
        )
    
    def compare_ppts(
        self,
        topic: str,
        ppt_a: Dict[str, Any],
        ppt_b: Dict[str, Any]
    ) -> Optional[MatchResult]:
        """
        Compare two PPTs head-to-head.
        
        Args:
            topic: Shared topic
            ppt_a: PPT A metadata dict with product, id, path
            ppt_b: PPT B metadata dict with product, id, path
            
        Returns:
            MatchResult or None if comparison failed
        """
        logger.info(f"Comparing {ppt_a['product']} vs {ppt_b['product']} on '{topic}'")
        
        # Get images - use grids if enabled and available
        if self.use_grids:
            images_a = self.get_grid_image_paths(ppt_a['path'])
            images_b = self.get_grid_image_paths(ppt_b['path'])
            # Fall back to individual images if grids not available
            if not images_a:
                images_a = self.get_image_paths(ppt_a['path'])
            if not images_b:
                images_b = self.get_image_paths(ppt_b['path'])
            
            if images_a and images_b:
                logger.info(f"Using grid images: A={len(images_a)}, B={len(images_b)}")
        else:
            images_a = self.get_image_paths(ppt_a['path'])
            images_b = self.get_image_paths(ppt_b['path'])
        
        if not images_a or not images_b:
            logger.warning(f"Missing images for comparison. A: {len(images_a)}, B: {len(images_b)}")
            return None
        
        original_count_a = len(images_a)
        original_count_b = len(images_b)
        
        # Limit images if needed
        max_images_arena = MAX_IMAGES_PER_PPT_ARENA
        
        if len(images_a) > max_images_arena:
            step = len(images_a) / max_images_arena
            indices = [int(i * step) for i in range(max_images_arena)]
            images_a = [images_a[i] for i in indices]
            logger.info(f"Limited PPT A images from {original_count_a} to {len(images_a)}")
        
        if len(images_b) > max_images_arena:
            step = len(images_b) / max_images_arena
            indices = [int(i * step) for i in range(max_images_arena)]
            images_b = [images_b[i] for i in indices]
            logger.info(f"Limited PPT B images from {original_count_b} to {len(images_b)}")
        
        total_images = len(images_a) + len(images_b)
        logger.info(f"Arena comparison with {len(images_a)} + {len(images_b)} = {total_images} total images")
        
        # Build prompt
        prompt = self.build_comparison_prompt(topic, len(images_a), len(images_b))
        
        # Create labeled image groups for the VLM
        image_groups = {
            "A": images_a,
            "B": images_b
        }
        
        try:
            # Get thread-local VLM instance for parallel safety
            vlm = self._get_thread_vlm()
            
            # Call VLM with labeled images (grids or individual)
            response = vlm.call_vlm_with_labeled_images(
                prompt, 
                image_groups, 
                parse_json=True
            )
            
            if isinstance(response, str):
                response = json.loads(response)
            
            # Parse response (Visual Design and Layout only)
            result = MatchResult(
                topic=topic,
                ppt_a_product=ppt_a['product'],
                ppt_a_id=ppt_a['id'],
                ppt_a_path=ppt_a['path'],
                ppt_b_product=ppt_b['product'],
                ppt_b_id=ppt_b['id'],
                ppt_b_path=ppt_b['path'],
                
                visual_winner=response.get("Visual_Design", response.get("Style", {})).get("winner", "Tie"),
                visual_reason=response.get("Visual_Design", response.get("Style", {})).get("reason", ""),
                visual_score_diff=response.get("Visual_Design", response.get("Style", {})).get("score_difference", 0),
                
                layout_winner=response.get("Layout", {}).get("winner", "Tie"),
                layout_reason=response.get("Layout", {}).get("reason", ""),
                layout_score_diff=response.get("Layout", {}).get("score_difference", 0),
                
                overall_winner=response.get("Overall_Winner", "Tie"),
                overall_reason=response.get("Overall_Reason", ""),
                confidence=response.get("Confidence", 3),
                key_differences=response.get("Key_Differences", []),
                
                evaluation_time=datetime.now().isoformat(),
                vlm_model=vlm.model_name
            )
            
            # Update ELO ratings (thread-safe)
            with self._elo_lock:
                winner = result.overall_winner
                if winner == "A":
                    winner_product = ppt_a['product']
                elif winner == "B":
                    winner_product = ppt_b['product']
                else:
                    winner_product = None
                
                self.elo.update_ratings(
                    ppt_a['product'],
                    ppt_b['product'],
                    result.overall_winner,
                    result.confidence
                )
            
            logger.info(f"  Winner: {result.overall_winner} "
                       f"(Confidence: {result.confidence}/5) - {result.overall_reason[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comparison: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_arena(
        self,
        products: List[str] = None,
        topics: List[str] = None,
        match_limit: int = None,
        comparison_strategy: str = "all",
        new_products: List[str] = None,
        use_cache: bool = False,
        output_file: str = None
    ) -> List[MatchResult]:
        """
        Run arena evaluation with incremental saving and optional parallel execution.
        
        Args:
            products: Products to include (None for all)
            topics: Topics to include (None for all)
            match_limit: Maximum matches to run
            comparison_strategy: "all" for all pairs, "round_robin" for one per topic
            new_products: List of new products - only matches involving these products will be evaluated,
                         others will use cached results. If None and use_cache=True, products not in
                         cache will be treated as new.
            use_cache: If True, use cached matches for existing product pairs
            output_file: Output filename for incremental saving (enables auto-resume)
            
        Returns:
            List of MatchResult objects (including cached and new matches)
        """
        results = []
        new_results = []
        match_count = 0
        skipped_cached = 0
        
        # Ensure output_file is set for incremental saving
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"arena_results_{timestamp}.json"
            logger.info(f"No output file specified, using default: {output_file}")
        
        # Set up incremental output path
        self.incremental_output_path = os.path.join(self.output_dir, output_file)
        
        # Auto-load existing results if file exists (for resume capability)
        if os.path.exists(self.incremental_output_path) and not self.cached_matches:
            logger.info(f"Found existing results file, loading for resume: {self.incremental_output_path}")
            self.load_cached_arena_results(self.incremental_output_path)
            use_cache = True
        
        # Determine new products set
        new_products_set = set(new_products) if new_products else set()
        
        # If use_cache but no explicit new_products, auto-detect new products
        if use_cache and not new_products_set and self.cached_products:
            # Will be populated during discovery
            pass
        
        # Discover available PPTs
        ppts_by_topic = self._discover_ppts(products, topics)
        
        logger.info(f"Found {len(ppts_by_topic)} topics with multiple products")
        
        # Auto-detect new products if using cache
        if use_cache and not new_products_set:
            all_discovered_products = set()
            for topic, ppts in ppts_by_topic.items():
                for ppt in ppts:
                    all_discovered_products.add(ppt['product'])
            new_products_set = all_discovered_products - self.cached_products
            if new_products_set:
                logger.info(f"Auto-detected new products: {sorted(new_products_set)}")
        
        # Count total matches to evaluate for progress tracking
        total_pairs = 0
        for topic, ppts in ppts_by_topic.items():
            if len(ppts) >= 2:
                if comparison_strategy == "round_robin":
                    total_pairs += len(list(itertools.combinations(ppts, 2)))
                elif comparison_strategy == "random":
                    total_pairs += min(len(list(itertools.combinations(ppts, 2))), 3)
                else:
                    total_pairs += len(list(itertools.combinations(ppts, 2)))
        
        logger.info(f"Total potential matches: {total_pairs}")
        
        # Collect all pairs that need evaluation
        pairs_to_evaluate = []
        for topic, ppts in ppts_by_topic.items():
            if len(ppts) < 2:
                continue
            
            # Generate pairs based on strategy
            if comparison_strategy == "round_robin":
                # One comparison per product pair per topic
                pairs = list(itertools.combinations(ppts, 2))
            elif comparison_strategy == "random":
                # Random sampling of pairs
                all_pairs = list(itertools.combinations(ppts, 2))
                pairs = random.sample(all_pairs, min(len(all_pairs), 3))
            else:  # "all"
                pairs = list(itertools.combinations(ppts, 2))
            
            for ppt_a, ppt_b in pairs:
                product_a = ppt_a['product']
                product_b = ppt_b['product']
                
                # Check if we should use cache or evaluate
                if use_cache and not self._needs_evaluation(product_a, product_b, topic, new_products_set):
                    # Find and use cached result
                    skipped_cached += 1
                    logger.debug(f"Using cached result for {product_a} vs {product_b} on '{topic}'")
                    continue
                
                pairs_to_evaluate.append((topic, ppt_a, ppt_b))
        
        # Apply match limit
        if match_limit and len(pairs_to_evaluate) > match_limit:
            pairs_to_evaluate = pairs_to_evaluate[:match_limit]
            logger.info(f"Limited to {match_limit} matches")
        
        logger.info(f"Pairs to evaluate: {len(pairs_to_evaluate)}, Cached: {skipped_cached}")
        
        # Run evaluations (parallel or sequential)
        if self.num_workers > 1 and len(pairs_to_evaluate) > 1:
            # Parallel execution
            logger.info(f"Running parallel evaluation with {self.num_workers} workers")
            new_results = self._run_parallel(pairs_to_evaluate, total_pairs, skipped_cached)
        else:
            # Sequential execution
            new_results = self._run_sequential(pairs_to_evaluate, total_pairs, skipped_cached)
        
        match_count = len(new_results)
        
        # Final incremental save
        if self.auto_save and new_results:
            self._save_incremental(new_results)
        
        # Log summary
        logger.info(f"Arena evaluation summary:")
        logger.info(f"  Cached matches reused: {skipped_cached}")
        logger.info(f"  New matches evaluated: {len(new_results)}")
        logger.info(f"  Total matches: {skipped_cached + len(new_results)}")
        
        # Combine cached matches with new results
        # Convert cached matches to MatchResult objects for consistent return type
        if self.cached_matches:
            for cached in self.cached_matches:
                cached_result = self._dict_to_match_result(cached)
                if cached_result:
                    results.append(cached_result)
        
        results.extend(new_results)
        
        return results
    
    def _run_sequential(
        self,
        pairs_to_evaluate: List[Tuple[str, Dict, Dict]],
        total_pairs: int,
        skipped_cached: int
    ) -> List[MatchResult]:
        """Run evaluations sequentially with tqdm progress bar."""
        new_results = []
        match_count = 0
        
        pbar = tqdm(pairs_to_evaluate, desc="Arena Evaluation", unit="match")
        for topic, ppt_a, ppt_b in pbar:
            pbar.set_postfix({
                "topic": topic[:20],
                "A": ppt_a['product'][:10],
                "B": ppt_b['product'][:10]
            })
            result = self.compare_ppts(topic, ppt_a, ppt_b)
            if result:
                new_results.append(result)
                match_count += 1
                
                # Incremental save after each match (or every N matches)
                if self.auto_save and len(new_results) % self.save_interval == 0:
                    self._save_incremental(new_results)
        
        return new_results
    
    def _run_parallel(
        self,
        pairs_to_evaluate: List[Tuple[str, Dict, Dict]],
        total_pairs: int,
        skipped_cached: int
    ) -> List[MatchResult]:
        """Run evaluations in parallel using ThreadPoolExecutor with tqdm progress bar."""
        new_results = []
        results_lock = threading.Lock()
        completed_count = 0
        
        def evaluate_pair(pair_data):
            topic, ppt_a, ppt_b = pair_data
            return self.compare_ppts(topic, ppt_a, ppt_b)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(evaluate_pair, pair): pair 
                for pair in pairs_to_evaluate
            }
            
            # Process completed tasks with tqdm progress bar
            pbar = tqdm(as_completed(future_to_pair), total=len(pairs_to_evaluate),
                       desc=f"Arena Evaluation ({self.num_workers} workers)", unit="match")
            for future in pbar:
                pair = future_to_pair[future]
                topic, ppt_a, ppt_b = pair
                pbar.set_postfix({
                    "topic": topic[:20],
                    "A": ppt_a['product'][:10],
                    "B": ppt_b['product'][:10]
                })
                try:
                    result = future.result()
                    if result:
                        with results_lock:
                            new_results.append(result)
                            completed_count = len(new_results)
                            
                            # Incremental save
                            if self.auto_save and completed_count % self.save_interval == 0:
                                self._save_incremental(new_results)
                except Exception as e:
                    logger.error(f"Error evaluating {ppt_a['product']} vs {ppt_b['product']} on '{topic}': {e}")
        
        return new_results
    
    def _dict_to_match_result(self, data: Dict[str, Any]) -> Optional[MatchResult]:
        """Convert a dictionary to a MatchResult object."""
        try:
            return MatchResult(
                topic=data.get("topic", ""),
                ppt_a_product=data.get("ppt_a_product", ""),
                ppt_a_id=data.get("ppt_a_id", ""),
                ppt_a_path=data.get("ppt_a_path", ""),
                ppt_b_product=data.get("ppt_b_product", ""),
                ppt_b_id=data.get("ppt_b_id", ""),
                ppt_b_path=data.get("ppt_b_path", ""),
                visual_winner=data.get("visual_winner", ""),
                visual_reason=data.get("visual_reason", ""),
                visual_score_diff=data.get("visual_score_diff", 0),
                layout_winner=data.get("layout_winner", ""),
                layout_reason=data.get("layout_reason", ""),
                layout_score_diff=data.get("layout_score_diff", 0),
                overall_winner=data.get("overall_winner", ""),
                overall_reason=data.get("overall_reason", ""),
                confidence=data.get("confidence", 0),
                key_differences=data.get("key_differences", []),
                evaluation_time=data.get("evaluation_time", ""),
                vlm_model=data.get("vlm_model", "")
            )
        except Exception as e:
            logger.warning(f"Failed to convert cached match to MatchResult: {e}")
            return None
    
    def _discover_ppts(
        self,
        products: List[str] = None,
        topics: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover available PPTs grouped by topic.
        
        Handles different product directory structures:
        - Standard: {product}/difficulty/topic/id
        - Kimi: Kimi/{Standard,Smart}/difficulty/topic/name
        - NotebookLM: NotebookLM/difficulty/topic (no subdirectory)
        
        Returns:
            Dict mapping topic -> list of PPT metadata dicts
        """
        return discover_ppts(
            ppt_gen_root=self.ppt_gen_root,
            products=products,
            topics=topics,
            group_by_topic=True,
            require_multiple_products=True
        )
    
    def save_results(
        self,
        results: List[MatchResult],
        filename: str = None
    ) -> str:
        """
        Save arena results to JSON file.
        
        Args:
            results: List of MatchResult objects
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arena_results_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert results to dictionaries
        matches = [r.to_dict() for r in results]
        
        # Compile output data
        output_data = {
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),
                "vlm_model": self.vlm.model_name,
                "vlm_provider": self.vlm.provider,
                "num_matches": len(results)
            },
            "matches": matches,
            "elo_ratings": dict(self.elo.ratings),
            "elo_rankings": [
                {"rank": i+1, "product": prod, "rating": rating}
                for i, (prod, rating) in enumerate(self.elo.get_rankings())
            ],
            "elo_statistics": self.elo.get_statistics(),
            "match_history": self.elo.match_history,
            "summary": self._compute_summary(results)
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Arena results saved to: {output_path}")
        return output_path
    
    def _compute_summary(self, results: List[MatchResult]) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not results:
            return {}
        
        # Count wins by product
        wins = {}
        losses = {}
        ties = {}
        
        for r in results:
            for prod in [r.ppt_a_product, r.ppt_b_product]:
                if prod not in wins:
                    wins[prod] = 0
                    losses[prod] = 0
                    ties[prod] = 0
            
            if r.overall_winner == "A":
                wins[r.ppt_a_product] += 1
                losses[r.ppt_b_product] += 1
            elif r.overall_winner == "B":
                wins[r.ppt_b_product] += 1
                losses[r.ppt_a_product] += 1
            else:
                ties[r.ppt_a_product] += 1
                ties[r.ppt_b_product] += 1
        
        # Win rates
        win_rates = {}
        for prod in wins:
            total = wins[prod] + losses[prod] + ties[prod]
            win_rates[prod] = {
                "wins": wins[prod],
                "losses": losses[prod],
                "ties": ties[prod],
                "total": total,
                "win_rate": wins[prod] / total if total > 0 else 0
            }
        
        return {
            "total_matches": len(results),
            "by_product": win_rates,
            "avg_confidence": sum(r.confidence for r in results) / len(results)
        }


def main():
    """Main entry point for arena evaluation."""
    parser = argparse.ArgumentParser(
        description="Arena PPT Evaluation - Head-to-Head Comparisons"
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "google", "ollama", "mock", "custom-google"],
        help="VLM provider to use"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model name"
    )
    parser.add_argument(
        "--ppt-root",
        default=None,
        help="Root directory for generated PPTs"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results"
    )
    parser.add_argument(
        "--quant-results",
        default=None,
        help="Path to quantitative results file"
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=None,
        help="Products to include"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Topics to include"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of matches"
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "round_robin", "random"],
        help="Comparison strategy"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename (default: arena_results_YYYYMMDD_HHMMSS.json)"
    )
    parser.add_argument(
        "--source-mode",
        choices=["selected", "lite", "all"],
        default="lite",
        help="Source content mode: 'selected' for full test (Selected/{topic}/), 'lite' for quick test (Lite/{difficulty}/{topic}/), 'all' for combined test (both Selected and Scene folders)"
    )
    parser.add_argument(
        "--new-products",
        nargs="+",
        default=None,
        help="New products to evaluate (uses cache for existing products). "
             "Example: --new-products NewTool1 NewTool2"
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Path to cached arena results for incremental evaluation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file (auto-skip completed matches)"
    )
    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Disable auto-saving after each match"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save results every N matches (default: 1)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for VLM calls (default: 1 for sequential). "
             "Recommended: 2-4 for most API providers to avoid rate limits."
    )
    parser.add_argument(
        "--use-grids",
        action="store_true",
        help="Use pre-generated grid images from slide_images_grid folder to reduce token consumption."
    )
    
    args = parser.parse_args()
    
    # Generate output filename with datetime if not specified
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"arena_results_{timestamp}.json"
    
    # Default quant results path
    if not args.quant_results:
        args.quant_results = os.path.join(
            os.path.dirname(__file__), 
            "quantitative_results.json"
        )
    
    # Initialize evaluator
    evaluator = ArenaEvaluator(
        vlm_provider=args.provider,
        vlm_model=args.model,
        ppt_gen_root=args.ppt_root,
        output_dir=args.output_dir,
        quantitative_results_path=args.quant_results,
        source_mode=args.source_mode,
        num_workers=args.workers,
        use_grids=args.use_grids
    )
    
    # Configure auto-save settings
    evaluator.auto_save = not args.no_auto_save
    evaluator.save_interval = args.save_interval
    
    # Load cache for incremental evaluation (explicit cache file)
    use_cache = args.resume  # Enable cache if --resume flag is set
    if args.cache_file and os.path.exists(args.cache_file):
        evaluator.load_cached_arena_results(args.cache_file)
        use_cache = True
        logger.info(f"Loaded explicit cache file: {args.cache_file}")
    elif args.new_products:
        # Try to load from output file location
        cache_path = os.path.join(
            args.output_dir or os.path.dirname(__file__),
            args.output_file
        )
        if os.path.exists(cache_path):
            evaluator.load_cached_arena_results(cache_path)
            use_cache = True
            logger.info(f"Incremental evaluation for new products: {args.new_products}")
    
    # Run arena with incremental saving
    # Note: run_arena will auto-load from output_file if it exists (for resume)
    results = evaluator.run_arena(
        products=args.products,
        topics=args.topics,
        match_limit=args.limit,
        comparison_strategy=args.strategy,
        new_products=args.new_products,
        use_cache=use_cache,
        output_file=args.output_file  # Enable incremental saving
    )
    
    # Save results
    output_path = evaluator.save_results(results, args.output_file)
    
    # Print rankings
    print(f"\n{'='*60}")
    print("ARENA EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Matches completed: {len(results)}")
    print(f"\nFinal ELO Rankings:")
    for i, (prod, rating) in enumerate(evaluator.elo.get_rankings()):
        print(f"  {i+1}. {prod}: {rating:.0f}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
