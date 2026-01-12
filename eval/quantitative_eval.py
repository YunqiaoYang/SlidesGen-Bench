"""
Quantitative Evaluation Module for PPT Assessment

This module performs detailed quantitative evaluation of PowerPoint presentations
using Vision-Language Models (VLMs) to assess multiple criteria including:
- Content accuracy and completeness
- Visual design quality
- Layout effectiveness
- Complexity and sophistication

Author: PPT Evaluation System
"""

import os
import json
import glob
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict


import numpy as np
from tqdm import tqdm

# Log file for skipped items
SKIP_LOG_FILE = "skipped_items.log"

def log_skipped_item(item_type: str, item_name: str, reason: str):
    """Log skipped file or topic due to inappropriate content or other reasons."""
    from datetime import datetime
    log_entry = {
        "type": item_type,
        "name": item_name,
        "reason": reason,
        "time": datetime.now().isoformat()
    }
    with open(SKIP_LOG_FILE, "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

from eval_config import (
    CRITERIA, 
    QUANTITATIVE_EVAL_PROMPT_TEMPLATE,
    SINGLE_SLIDE_EXTRACTION_PROMPT,
    CONTENT_EVALUATION_PROMPT,
    QUIZ_EVALUATION_PROMPT,
    VISUAL_EVALUATION_PROMPT,
    VISUAL_EVALUATION_GRADE_PROMPT,
    generate_criteria_description,
    calculate_weighted_score,
    EvalConfig,
    PRODUCTS,
    get_all_products,
    get_product_path,
    get_product_config
)
from vlm_interface import VLMInterface
from llm_interface import LLMInterface
from utils import discover_ppts
from aesthetics_metrics import (
    SlideAestheticsCalculator,
    ScoreParameters,
    load_parameters_from_config,
    get_default_parameters,
    calculate_total_aesthetics_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mapping used when visual evaluation returns letter grades instead of numeric scores
VISUAL_GRADE_TO_SCORE = {
    "bad": 2,
    "moderate": 6,
    "good": 9,
}


# Simplified prompt for batch evaluation (evaluating subset of slides)
BATCH_EVAL_PROMPT = """You are evaluating a PORTION of a PowerPoint presentation about "{topic}".
This batch contains slides {start_slide} to {end_slide} of {total_slides} total slides.

**Source Document Content:**
{document_content}

**Your Task:**
Evaluate ONLY these slides based on the criteria below. Focus on what you can observe in these specific slides.

**EVALUATION CRITERIA:**

1. **Content**
{content_criteria}

2. **Visual Design**
{visual_design_criteria}

3. **Layout**
{layout_criteria}

4. **Complexity**
{complexity_criteria}

**Output JSON format:**
{{
    "Content": {{"observations": "specific observations for these slides", "sub_scores": {{"Accuracy_and_Completeness": 1-10, "Logical_Flow": 1-10, "Cognitive_Rhythm": 1-10}}}},
    "Visual_Design": {{"observations": "specific observations", "sub_scores": {{"Color_Scheme": 1-10, "Typography": 1-10, "Visual_Consistency": 1-10, "Image_Quality": 1-10, "Theme_Appropriateness": 1-10}}}},
    "Layout": {{"observations": "specific observations", "sub_scores": {{"Spatial_Balance": 1-10, "Element_Alignment": 1-10, "No_Overlapping": 1-10}}}},
    "Complexity": {{"observations": "specific observations", "sub_scores": {{"Charts_and_Data": 1-10, "Visual_Elements": 1-10, "Advanced_Design": 1-10, "Layout_Variety": 1-10}}}},
    "slide_notes": ["notable points about individual slides"]
}}
"""


@dataclass
class EvaluationResult:
    """Data class for storing evaluation results."""
    product: str
    difficulty: str
    topic: str
    ppt_id: str
    ppt_path: str
    num_slides: int
    
    # Main scores (calculated from sub-scores)
    content_score: float = 0.0
    visual_design_score: float = 0.0
    layout_score: float = 0.0
    complexity_score: float = 0.0
    
    # Sub-scores (now the primary scores returned by VLM)
    content_sub_scores: Dict[str, float] = None
    visual_design_sub_scores: Dict[str, float] = None
    layout_sub_scores: Dict[str, float] = None
    complexity_sub_scores: Dict[str, float] = None

    # Grade summaries when visual_grade_mode is used
    visual_design_grades: Dict[str, str] = None
    layout_grades: Dict[str, str] = None
    complexity_grades: Dict[str, str] = None
    
    # Reasoning
    content_reason: str = ""
    visual_design_reason: str = ""
    layout_reason: str = ""
    complexity_reason: str = ""
    
    # Aggregated scores
    weighted_total: float = 0.0
    raw_total: float = 0.0
    
    # Feedback
    overall_feedback: str = ""
    strengths: List[str] = None
    improvements: List[str] = None
    
    # Quiz evaluation results
    quiz_accuracy: float = 0.0
    quiz_total_questions: int = 0
    quiz_correct_answers: int = 0
    quiz_details: Dict[str, Any] = None
    
    # Aesthetics metrics (computed from images)
    aesthetics_metrics: Dict[str, Any] = None
    
    # Metadata
    evaluation_time: str = ""
    vlm_model: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding reason fields."""
        result = asdict(self)
        # Remove reason fields to save tokens
        result.pop("content_reason", None)
        result.pop("visual_design_reason", None)
        result.pop("layout_reason", None)
        result.pop("complexity_reason", None)
        return result


class QuantitativeEvaluator:
    """
    Quantitative evaluator for PowerPoint presentations.
    
    Uses VLM to assess presentations against detailed criteria and sub-criteria.
    """
    
    def __init__(
        self,
        vlm_provider: str = "openai",
        vlm_model: str = None,
        llm_provider: str = None,
        llm_model: str = None,
        ppt_gen_root: str = None,
        ppt_eval_root: str = None,
        output_dir: str = None,
        max_images: int = None,
        max_images_per_call: int = None,
        source_mode: str = "all",
        num_workers: int = 1,
        compute_aesthetics: bool = True,
        aesthetics_metrics: Optional[str] = None,
        aesthetics_config: Optional[str] = None,
        eval_mode: str = "full",
        use_grids: bool = False,
        visual_grade_mode: bool = False,
        quiz_data_file: str = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            vlm_provider: VLM provider to use
            vlm_model: Specific model name
            ppt_gen_root: Root directory for generated PPTs
            ppt_eval_root: Root directory for evaluation data
            output_dir: Directory to save results
            max_images: Maximum images to process per PPT
            max_images_per_call: Maximum images per VLM call (for batching)
            source_mode: Source content mode - "selected" (full test), "lite" (quick test), or "all" (both Selected and Scene)
            num_workers: Number of parallel workers for VLM calls (default: 1 for sequential)
            compute_aesthetics: Whether to compute image-based aesthetics metrics
            aesthetics_metrics: Optional comma-separated list of aesthetics metrics to compute.
                               If None, compute all available aesthetics metrics.
            aesthetics_config: Optional path to JSON config file for aesthetics score parameters.
                              If provided, loads parameters like weight_contrast, harmony_sigma, etc.
            eval_mode: Evaluation mode - "full" (VLM + aesthetics), "vlm_only" (legacy content + visual only),
                      "content_only", "visual_only", or "aesthetics_only"
            use_grids: Whether to use slide grids to reduce VLM token consumption
            visual_grade_mode: If True, request three-level grades (bad/moderate/good) for visual evaluation
            quiz_data_file: Path to JSON file containing quiz data (default: PPT_Product_Generation/final_with_quiz.json)
        """
        # Store VLM config for creating per-thread instances
        self.vlm_provider = vlm_provider
        self.vlm_model = vlm_model
        
        # Store LLM config for creating per-thread instances (use same as VLM if not specified)
        self.llm_provider = llm_provider or vlm_provider
        self.llm_model = llm_model or vlm_model
        
        self.num_workers = max(1, num_workers)
        self.compute_aesthetics = compute_aesthetics
        self.aesthetics_metrics = aesthetics_metrics
        self.aesthetics_config = aesthetics_config
        self.use_grids = use_grids
        self.visual_grade_mode = visual_grade_mode
        
        # Load aesthetics score parameters from config file if provided
        self.score_parameters = None
        if aesthetics_config and os.path.exists(aesthetics_config):
            self.score_parameters = load_parameters_from_config(aesthetics_config)
            logger.info(f"Loaded aesthetics score parameters from: {aesthetics_config}")
        else:
            self.score_parameters = get_default_parameters()
            if aesthetics_config:
                logger.warning(f"Aesthetics config file not found: {aesthetics_config}, using defaults")
        
        # Set evaluation mode
        self.eval_mode = eval_mode.lower()
        allowed_eval_modes = ["full", "vlm_only", "content_only", "visual_only", "aesthetics_only"]
        if self.eval_mode not in allowed_eval_modes:
            raise ValueError(
                f"Invalid eval_mode: {eval_mode}. Must be one of {', '.join(allowed_eval_modes)}"
            )

        # Determine what to compute based on eval_mode
        self.run_content_vlm = self.eval_mode in ["full", "vlm_only", "content_only"]
        self.run_visual_vlm = self.eval_mode in ["full", "vlm_only", "visual_only"]
        self.run_vlm = self.run_content_vlm or self.run_visual_vlm
        self.run_aesthetics = self.eval_mode in ["full", "aesthetics_only"] and compute_aesthetics
        
        # Create main VLM instance (used for sequential mode or main thread)
        if self.run_vlm:
            self.vlm = VLMInterface(provider=vlm_provider, model_name=vlm_model)
        else:
            self.vlm = None
        
        # Create aesthetics calculator
        if self.run_aesthetics:
            self.aesthetics_calculator = SlideAestheticsCalculator()
        else:
            self.aesthetics_calculator = None
        
        # Thread-local storage for VLM instances in parallel mode
        self._thread_local = threading.local()
        
        self.ppt_gen_root = ppt_gen_root or EvalConfig.PPT_GEN_ROOT
        self.ppt_eval_root = ppt_eval_root or EvalConfig.PPT_EVAL_ROOT
        self.output_dir = output_dir or EvalConfig.OUTPUT_DIR
        self.max_images = max_images or EvalConfig.MAX_IMAGES_PER_PPT
        self.max_images_per_call = max_images_per_call or EvalConfig.MAX_IMAGES_PER_VLM_CALL
        self.source_mode = source_mode.lower()
        
        # Quiz data
        self.quiz_data_file = quiz_data_file
        self.quiz_data_by_topic = {}
        self._load_quiz_data()
        
        # Cache for incremental evaluation
        self.cached_results: Dict[str, EvaluationResult] = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"QuantitativeEvaluator initialized:")
        logger.info(f"  Eval mode: {self.eval_mode}")
        if self.vlm:
            logger.info(f"  VLM: {self.vlm.provider}/{self.vlm.model_name}")
        else:
            logger.info(f"  VLM: disabled (aesthetics_only mode)")
        logger.info(f"  PPT Gen Root: {self.ppt_gen_root}")
        logger.info(f"  Output Dir: {self.output_dir}")
        logger.info(f"  Max images per PPT: {self.max_images}, per VLM call: {self.max_images_per_call}")
        logger.info(f"  Source mode: {self.source_mode}")
        logger.info(f"  Parallel workers: {self.num_workers}")
        logger.info(f"  Run VLM: {self.run_vlm}")
        logger.info(f"  Run content VLM: {self.run_content_vlm}")
        logger.info(f"  Run visual VLM: {self.run_visual_vlm}")
        logger.info(f"  Run aesthetics: {self.run_aesthetics}")
        logger.info(f"  Aesthetics config: {self.aesthetics_config or 'default'}")
        logger.info(f"  Use grids: {self.use_grids}")
        logger.info(f"  Quiz data file: {self.quiz_data_file}")
        logger.info(f"  Quiz data loaded: {len(self.quiz_data_by_topic)} topics")
    
    def _load_quiz_data(self):
        """Load quiz data from JSON file and index by topic."""
        if not os.path.exists(self.quiz_data_file):
            logger.warning(f"Quiz data file not found: {self.quiz_data_file}")
            return
        
        try:
            with open(self.quiz_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = data.get("documents", [])
            for doc in documents:
                # Extract topic from meta_info
                meta_info = doc.get("meta_info", {})
                quiz_data = doc.get("quiz_data", {})
                
                if not quiz_data:
                    continue
                
                # Use folder name from source_path as the primary key
                # source_path format: ./domain/folder_name/filename.pdf
                source_path = meta_info.get("source_path", "")
                if source_path:
                    parts = source_path.split("/")
                    if len(parts) >= 3:
                        folder_name = parts[2]  # The folder containing the PDF
                        self.quiz_data_by_topic[folder_name] = quiz_data
                
                # Also index by filename (without extension) as fallback for compatibility
                filename = meta_info.get("filename", "")
                if filename:
                    topic = filename.rsplit(".", 1)[0]  # Remove extension properly
                    if topic and topic not in self.quiz_data_by_topic:
                        self.quiz_data_by_topic[topic] = quiz_data
            
            logger.info(f"Loaded quiz data for {len(self.quiz_data_by_topic)} topics")
            
        except Exception as e:
            logger.error(f"Failed to load quiz data: {e}")
            import traceback
            traceback.print_exc()
    
    def get_quiz_data(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get quiz data for a specific topic."""
        return self.quiz_data_by_topic.get(topic, None)
    
    def _get_thread_vlm(self) -> VLMInterface:
        """
        Get or create a VLM instance for the current thread.
        
        In parallel mode, each thread needs its own VLM client to avoid
        race conditions and connection issues.
        
        Returns:
            VLMInterface instance for the current thread, or None if VLM is disabled
        """
        if not self.run_vlm:
            return None
            
        if not hasattr(self._thread_local, 'vlm'):
            self._thread_local.vlm = VLMInterface(
                provider=self.vlm_provider, 
                model_name=self.vlm_model
            )
        return self._thread_local.vlm
    
    def _get_thread_llm(self) -> LLMInterface:
        """
        Get or create a thread-local LLM instance.
        
        Each thread gets its own LLM instance to avoid thread-safety issues,
        race conditions and connection issues.
        
        Returns:
            LLMInterface instance for the current thread
        """
        if not hasattr(self._thread_local, 'llm'):
            self._thread_local.llm = LLMInterface(
                provider=self.llm_provider, 
                model_name=self.llm_model
            )
        return self._thread_local.llm
    
    def load_cache(self, cache_path: str) -> int:
        """
        Load cached evaluation results from a JSON file.
        
        Args:
            cache_path: Path to the cached results JSON file
            
        Returns:
            Number of cached results loaded
        """
        if not os.path.exists(cache_path):
            logger.info(f"No cache file found at: {cache_path}")
            return 0
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            results = data.get("results", [])
            
            for r in results:
                # Create a unique key for each PPT
                key = self._make_cache_key(
                    r.get("product", ""),
                    r.get("difficulty", ""),
                    r.get("topic", ""),
                    r.get("ppt_id", "")
                )
                self.cached_results[key] = r
            
            logger.info(f"Loaded {len(self.cached_results)} cached results from: {cache_path}")
            return len(self.cached_results)
            
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return 0
    
    def _make_cache_key(self, product: str, difficulty: str, topic: str, ppt_id: str) -> str:
        """Create a unique cache key for a PPT."""
        return f"{product}|{difficulty}|{topic}|{ppt_id}"
    
    def is_cached(self, product: str, difficulty: str, topic: str, ppt_id: str) -> bool:
        """Check if a PPT has already been evaluated."""
        key = self._make_cache_key(product, difficulty, topic, ppt_id)
        return key in self.cached_results
    
    def get_cached_result(self, product: str, difficulty: str, topic: str, ppt_id: str) -> Optional[Dict]:
        """Get cached result for a PPT if available, with recalculated scores."""
        key = self._make_cache_key(product, difficulty, topic, ppt_id)
        cached = self.cached_results.get(key)
        
        if cached is not None:
            # Recalculate weighted_total and raw_total using current scoring formula
            cached = self._recalculate_scores(cached)
        
        return cached
    
    def needs_quiz_recalculation(self, cached_result: Dict) -> bool:
        """Check if a cached result needs quiz recalculation (quiz_details is null)."""
        if cached_result is None:
            return False
        
        quiz_details = cached_result.get('quiz_details')
        return quiz_details is None
    
    def _recalculate_scores(self, result: Dict) -> Dict:
        """
        Recalculate weighted_total and raw_total for a cached result.
        
        This ensures cached results use the current scoring weights/formula.
        
        Args:
            result: Cached result dictionary
            
        Returns:
            Result with recalculated scores
        """
        # Extract individual scores (handle both attribute and dict access)
        content_score = result.get('content_score', 0)
        visual_design_score = result.get('visual_design_score', 0)
        layout_score = result.get('layout_score', 0)
        complexity_score = result.get('complexity_score', 0)
        
        # Recalculate raw_total
        result['raw_total'] = (
            content_score +
            visual_design_score +
            layout_score +
            complexity_score
        )
        
        # Recalculate weighted_total using current weights
        scores = {
            "Content": content_score,
            "Visual_Design": visual_design_score,
            "Layout": layout_score,
            "Complexity": complexity_score
        }
        result['weighted_total'] = calculate_weighted_score(scores)
        
        return result
    
    def recalculate_quiz_for_cached(self, cached_result: Dict, ppt_path: str, topic: str) -> Dict:
        """
        Recalculate quiz results for a cached result where quiz_details is null.
        
        Args:
            cached_result: Cached result dictionary
            ppt_path: Path to PPT directory
            topic: Topic name
            
        Returns:
            Updated result with quiz data
        """
        logger.info(f"Recalculating quiz for cached result: {topic}")
        
        # Get quiz data for this topic
        quiz_data = self.get_quiz_data(topic)
        if not quiz_data:
            logger.warning(f"No quiz data available for topic: {topic}")
            return cached_result
        
        # Extract slide contents from images
        slide_images = self.get_slide_images(ppt_path)
        if not slide_images:
            logger.warning(f"No slide images found for quiz extraction: {ppt_path}")
            return cached_result
        
        # Extract slide contents
        slide_contents = self.extract_slide_contents(ppt_path, slide_images)
        if not slide_contents:
            logger.warning(f"Failed to extract slide contents for quiz")
            return cached_result
        
        # Evaluate quiz
        quiz_result = self.evaluate_quiz(topic, slide_contents, quiz_data)
        
        if quiz_result:
            # Update cached result with new quiz data
            cached_result['quiz_accuracy'] = quiz_result.get('quiz_accuracy', 0.0)
            cached_result['quiz_total_questions'] = quiz_result.get('quiz_total_questions', 0)
            cached_result['quiz_correct_answers'] = quiz_result.get('quiz_correct_answers', 0)
            cached_result['quiz_details'] = quiz_result.get('quiz_details', None)
            logger.info(f"Quiz recalculated: {cached_result['quiz_accuracy']:.1%} ({cached_result['quiz_correct_answers']}/{cached_result['quiz_total_questions']})")
        else:
            logger.warning(f"Failed to recalculate quiz for {topic}")
        
        return cached_result

    def get_source_content(self, topic: str, difficulty: str = None) -> str:
        """
        Retrieve source document content for a topic.
        
        Supports three modes:
        - "selected": Full test - sources in Selected/{topic}/{topic}.md
        - "lite": Quick test - sources in Lite/{difficulty}/{topic}/{topic}.md
        - "all": Combined test - searches both Selected and Scene folders
        
        Args:
            topic: Topic name to find source content for
            difficulty: Difficulty level (required for "lite" and "all" modes when topic is in Scene)
            
        Returns:
            Source document content or placeholder message
        """
        # Build possible paths based on source mode
        possible_paths = []
        
        if self.source_mode == "topic" or self.source_mode == "all":

            possible_paths.extend([
                os.path.join(self.ppt_eval_root, "topic_introduction", topic, f"{topic}.md"),
                os.path.join(self.ppt_eval_root, "topic_introduction", topic, "content.md"),
                os.path.join(self.ppt_eval_root, "topic_introduction", topic, f"{topic}.txt"),
            ])
        
        
        if self.source_mode == "scene" or self.source_mode == "all":
            # Then try Scene folders (if difficulty is provided and is a Scene category)
            scene_categories = ["work_report", "business_plan", "brand_promote", 
                               "personal_statement", "product_launch", "course_preparation"]
            if difficulty and difficulty in scene_categories:
                # Scene/{difficulty}/{topic}/{topic}.md
                possible_paths.extend([
                    os.path.join(self.ppt_eval_root, "Scene", difficulty, topic, f"{topic}.md"),
                    os.path.join(self.ppt_eval_root, "Scene", difficulty, topic, "content.md"),
                    os.path.join(self.ppt_eval_root, "Scene", difficulty, topic, f"{topic}.txt"),
                    os.path.join(self.ppt_eval_root, "Scene", difficulty, topic, "paragraph.txt"),
                    os.path.join(self.ppt_eval_root, "Scene", difficulty, topic, "one_sentence.txt"),
                ])
            
        
        
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Truncate to avoid token limits
                        max_chars = EvalConfig.SOURCE_CONTENT_MAX_CHARS
                        if len(content) > max_chars:
                            content = content[:max_chars] + "\n\n[Content truncated...]"
                        return content
                except Exception as e:
                    logger.warning(f"Error reading source file {path}: {e}")
        
        logger.warning(f"Source content not found for topic: {topic}")
        return f"[Source content for '{topic}' not available. Evaluate based on general presentation quality and internal consistency.]"
    
    def get_slide_images(self, ppt_path: str) -> List[str]:
        """
        Get paths to slide images for a PPT.
        
        Args:
            ppt_path: Path to PPT directory
            
        Returns:
            Sorted list of image file paths
        """
        slide_images_dir = os.path.join(ppt_path, "slide_images")
        
        if not os.path.exists(slide_images_dir):
            logger.warning(f"Slide images directory not found: {slide_images_dir}")
            return []
        
        # Find all image files
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(os.path.join(slide_images_dir, pattern)))
        
        # Sort by filename to maintain slide order
        images = sorted(images, key=lambda x: os.path.basename(x))
        
        # Limit number of images if needed
        if len(images) > self.max_images:
            logger.warning(f"Limiting images from {len(images)} to {self.max_images}")
            # Take first, last, and evenly spaced middle images
            step = len(images) / self.max_images
            selected_indices = [int(i * step) for i in range(self.max_images)]
            images = [images[i] for i in selected_indices]
        
        return images
    
    def get_or_create_slide_grids(self, ppt_path: str, images: List[str]) -> List[str]:
        """
        Get or create slide grid images for a PPT.
        
        This method checks if grid images already exist in the slide_images_grid folder.
        If they exist, it returns those paths. Otherwise, it creates the grids and saves them.
        
        Args:
            ppt_path: Path to PPT directory
            images: List of individual slide image paths
            
        Returns:
            List of grid image paths
        """
        from vlm_interface import create_slide_grids, CONCAT_SLIDES_PER_IMAGE, CONCAT_GRID_COLS, CONCAT_GRID_ROWS
        
        # Define grid directory
        grid_dir = os.path.join(ppt_path, "slide_images_grid")
        
        # Check if grids already exist
        if os.path.exists(grid_dir):
            # Find all grid images
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            existing_grids = []
            for pattern in patterns:
                existing_grids.extend(glob.glob(os.path.join(grid_dir, pattern)))
            
            if existing_grids:
                # Sort by filename to maintain order
                existing_grids = sorted(existing_grids, key=lambda x: os.path.basename(x))
                logger.info(f"Using cached grid images from {grid_dir} ({len(existing_grids)} grids)")
                return existing_grids
        
        # Create grids if they don't exist
        logger.info(f"Creating slide grids in {grid_dir}...")
        os.makedirs(grid_dir, exist_ok=True)
        
        # Generate identifier from ppt_path (use last component)
        identifier = os.path.basename(ppt_path)
        
        # Create grids and save to directory
        grid_paths = create_slide_grids(
            image_paths=images,
            slides_per_grid=CONCAT_SLIDES_PER_IMAGE,
            cols=CONCAT_GRID_COLS,
            rows=CONCAT_GRID_ROWS,
            label_prefix="",
            output_dir=grid_dir,
            identifier=identifier
        )
        
        logger.info(f"Created {len(grid_paths)} grid images in {grid_dir}")
        return grid_paths
    
    def build_prompt(self, topic: str, source_content: str, num_slides: int) -> str:
        """
        Build the evaluation prompt with criteria descriptions.
        
        Args:
            topic: Presentation topic
            source_content: Source document content
            num_slides: Number of slides in presentation
            
        Returns:
            Formatted prompt string
        """
        return QUANTITATIVE_EVAL_PROMPT_TEMPLATE.format(
            document_content=source_content,
            topic=topic,
            num_slides=num_slides,
            content_criteria=generate_criteria_description("Content"),
            visual_design_criteria=generate_criteria_description("Visual_Design"),
            layout_criteria=generate_criteria_description("Layout"),
            complexity_criteria=generate_criteria_description("Complexity")
        )
    
    def parse_vlm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate VLM response.
        
        Args:
            response: Raw response from VLM (already JSON parsed)
            
        Returns:
            Validated and normalized response with calculated scores
        """
        # Handle both old format (Style) and new format (Visual_Design)
        if "Style" in response and "Visual_Design" not in response:
            response["Visual_Design"] = response.pop("Style")
        
        # Remove Professionalism if present (old format)
        if "Professionalism" in response:
            logger.info("Removing deprecated Professionalism criterion")
            response.pop("Professionalism")
        
        # Ensure all criteria are present
        required_criteria = ["Content", "Visual_Design", "Layout", "Complexity"]
        
        for criterion in required_criteria:
            if criterion not in response:
                logger.warning(f"Missing criterion in response: {criterion}")
                response[criterion] = {
                    "sub_scores": {}
                }
            
            # Calculate main score from sub-scores using weights
            if isinstance(response[criterion], dict):
                # Remove reason field if present (we don't need it anymore)
                if "reason" in response[criterion]:
                    del response[criterion]["reason"]
                
                sub_scores = response[criterion].get("sub_scores", {})
                if sub_scores and criterion in CRITERIA:
                    # Calculate weighted score from sub-scores
                    criterion_config = CRITERIA[criterion]
                    sub_criteria_config = criterion_config.get("sub_criteria", {})
                    
                    weighted_score = 0.0
                    total_weight = 0.0
                    
                    for sub_name, sub_score in sub_scores.items():
                        coerced_score = self._coerce_score(sub_score)
                        sub_scores[sub_name] = coerced_score
                        
                        # Get weight for this sub-criterion
                        if sub_name in sub_criteria_config:
                            weight = sub_criteria_config[sub_name].get("weight", 0.0)
                            weighted_score += coerced_score * weight
                            total_weight += weight
                    
                    # Normalize if weights don't sum to 1.0
                    if total_weight > 0:
                        response[criterion]["score"] = weighted_score / total_weight
                    else:
                        # Fallback to average if no weights
                        response[criterion]["score"] = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 5.0
                else:
                    # No sub-scores provided, use default
                    response[criterion]["score"] = response[criterion].get("score", 5.0)
                
                # Ensure final score is valid (keep as float for weighted calculation)
                response[criterion]["score"] = max(0, min(10, float(response[criterion]["score"])))
        
        return response

    def _coerce_score(self, value: Any) -> int:
        """Convert grade labels or numeric inputs to a clamped 0-10 integer."""
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in VISUAL_GRADE_TO_SCORE:
                return VISUAL_GRADE_TO_SCORE[lowered]
        try:
            return int(round(max(0, min(10, float(value)))))
        except Exception:
            return VISUAL_GRADE_TO_SCORE.get("moderate", 5)
    
    def _evaluate_batched(
        self,
        images: List[str],
        topic: str,
        source_content: str,
        total_slides: int
    ) -> Dict[str, Any]:
        """
        Evaluate a presentation in batches when there are too many images.
        
        Args:
            images: List of image paths
            topic: Presentation topic
            source_content: Source document content
            total_slides: Total number of slides
            
        Returns:
            Aggregated evaluation response
        """
        batch_size = self.max_images_per_call
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        logger.info(f"Using batched evaluation: {len(images)} images in {num_batches} batches of {batch_size}")
        
        batch_results = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch_images = images[start_idx:end_idx]
            
            logger.info(f"  Processing batch {batch_idx + 1}/{num_batches}: slides {start_idx + 1}-{end_idx}")
            
            # Build batch-specific prompt
            prompt = BATCH_EVAL_PROMPT.format(
                topic=topic,
                start_slide=start_idx + 1,
                end_slide=end_idx,
                total_slides=total_slides,
                document_content=source_content[:4000],  # Shorter content for batch calls
                content_criteria=generate_criteria_description("Content"),
                visual_design_criteria=generate_criteria_description("Visual_Design"),
                layout_criteria=generate_criteria_description("Layout"),
                complexity_criteria=generate_criteria_description("Complexity")
            )
            
            try:
                # Use thread-local VLM for parallel safety
                vlm = self._get_thread_vlm()
                response = vlm.call_vlm(prompt, batch_images, parse_json=True)
                if isinstance(response, dict):
                    batch_results.append(response)
                else:
                    logger.warning(f"Batch {batch_idx + 1} returned non-dict response")
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                # Continue with other batches
        
        # Aggregate batch results
        return self._aggregate_batch_results(batch_results, topic, total_slides)
    
    def _aggregate_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        topic: str,
        total_slides: int
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple batch evaluations.
        
        Args:
            batch_results: List of batch evaluation results
            topic: Presentation topic
            total_slides: Total number of slides
            
        Returns:
            Aggregated evaluation response
        """
        if not batch_results:
            logger.error("No batch results to aggregate")
            return {
                "Content": {"reason": "Evaluation failed", "sub_scores": {}},
                "Visual_Design": {"reason": "Evaluation failed", "sub_scores": {}},
                "Layout": {"reason": "Evaluation failed", "sub_scores": {}},
                "Complexity": {"reason": "Evaluation failed", "sub_scores": {}}
            }
        
        criteria = ["Content", "Visual_Design", "Layout", "Complexity"]
        aggregated = {}
        all_observations = {c: [] for c in criteria}
        
        # Collect sub-scores and observations from all batches
        for result in batch_results:
            for criterion in criteria:
                if criterion in result:
                    sub_scores = result[criterion].get("sub_scores", {})
                    obs = result[criterion].get("observations", "")
                    
                    if criterion not in aggregated:
                        aggregated[criterion] = {"sub_scores_list": [], "observations": []}
                    
                    if sub_scores:
                        aggregated[criterion]["sub_scores_list"].append(sub_scores)
                    if obs:
                        aggregated[criterion]["observations"].append(obs)
        
        # Calculate final aggregated response by averaging sub-scores
        final_response = {}
        for criterion in criteria:
            if criterion in aggregated and aggregated[criterion]["sub_scores_list"]:
                sub_scores_list = aggregated[criterion]["sub_scores_list"]
                observations = aggregated[criterion]["observations"]
                
                # Average each sub-score across batches
                aggregated_sub_scores = {}
                all_sub_names = set()
                for sub_scores in sub_scores_list:
                    all_sub_names.update(sub_scores.keys())
                
                for sub_name in all_sub_names:
                    values = [ss.get(sub_name, 5.0) for ss in sub_scores_list if sub_name in ss]
                    if values:
                        aggregated_sub_scores[sub_name] = round(sum(values) / len(values), 1)
                
                final_response[criterion] = {
                    "reason": " | ".join(observations[:3]) if observations else "Aggregated from batch evaluation",
                    "sub_scores": aggregated_sub_scores
                }
            else:
                final_response[criterion] = {
                    "reason": "No batch data available",
                    "sub_scores": {}
                }
        
        # Calculate scores from sub-scores and add summary fields
        for criterion in criteria:
            if final_response[criterion]["sub_scores"]:
                # Calculate weighted score from sub-scores
                sub_scores = final_response[criterion]["sub_scores"]
                criterion_config = CRITERIA.get(criterion, {})
                sub_criteria_config = criterion_config.get("sub_criteria", {})
                
                weighted_score = 0.0
                total_weight = 0.0
                
                for sub_name, sub_score in sub_scores.items():
                    if sub_name in sub_criteria_config:
                        weight = sub_criteria_config[sub_name].get("weight", 0.0)
                        weighted_score += sub_score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    final_response[criterion]["score"] = weighted_score / total_weight
                else:
                    final_response[criterion]["score"] = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 5.0
            else:
                final_response[criterion]["score"] = 5.0
        
        final_response["Weighted_Total"] = calculate_weighted_score(
            {c: final_response[c]["score"] for c in criteria}
        )
        final_response["Overall_Feedback"] = f"Evaluation based on {len(batch_results)} batch(es) covering {total_slides} slides."
        final_response["Top_Strengths"] = ["Batch evaluation completed successfully"]
        final_response["Areas_for_Improvement"] = ["Consider running full evaluation for detailed sub-scores"]
        
        logger.info(f"Aggregated {len(batch_results)} batches into final scores")
        
        return final_response
    
    def extract_slide_contents(
        self,
        ppt_path: str,
        images: List[str],
        force_reextract: bool = False
    ) -> Optional[str]:
        """
        Extract textual content from each slide using VLM, in parallel.
        """
        import concurrent.futures

        contents_dir = os.path.join(ppt_path, "slide_contents")
        os.makedirs(contents_dir, exist_ok=True)

        logger.info(f"Extracting contents from {len(images)} slides in parallel...")

        def process_slide(idx_image):
            idx, image_path = idx_image
            slide_num = idx + 1
            md_filename = os.path.join(contents_dir, f"slide_{slide_num:04d}.md")
            if os.path.exists(md_filename) and not force_reextract:
                try:
                    with open(md_filename, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    return (idx, md_content)
                except Exception as e:
                    logger.error(f"Failed to read Markdown for slide {slide_num}: {e}")
                    return (idx, "")

            prompt = SINGLE_SLIDE_EXTRACTION_PROMPT
            vlm = self._get_thread_vlm()
            try:
                response = vlm.call_vlm(prompt, [image_path], parse_json=False)
            except Exception as e:
                msg = str(e)
                if (
                    "inappropriate content" in msg.lower() or
                    "data_inspection_failed" in msg.lower() or
                    ("Error code: 400" in msg and "inappropriate content" in msg)
                ):
                    log_skipped_item("slide", f"{ppt_path}/slide_{slide_num:04d}", f"Inappropriate content: {msg}")
                    logger.warning(f"Skipped slide {slide_num} in {ppt_path} due to inappropriate content.")
                else:
                    log_skipped_item("slide", f"{ppt_path}/slide_{slide_num:04d}", f"Other error: {msg}")
                    logger.error(f"Skipped slide {slide_num} in {ppt_path} due to error: {msg}")
                return (idx, "")
            try:
                cleaned_response = response
                if cleaned_response.startswith('```markdown'):
                    cleaned_response = cleaned_response[len('```markdown'):].lstrip('\n')
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[len('```'):].lstrip('\n')
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-len('```')].rstrip('\n')
                with open(md_filename, 'w', encoding='utf-8') as f:
                    f.write(cleaned_response)
                logger.info(f"Saved slide {slide_num} content to {md_filename}")
            except Exception as e:
                logger.error(f"Failed to write Markdown for slide {slide_num}: {e}")
                cleaned_response = ""
            return (idx, cleaned_response)

        results = [None] * len(images)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_slide, (idx, image_path)): idx for idx, image_path in enumerate(images)}
            for future in concurrent.futures.as_completed(futures):
                idx, content = future.result()
                results[idx] = content

        extracted_contents = "\n".join(results)
        return extracted_contents
    
    def evaluate_content(
        self,
        topic: str,
        source_content: str,
        slide_contents: List[Dict[str, Any]],
        num_slides: int
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate content quality based on extracted slide contents.
        
        Args:
            topic: Presentation topic
            source_content: Source document content
            slide_contents: Extracted slide contents
            num_slides: Number of slides
            
        Returns:
            Content evaluation result or None if evaluation failed
        """
        logger.info(f"Evaluating content quality based on extracted contents...")
        
        
        # Build content evaluation prompt
        prompt = CONTENT_EVALUATION_PROMPT.format(
            document_content=source_content[:EvalConfig.SOURCE_CONTENT_MAX_CHARS],
            topic=topic,
            num_slides=num_slides,
            slide_contents=slide_contents,
            content_criteria=generate_criteria_description("Content")
        )
        
        # Get thread-local LLM
        llm = self._get_thread_llm()
        
        try:
            # Call LLM (text-only evaluation)
            response = llm.call_llm(prompt, parse_json=True)
            
            # Parse response
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse content evaluation response")
                    return None
            
            # Validate and normalize response
            if not isinstance(response, dict) or "Content" not in response:
                logger.error(f"Invalid content evaluation response format")
                return None
            
            # Ensure sub_scores are integers
            if "sub_scores" in response["Content"]:
                for sub_name, sub_score in response["Content"]["sub_scores"].items():
                    response["Content"]["sub_scores"][sub_name] = int(round(max(0, min(10, float(sub_score)))))
            
            # Calculate main content score from sub-scores
            sub_scores = response["Content"].get("sub_scores", {})
            if sub_scores and "Content" in CRITERIA:
                criterion_config = CRITERIA["Content"]
                sub_criteria_config = criterion_config.get("sub_criteria", {})
                
                weighted_score = 0.0
                total_weight = 0.0
                
                for sub_name, sub_score in sub_scores.items():
                    if sub_name in sub_criteria_config:
                        weight = sub_criteria_config[sub_name].get("weight", 0.0)
                        weighted_score += sub_score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    response["Content"]["score"] = weighted_score / total_weight
                else:
                    response["Content"]["score"] = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 5.0
            else:
                response["Content"]["score"] = 5.0
            
            logger.info(f"Content evaluation completed: score={response['Content']['score']:.1f}")
            return response
            
        except Exception as e:
            logger.error(f"Error evaluating content: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_quiz(
        self,
        topic: str,
        slide_contents: List[Dict[str, Any]],
        quiz_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate presentation by answering quiz questions based on slide contents.
        
        Args:
            topic: Presentation topic
            slide_contents: Extracted slide contents
            quiz_data: Quiz questions and answers
            
        Returns:
            Quiz evaluation result with accuracy metrics, or None if evaluation failed
        """
        if not quiz_data or "quiz_bank" not in quiz_data:
            logger.warning(f"No quiz data available for topic: {topic}")
            return None
        
        quiz_bank = quiz_data["quiz_bank"]
        if not quiz_bank:
            logger.warning(f"Empty quiz bank for topic: {topic}")
            return None
        
        logger.info(f"Evaluating quiz: {len(quiz_bank)} questions...")
        
        # Format quiz questions
        quiz_text = self._format_quiz_questions(quiz_bank)
        
        # Build quiz evaluation prompt
        prompt = QUIZ_EVALUATION_PROMPT.format(
            topic=topic,
            slide_contents=slide_contents,
            quiz_questions=quiz_text
        )
        
        # Get thread-local LLM
        llm = self._get_thread_llm()
        
        try:
            # Call LLM (text-only evaluation)
            response = llm.call_llm(prompt, parse_json=True)
            
            # Parse response
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse quiz evaluation response")
                    return None
            
            # Validate response
            if not isinstance(response, dict) or "answers" not in response:
                logger.error(f"Invalid quiz evaluation response format")
                return None
            
            # Calculate accuracy
            answers = response["answers"]
            correct_count = 0
            total_count = len(quiz_bank)
            
            answer_details = []
            for answer in answers:
                question_id = answer.get("question_id")
                selected_answer = answer.get("selected_answer")
                
                # Find corresponding question in quiz bank
                question = next((q for q in quiz_bank if q.get("id") == question_id), None)
                if question:
                    correct_answer = question.get("correct_answer")
                    is_correct = (selected_answer == correct_answer)
                    
                    if is_correct:
                        correct_count += 1
                    
                    answer_details.append({
                        "question_id": question_id,
                        "question": question.get("question", ""),
                        "selected_answer": selected_answer,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct,
                        "reasoning": answer.get("reasoning", "")
                    })
            
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            
            result = {
                "quiz_accuracy": accuracy,
                "quiz_total_questions": total_count,
                "quiz_correct_answers": correct_count,
                "quiz_details": answer_details
            }
            
            logger.info(f"Quiz evaluation completed: accuracy={accuracy:.2%} ({correct_count}/{total_count})")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating quiz: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_slide_contents(self, slide_contents: List[Dict[str, Any]]) -> str:
        """Format extracted slide contents for prompts."""
        formatted = []
        
        for content in slide_contents:
            slide_num = content.get("slide_number", "?")
            parts = [f"**Slide {slide_num}:**"]
            
            if content.get("title"):
                parts.append(f"Title: {content['title']}")
            if content.get("subtitle"):
                parts.append(f"Subtitle: {content['subtitle']}")
            if content.get("body_text"):
                parts.append(f"Body Text:\n  - " + "\n  - ".join(content["body_text"]))
            if content.get("callouts"):
                parts.append(f"Callouts: " + ", ".join(content["callouts"]))
            if content.get("chart_data"):
                parts.append(f"Chart/Data: " + ", ".join(content["chart_data"]))
            if content.get("metrics"):
                parts.append(f"Metrics: " + ", ".join(content["metrics"]))
            if content.get("visual_emphasis"):
                parts.append(f"Visual Emphasis: {content['visual_emphasis']}")
            if content.get("fine_print"):
                parts.append(f"Fine Print: " + ", ".join(content["fine_print"]))
            if content.get("visual_elements"):
                parts.append(f"Visual Elements: {content['visual_elements']}")
            
            formatted.append("\n".join(parts))
        
        return "\n\n".join(formatted)
    
    def _format_quiz_questions(self, quiz_bank: List[Dict[str, Any]]) -> str:
        """Format quiz questions for prompts."""
        formatted = []
        
        for question in quiz_bank:
            q_id = question.get("id", "?")
            q_type = question.get("type", "Unknown")
            q_text = question.get("question", "")
            q_options = question.get("options", [])
            
            parts = [
                f"Question {q_id} (Type: {q_type}):",
                q_text,
                "Options:"
            ]
            parts.extend(q_options)
            
            formatted.append("\n".join(parts))
        
        return "\n\n".join(formatted)
    
    def evaluate_ppt(
        self,
        product: str,
        difficulty: str,
        topic: str,
        ppt_id: str,
        ppt_path: str,
        quiz_data: Optional[Dict[str, Any]] = None
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a single PowerPoint presentation using the new workflow:
        1. Extract slide contents (text extraction from images)
        2. Evaluate Content criteria based on extracted contents + source document
        3. Evaluate quiz questions based on extracted contents (if quiz_data provided)
        4. Evaluate Visual_Design, Layout, Complexity based on slide images only
        5. Aggregate all scores
        
        Args:
            product: Product/tool name that generated the PPT
            difficulty: Difficulty level
            topic: Presentation topic
            ppt_id: Unique PPT identifier
            ppt_path: Path to PPT directory
            quiz_data: Optional quiz questions for content evaluation
            
        Returns:
            EvaluationResult or None if evaluation failed
        """
        logger.info(f"Evaluating: {product}/{topic}/{ppt_id} (mode={self.eval_mode})")
        
        # Get slide images
        images = self.get_slide_images(ppt_path)
        if not images:
            logger.error(f"No images found for {ppt_id}")
            return None
        
        num_slides = len(images)
        
        try:
            # Initialize variables
            content_response = None
            visual_response = None
            quiz_result = None
            aesthetics_metrics = None
            slide_contents = None
            visual_grades = {"Visual_Design": {}, "Layout": {}, "Complexity": {}}
            
            # =========================================================================
            # STEP 1: Extract slide contents (if content evaluation is requested)
            # =========================================================================
            if self.run_content_vlm:
                slide_contents = self.extract_slide_contents(ppt_path, images)
                if not slide_contents:
                    logger.error(f"Failed to extract slide contents for {ppt_id}")
                    if not self.run_visual_vlm and not self.run_aesthetics:
                        return None
            
            # =========================================================================
            # STEP 2: Evaluate Content criteria (text-based, using extracted contents)
            # =========================================================================
            if self.run_content_vlm and slide_contents:
                # Get source content
                source_content = self.get_source_content(topic, difficulty)
                
                # Evaluate content based on extracted text
                content_response = self.evaluate_content(
                    topic=topic,
                    source_content=source_content,
                    slide_contents=slide_contents,
                    num_slides=num_slides
                )
                
                if not content_response:
                    logger.warning(f"Content evaluation failed for {ppt_id}")
            
            # =========================================================================
            # STEP 3: Evaluate quiz (if quiz data provided)
            # =========================================================================
            if self.run_content_vlm and slide_contents and quiz_data:
                quiz_result = self.evaluate_quiz(
                    topic=topic,
                    slide_contents=slide_contents,
                    quiz_data=quiz_data
                )
                
                if not quiz_result:
                    logger.warning(f"Quiz evaluation failed or no quiz data for {ppt_id}")
            
            # =========================================================================
            # STEP 4: Evaluate Visual_Design, Layout, Complexity (image-based only)
            # =========================================================================
            if self.run_visual_vlm:
                # Use thread-local VLM for parallel safety
                vlm = self._get_thread_vlm()
                
                # Decide which images to use for VLM evaluation
                if self.use_grids:
                    # Use grid images to reduce token consumption
                    vlm_images = self.get_or_create_slide_grids(ppt_path, images)
                    logger.info(f"Using {len(vlm_images)} grid images for visual evaluation (original: {num_slides} slides)")
                else:
                    vlm_images = images
                
                # Build visual evaluation prompt (NO source document content)
                prompt_template = VISUAL_EVALUATION_GRADE_PROMPT if self.visual_grade_mode else VISUAL_EVALUATION_PROMPT
                prompt = prompt_template.format(
                    topic=topic,
                    num_slides=num_slides,
                    visual_design_criteria=generate_criteria_description("Visual_Design"),
                    layout_criteria=generate_criteria_description("Layout"),
                    complexity_criteria=generate_criteria_description("Complexity")
                )
                
                # Call VLM for visual evaluation
                try:
                    if len(vlm_images) > self.max_images_per_call:
                        # For large presentations, we might need batching
                        # For now, just use the first N images to stay within limits
                        logger.warning(f"Too many images ({len(vlm_images)}), using first {self.max_images_per_call}")
                        vlm_images = vlm_images[:self.max_images_per_call]
                    
                    visual_response = vlm.call_vlm(prompt, vlm_images, parse_json=True)
                    
                    # Parse response
                    if isinstance(visual_response, str):
                        try:
                            visual_response = json.loads(visual_response)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse visual evaluation response as JSON")
                            visual_response = None
                    
                    # Validate and normalize response
                    if visual_response:
                        # Ensure sub_scores are integers, convert grades when present, and calculate main scores
                        for criterion in ["Visual_Design", "Layout", "Complexity"]:
                            if criterion in visual_response and isinstance(visual_response[criterion], dict):
                                criterion_data = visual_response[criterion]

                                # Convert grade labels to scores when present
                                detected_grades = {}
                                if isinstance(criterion_data.get("sub_grades"), dict):
                                    detected_grades = {k: str(v).strip().lower() for k, v in criterion_data.get("sub_grades", {}).items()}
                                elif isinstance(criterion_data.get("sub_scores"), dict):
                                    detected_grades = {
                                        k: str(v).strip().lower()
                                        for k, v in criterion_data.get("sub_scores", {}).items()
                                        if isinstance(v, str) and str(v).strip().lower() in VISUAL_GRADE_TO_SCORE
                                    }

                                if detected_grades:
                                    visual_grades[criterion] = detected_grades
                                    criterion_data["sub_scores"] = {k: self._coerce_score(v) for k, v in detected_grades.items()}
                                else:
                                    sub_scores = criterion_data.get("sub_scores", {})
                                    criterion_data["sub_scores"] = {k: self._coerce_score(v) for k, v in sub_scores.items()} if sub_scores else {}

                                sub_scores = criterion_data.get("sub_scores", {})
                                if sub_scores:
                                    criterion_config = CRITERIA.get(criterion, {})
                                    sub_criteria_config = criterion_config.get("sub_criteria", {})

                                    weighted_score = 0.0
                                    total_weight = 0.0

                                    for sub_name, sub_score in sub_scores.items():
                                        if sub_name in sub_criteria_config:
                                            weight = sub_criteria_config[sub_name].get("weight", 0.0)
                                            weighted_score += sub_score * weight
                                            total_weight += weight

                                    if total_weight > 0:
                                        visual_response[criterion]["score"] = weighted_score / total_weight
                                    else:
                                        visual_response[criterion]["score"] = sum(sub_scores.values()) / len(sub_scores)
                                else:
                                    visual_response[criterion]["score"] = 5.0
                
                except Exception as e:
                    logger.error(f"Error in visual evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    visual_response = None
            
            # =========================================================================
            # STEP 5: Compute aesthetics metrics if enabled
            # =========================================================================
            if self.run_aesthetics and self.aesthetics_calculator:
                try:
                    logger.info(f"  Computing aesthetics metrics for {num_slides} slides...")
                    # Get the slide_images directory path
                    slide_images_dir = os.path.join(ppt_path, "slide_images")
                    
                    # Calculate metrics using the slide_images directory
                    full_aesthetics_metrics = self.aesthetics_calculator.calculate_presentation_metrics(
                        slide_images_dir if os.path.isdir(slide_images_dir) else images, 
                        aggregate=True,
                        metrics_to_compute=self.aesthetics_metrics,
                        n_workers=self.num_workers
                    )
                    
                    # Save full metrics to PPT's parent directory (ppt_path)
                    output_json_path = os.path.join(ppt_path, "aesthetics_metrics.json")
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(full_aesthetics_metrics, f, indent=2, ensure_ascii=False)
                    logger.info(f"  Aesthetics metrics saved to: {output_json_path}")
                    
                    # Only keep aggregated metrics for the result object to reduce memory
                    if full_aesthetics_metrics and "aggregated" in full_aesthetics_metrics:
                        aesthetics_summary = full_aesthetics_metrics.get("aggregated", {})
                        # Add some key per-slide metrics for analysis
                        aesthetics_summary["num_slides"] = full_aesthetics_metrics.get("num_slides", num_slides)
                        aesthetics_metrics = aesthetics_summary
                    else:
                        aesthetics_metrics = full_aesthetics_metrics
                    logger.info(f"  Aesthetics metrics computed successfully")
                except Exception as e:
                    logger.warning(f"  Failed to compute aesthetics metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    aesthetics_metrics = {"error": str(e)}
            
            # =========================================================================
            # STEP 6: Aggregate scores and create result object
            # =========================================================================
            
            # Extract scores from responses
            content_score = 0.0
            content_sub_scores = {}
            content_reason = ""
            
            if content_response and "Content" in content_response:
                content_score = content_response["Content"].get("score", 0.0)
                content_sub_scores = content_response["Content"].get("sub_scores", {})
                content_reason = content_response["Content"].get("reason", "")
            
            visual_design_score = 0.0
            visual_design_sub_scores = {}
            visual_design_reason = ""
            
            layout_score = 0.0
            layout_sub_scores = {}
            layout_reason = ""
            
            complexity_score = 0.0
            complexity_sub_scores = {}
            complexity_reason = ""
            
            overall_feedback = ""
            strengths = []
            improvements = []
            
            if visual_response:
                if "Visual_Design" in visual_response:
                    visual_design_score = visual_response["Visual_Design"].get("score", 0.0)
                    visual_design_sub_scores = visual_response["Visual_Design"].get("sub_scores", {})
                    visual_design_reason = visual_response["Visual_Design"].get("reason", "")
                
                if "Layout" in visual_response:
                    layout_score = visual_response["Layout"].get("score", 0.0)
                    layout_sub_scores = visual_response["Layout"].get("sub_scores", {})
                    layout_reason = visual_response["Layout"].get("reason", "")
                
                if "Complexity" in visual_response:
                    complexity_score = visual_response["Complexity"].get("score", 0.0)
                    complexity_sub_scores = visual_response["Complexity"].get("sub_scores", {})
                    complexity_reason = visual_response["Complexity"].get("reason", "")
                
                overall_feedback = visual_response.get("Overall_Feedback", "")
                strengths = visual_response.get("Top_Strengths", [])
                improvements = visual_response.get("Areas_for_Improvement", [])
            
            # Calculate aggregated scores
            raw_total = content_score + visual_design_score + layout_score + complexity_score
            
            scores = {
                "Content": content_score,
                "Visual_Design": visual_design_score,
                "Layout": layout_score,
                "Complexity": complexity_score
            }
            weighted_total = calculate_weighted_score(scores)
            
            # Extract quiz metrics
            quiz_accuracy = 0.0
            quiz_total_questions = 0
            quiz_correct_answers = 0
            quiz_details = None
            
            if quiz_result:
                quiz_accuracy = quiz_result.get("quiz_accuracy", 0.0)
                quiz_total_questions = quiz_result.get("quiz_total_questions", 0)
                quiz_correct_answers = quiz_result.get("quiz_correct_answers", 0)
                quiz_details = quiz_result.get("quiz_details", None)
            
            # Create result object
            result = EvaluationResult(
                product=product,
                difficulty=difficulty,
                topic=topic,
                ppt_id=ppt_id,
                ppt_path=ppt_path,
                num_slides=num_slides,
                
                content_score=content_score,
                visual_design_score=visual_design_score,
                layout_score=layout_score,
                complexity_score=complexity_score,
                
                content_sub_scores=content_sub_scores,
                visual_design_sub_scores=visual_design_sub_scores,
                layout_sub_scores=layout_sub_scores,
                complexity_sub_scores=complexity_sub_scores,

                visual_design_grades=visual_grades.get("Visual_Design"),
                layout_grades=visual_grades.get("Layout"),
                complexity_grades=visual_grades.get("Complexity"),
                
                content_reason=content_reason,
                visual_design_reason=visual_design_reason,
                layout_reason=layout_reason,
                complexity_reason=complexity_reason,
                
                weighted_total=weighted_total,
                raw_total=raw_total,
                
                overall_feedback=overall_feedback,
                strengths=strengths,
                improvements=improvements,
                
                quiz_accuracy=quiz_accuracy,
                quiz_total_questions=quiz_total_questions,
                quiz_correct_answers=quiz_correct_answers,
                quiz_details=quiz_details,
                
                aesthetics_metrics=aesthetics_metrics,
                
                evaluation_time=datetime.now().isoformat(),
                vlm_model=self.vlm.model_name if self.vlm else "none"
            )
            
            # Log results
            if self.run_vlm:
                logger.info(f"  Scores: Content={result.content_score:.1f}, Visual={result.visual_design_score:.1f}, "
                           f"Layout={result.layout_score:.1f}, Complexity={result.complexity_score:.1f} | "
                           f"Total={result.weighted_total:.2f}")
                if quiz_result:
                    logger.info(f"  Quiz: {quiz_accuracy:.1%} ({quiz_correct_answers}/{quiz_total_questions})")
            
            if self.run_aesthetics and aesthetics_metrics:
                # Log key aesthetics metrics dynamically based on what was computed
                metric_summary = []
                
                # Figure-ground contrast
                if "figure_ground_contrast" in aesthetics_metrics:
                    fgc = aesthetics_metrics["figure_ground_contrast"]
                    if isinstance(fgc, dict):
                        fgc_val = fgc.get("mean", "N/A")
                    else:
                        fgc_val = fgc
                    metric_summary.append(f"fg_contrast={fgc_val}")
                
                # Color harmony
                if "color_harmony" in aesthetics_metrics:
                    harmony = aesthetics_metrics["color_harmony"]
                    if isinstance(harmony, dict):
                        harmony_val = harmony.get("mean_distance", "N/A")
                    else:
                        harmony_val = harmony
                    metric_summary.append(f"harmony_dist={harmony_val}")
                
                # Colorfulness
                if "colorfulness" in aesthetics_metrics:
                    colorfulness = aesthetics_metrics["colorfulness"]
                    if isinstance(colorfulness, dict):
                        colorfulness_val = colorfulness.get("mean", "N/A")
                    else:
                        colorfulness_val = colorfulness
                    metric_summary.append(f"colorfulness={colorfulness_val}")
                
                
                # Subband entropy
                if "subband_entropy" in aesthetics_metrics:
                    entropy = aesthetics_metrics["subband_entropy"]
                    if isinstance(entropy, dict):
                        entropy_val = entropy.get("mean", "N/A")
                    else:
                        entropy_val = entropy
                    metric_summary.append(f"subband_entropy={entropy_val}")
                
                # Feature congestion
                if "feature_congestion" in aesthetics_metrics:
                    congestion = aesthetics_metrics["feature_congestion"]
                    if isinstance(congestion, dict):
                        congestion_val = congestion.get("clutter_mean", "N/A")
                    else:
                        congestion_val = congestion
                    metric_summary.append(f"feature_congestion={congestion_val}")
                
                if metric_summary:
                    logger.info(f"  Aesthetics: {', '.join(metric_summary)}")
                else:
                    logger.info(f"  Aesthetics: computed but no summary available")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {ppt_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_all(
        self,
        products: List[str] = None,
        difficulties: List[str] = None,
        topics: List[str] = None,
        limit: int = None,
        use_cache: bool = False,
        output_file: str = None
    ) -> List[EvaluationResult]:
        """
        Evaluate all PPTs matching the given filters, with progress bar and skip logging.
        """
        logger.info(f"Evaluation filters - products: {products}, difficulties: {difficulties}, topics: {topics}, limit: {limit}")
        ppts_to_evaluate = self._discover_ppts(products, difficulties, topics, limit, use_cache)
        if not ppts_to_evaluate['to_evaluate'] and not ppts_to_evaluate['cached']:
            logger.warning("No PPTs found to evaluate")
            return []
        results = []
        cached_count = len(ppts_to_evaluate['cached'])
        quiz_recalc_count = 0
        
        for cached_info in ppts_to_evaluate['cached']:
            cached_result = self.get_cached_result(
                cached_info['product'],
                cached_info['difficulty'],
                cached_info['topic'],
                cached_info['ppt_id']
            )
            
            # Check if quiz needs recalculation
            if not self.run_aesthetics and self.needs_quiz_recalculation(cached_result):
                logger.info(f"  Quiz recalculation needed for {cached_info['product']}/{cached_info['difficulty']}/{cached_info['topic']}/{cached_info['ppt_id']}")
                cached_result = self.recalculate_quiz_for_cached(
                    cached_result,
                    cached_info['ppt_path'],
                    cached_info['topic']
                )
                quiz_recalc_count += 1
            else:
                logger.info(f"  Using cached result for {cached_info['product']}/{cached_info['difficulty']}/{cached_info['topic']}/{cached_info['ppt_id']}")
            
            results.append(cached_result)
        to_eval = ppts_to_evaluate['to_evaluate']
        if not to_eval:
            logger.info(f"All {cached_count} results loaded from cache")
            return results
        logger.info(f"Found {len(to_eval)} PPTs to evaluate, {cached_count} from cache")
        logger.info(f"Using {self.num_workers} parallel worker(s)")
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.eval_mode == "aesthetics_only":
                output_file = f"aesthetics_results_{timestamp}.json"
            else:
                output_file = f"quantitative_results_{timestamp}.json"
            logger.info(f"No output file specified, using default: {output_file}")
        self._last_output_file = output_file
        # Progress bar for the whole process
        skipped_count = 0
        new_results = []
        with tqdm(total=len(to_eval), desc="Quantitative Evaluation", ncols=80) as pbar:
            for ppt_info in to_eval:
                try:
                    quiz_data = self.get_quiz_data(ppt_info['topic'])
                    result = self.evaluate_ppt(
                        ppt_info['product'],
                        ppt_info['difficulty'],
                        ppt_info['topic'],
                        ppt_info['ppt_id'],
                        ppt_info['ppt_path'],
                        quiz_data=quiz_data
                    )
                    if result:
                        new_results.append(result)
                        # Save incrementally
                        if output_file:
                            all_results = results + new_results
                            self.save_results(all_results, output_file)
                except Exception as e:
                    msg = str(e)
                    if (
                        "inappropriate content" in msg.lower() or
                        "data_inspection_failed" in msg.lower() or
                        ("Error code: 400" in msg and "inappropriate content" in msg)
                    ):
                        log_skipped_item("ppt", f"{ppt_info['product']}/{ppt_info['difficulty']}/{ppt_info['topic']}/{ppt_info['ppt_id']}", f"Inappropriate content: {msg}")
                        skipped_count += 1
                        logger.warning(f"Skipped {ppt_info['ppt_path']} due to inappropriate content.")
                    else:
                        log_skipped_item("ppt", f"{ppt_info['product']}/{ppt_info['difficulty']}/{ppt_info['topic']}/{ppt_info['ppt_id']}", f"Other error: {msg}")
                        logger.error(f"Skipped {ppt_info['ppt_path']} due to error: {msg}")
                pbar.update(1)
        results.extend(new_results)
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} items due to inappropriate content or errors. See {SKIP_LOG_FILE}.")
        
        log_msg = f"Evaluation complete: {len(results)} total, {cached_count} from cache"
        if quiz_recalc_count > 0:
            log_msg += f" ({quiz_recalc_count} with quiz recalculated)"
        log_msg += f", {len(new_results)} newly evaluated"
        logger.info(log_msg)
        return results
    
    def _discover_ppts(
        self,
        products: List[str] = None,
        difficulties: List[str] = None,
        topics: List[str] = None,
        limit: int = None,
        use_cache: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        Discover all PPTs to evaluate and separate cached from non-cached.
        
        Args:
            products: List of products to include (None for all)
            difficulties: List of difficulties to include (None for all)
            topics: List of topics to include (None for all)
            limit: Maximum number of PPTs to evaluate **per product** (None for no limit)
            use_cache: If True, separate cached PPTs
            
        Returns:
            Dict with 'to_evaluate' and 'cached' lists of PPT info dicts
        """
        # Create cache check function if caching is enabled
        cache_check_fn = None
        if use_cache:
            cache_check_fn = self.is_cached
        
        return discover_ppts(
            ppt_gen_root=self.ppt_gen_root,
            products=products,
            difficulties=difficulties,
            topics=topics,
            limit=limit,
            group_by_topic=False,
            cache_check_fn=cache_check_fn
        )
    
    def _evaluate_sequential(
        self,
        ppts: List[Dict],
        output_file: str,
        existing_results: List
    ) -> List[EvaluationResult]:
        """
        Evaluate PPTs sequentially (original behavior).
        
        Args:
            ppts: List of PPT info dicts to evaluate
            output_file: Output file for incremental saves
            existing_results: Existing results (for incremental saves)
            
        Returns:
            List of new EvaluationResult objects
        """
        new_results = []
        
        for i, ppt_info in enumerate(ppts):
            logger.info(f"[{i+1}/{len(ppts)}] Evaluating: {ppt_info['product']}/{ppt_info['topic']}/{ppt_info['ppt_id']}")
            
            # Get quiz data for this topic
            quiz_data = self.get_quiz_data(ppt_info['topic'])
            
            result = self.evaluate_ppt(
                ppt_info['product'],
                ppt_info['difficulty'],
                ppt_info['topic'],
                ppt_info['ppt_id'],
                ppt_info['ppt_path'],
                quiz_data=quiz_data
            )
            
            if result:
                new_results.append(result)
                # Save incrementally
                if output_file:
                    all_results = existing_results + new_results
                    self.save_results(all_results, output_file)
                    logger.info(f"  Saved incremental results ({len(all_results)} total)")
        
        return new_results
    
    def _evaluate_parallel(
        self,
        ppts: List[Dict],
        output_file: str,
        existing_results: List
    ) -> List[EvaluationResult]:
        """
        Evaluate PPTs in parallel using ThreadPoolExecutor.
        
        Args:
            ppts: List of PPT info dicts to evaluate
            output_file: Output file for incremental saves
            existing_results: Existing results (for incremental saves)
            
        Returns:
            List of new EvaluationResult objects
        """
        new_results = []
        results_lock = threading.Lock()
        completed_count = 0
        
        def evaluate_single(ppt_info: Dict) -> Optional[EvaluationResult]:
            """Worker function for parallel evaluation."""
            nonlocal completed_count
            
            try:
                # Get quiz data for this topic
                quiz_data = self.get_quiz_data(ppt_info['topic'])
                
                result = self.evaluate_ppt(
                    ppt_info['product'],
                    ppt_info['difficulty'],
                    ppt_info['topic'],
                    ppt_info['ppt_id'],
                    ppt_info['ppt_path'],
                    quiz_data=quiz_data
                )
                
                with results_lock:
                    completed_count += 1
                    logger.info(f"[{completed_count}/{len(ppts)}] Completed: {ppt_info['product']}/{ppt_info['topic']}/{ppt_info['ppt_id']}")
                
                return result
            except Exception as e:
                logger.error(f"Error evaluating {ppt_info['ppt_id']}: {e}")
                return None
        
        logger.info(f"Starting parallel evaluation with {self.num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_ppt = {executor.submit(evaluate_single, ppt): ppt for ppt in ppts}
            
            # Process results as they complete
            for future in as_completed(future_to_ppt):
                ppt_info = future_to_ppt[future]
                try:
                    result = future.result()
                    if result:
                        with results_lock:
                            new_results.append(result)
                            # Save incrementally
                            if output_file:
                                all_results = existing_results + new_results
                                self.save_results(all_results, output_file)
                except Exception as e:
                    logger.error(f"Future failed for {ppt_info['ppt_id']}: {e}")
        
        logger.info(f"Parallel evaluation complete: {len(new_results)} successful out of {len(ppts)}")
        return new_results
    
    def save_results(
        self,
        results: List,
        filename: str = None
    ) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: List of EvaluationResult objects or dicts (from cache)
            filename: Output filename (default: timestamped)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.eval_mode == "aesthetics_only":
                filename = f"aesthetics_results_{timestamp}.json"
            else:
                filename = f"quantitative_results_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert results to dictionaries (handle both EvaluationResult and dict)
        results_dict = []
        for r in results:
            if hasattr(r, 'to_dict'):
                results_dict.append(r.to_dict())
            elif isinstance(r, dict):
                results_dict.append(r)
            else:
                logger.warning(f"Unknown result type: {type(r)}")
        
        # Add metadata (handle case when VLM is disabled)
        output_data = {
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),
                "vlm_model": self.vlm.model_name if self.vlm else "none",
                "vlm_provider": self.vlm.provider if self.vlm else "none",
                "eval_mode": self.eval_mode,
                "num_evaluations": len(results),
                "criteria": list(CRITERIA.keys()) if self.run_vlm else []
            },
            "results": results_dict,
            "summary": self._compute_summary(results)
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def _compute_summary(self, results: List) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        if not results:
            return {}
        
        # Helper to get attribute from either EvaluationResult or dict
        def get_attr(r, attr):
            if hasattr(r, attr):
                return getattr(r, attr)
            elif isinstance(r, dict):
                return r.get(attr, 0)
            return 0
        
        # Group by product
        product_scores = {}
        for r in results:
            product = get_attr(r, 'product')
            weighted_total = get_attr(r, 'weighted_total')
            if product not in product_scores:
                product_scores[product] = []
            product_scores[product].append(weighted_total)
        
        # Calculate averages
        product_averages = {
            prod: sum(scores) / len(scores) 
            for prod, scores in product_scores.items()
        }
        
        # Overall statistics
        all_totals = [get_attr(r, 'weighted_total') for r in results]
        
        summary = {
            "overall": {
                "mean": sum(all_totals) / len(all_totals),
                "min": min(all_totals),
                "max": max(all_totals),
                "count": len(all_totals)
            },
            "by_product": product_averages,
            "by_criterion": {
                "Content": sum(get_attr(r, 'content_score') for r in results) / len(results),
                "Visual_Design": sum(get_attr(r, 'visual_design_score') for r in results) / len(results),
                "Layout": sum(get_attr(r, 'layout_score') for r in results) / len(results),
                "Complexity": sum(get_attr(r, 'complexity_score') for r in results) / len(results)
            }
        }
        
        # Aggregate aesthetics metrics if available
        aesthetics_summary = self._aggregate_aesthetics_metrics(results)
        if aesthetics_summary:
            summary["aesthetics"] = aesthetics_summary
        
        return summary
    
    def _aggregate_aesthetics_metrics(self, results: List) -> Optional[Dict[str, Any]]:
        """Aggregate aesthetics metrics across all results."""
        if not results:
            return None
        
        # Helper to get attribute from either EvaluationResult or dict
        def get_attr(r, attr):
            if hasattr(r, attr):
                return getattr(r, attr)
            elif isinstance(r, dict):
                return r.get(attr)
            return None
        
        # Collect aesthetics metrics from all results
        all_metrics = []
        for r in results:
            metrics = get_attr(r, 'aesthetics_metrics')
            if metrics and not metrics.get('error'):
                all_metrics.append(metrics)
        
        if not all_metrics:
            return None
        
        aggregated = {}
        
        # Aggregate figure-ground contrast
        fgc_values = []
        for m in all_metrics:
            fgc = m.get('figure_ground_contrast')
            if fgc and isinstance(fgc, dict) and 'mean' in fgc:
                fgc_values.append(fgc['mean'])
        if fgc_values:
            aggregated['figure_ground_contrast'] = {
                'mean': float(np.mean(fgc_values)),
                'std': float(np.std(fgc_values)),
            }
        
        # Aggregate color harmony
        harmony_values = []
        for m in all_metrics:
            ch = m.get('color_harmony')
            if ch and isinstance(ch, dict) and 'mean_distance' in ch:
                harmony_values.append(ch['mean_distance'])
        if harmony_values:
            aggregated['color_harmony'] = {
                'mean_distance': float(np.mean(harmony_values)),
                'std_distance': float(np.std(harmony_values)),
            }
        
        # Aggregate colorfulness
        colorfulness_values = []
        for m in all_metrics:
            cf = m.get('colorfulness')
            if cf and isinstance(cf, dict) and 'mean' in cf:
                colorfulness_values.append(cf['mean'])
        if colorfulness_values:
            aggregated['colorfulness'] = {
                'mean': float(np.mean(colorfulness_values)),
                'std': float(np.std(colorfulness_values)),
            }
        
        # Aggregate visual complexity
        complexity_values = []
        for m in all_metrics:
            vc = m.get('visual_complexity')
            if vc and isinstance(vc, dict) and 'mean_complexity' in vc:
                complexity_values.append(vc['mean_complexity'])
        if complexity_values:
            aggregated['visual_complexity'] = {
                'mean': float(np.mean(complexity_values)),
                'std': float(np.std(complexity_values)),
            }
        
        aggregated['num_presentations_with_metrics'] = len(all_metrics)
        
        return aggregated if len(aggregated) > 1 else None


def main():
    """Main entry point for quantitative evaluation."""
    parser = argparse.ArgumentParser(
        description="Quantitative PPT Evaluation using VLMs"
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
        help="Specific model name (default: provider's default)"
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
        "--products",
        nargs="+",
        choices=["Gamma", "NotebookLM", "Kimi-Standard", "Kimi-Smart", "Kimi-Banana", 
                 "Skywork","Skywork-Banana", "Zhipu", "Quake"],
        default=None,
        help="Products to evaluate (default: all)"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=["topic_introduction", "work_report", "business_plan","brand_promote","personal_statement","product_launch","course_preparation"],
        default=["topic_introduction", "work_report", "business_plan","brand_promote","personal_statement","product_launch","course_preparation"],
        help="Difficulties to evaluate (default: all)"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Topics to evaluate (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PPTs to evaluate"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images per PPT"
    )
    parser.add_argument(
        "--max-images-per-call",
        type=int,
        default=None,
        help="Maximum images per VLM call (for batching)"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename"
    )
    parser.add_argument(
        "--list-products",
        action="store_true",
        help="List all available products and exit"
    )
    parser.add_argument(
        "--source-mode",
        choices=["topic", "scene", "all"],
        default="all",
        help="Source content mode: 'topic' for topic_introduction test (topic_introduction/{topic}/), 'scene' for scene test (scene/{topic}/), or 'all' for both"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached results if available, only evaluate missing PPTs"
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Path to cache file for incremental evaluation (default: {output_dir}/{output_file})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for VLM calls (default: 1 for sequential). "
             "Recommended: 2-4 for most API providers to avoid rate limits."
    )
    parser.add_argument(
        "--no-aesthetics",
        action="store_true",
        help="Disable computation of image-based aesthetics metrics (faster evaluation)"
    )
    parser.add_argument(
        "--aesthetics-metrics",
        default=None,
        help=(
            "Comma-separated aesthetics metrics to compute. "
            "Example: lab,hsv,visual_complexity,subband_entropy,colorfulness,figure_ground_contrast. "
            "If omitted, computes all aesthetics metrics."
        ),
    )
    parser.add_argument(
        "--aesthetics-config",
        default=None,
        help=(
            "Path to JSON config file for aesthetics score parameters. "
            "The config controls weights for contrast, colorfulness, harmony, entropy, and RMSSD. "
            "Example: aesthetics_config/aesthetics_config.json"
        ),
    )
    parser.add_argument(
        "--eval-mode",
        choices=["full", "vlm_only", "content_only", "visual_only", "aesthetics_only"],
        default="full",
        help=(
            "Evaluation mode: 'full' (VLM + aesthetics), 'vlm_only' (legacy content + visual evaluation), "
            "'content_only' (text-based VLM evaluation only), 'visual_only' (image-only evaluation), "
            "'aesthetics_only' (image-based metrics only, no VLM calls)"
        )
    )
    parser.add_argument(
        "--use-grids",
        action="store_true",
        help="Use slide grids to reduce VLM token consumption. Grids are cached in slide_images_grid folder."
    )
    parser.add_argument(
        "--visual-grade-mode",
        action="store_true",
        help="Request bad/moderate/good grades for visual evaluation and map them to scores internally."
    )
    parser.add_argument(
        "--target-folder",
        default=None,
        help="Path to a target folder whose immediate subfolders will be processed as individual PPTs (no product/difficulty parsing)."
    )
    
    args = parser.parse_args()
    
    # List products if requested
    if args.list_products:
        print("Available products:")
        for prod, config in PRODUCTS.items():
            print(f"  {prod}: {config.get('description', 'No description')}")
            print(f"      Directory: {config.get('directory')}")
            print(f"      Format: {config.get('input_format')}")
        return
    
    if args.source_mode == "topic":
        args.difficulties = ["topic_introduction"]
    elif args.source_mode == "scene":
        args.difficulties = ["work_report", "business_plan","brand_promote","personal_statement","product_launch","course_preparation"]
    elif args.source_mode == "all":
        args.difficulties = ["topic_introduction", "work_report", "business_plan","brand_promote","personal_statement","product_launch","course_preparation"]

    # Initialize evaluator
    evaluator = QuantitativeEvaluator(
        vlm_provider=args.provider,
        vlm_model=args.model,
        ppt_gen_root=args.ppt_root,
        output_dir=args.output_dir,
        max_images=args.max_images,
        max_images_per_call=args.max_images_per_call,
        source_mode=args.source_mode,
        num_workers=args.workers,
        compute_aesthetics=not args.no_aesthetics,
        aesthetics_metrics=args.aesthetics_metrics,
        aesthetics_config=args.aesthetics_config,
        eval_mode=args.eval_mode,
        use_grids=args.use_grids,
        visual_grade_mode=args.visual_grade_mode
    )
    
    # Load cache if requested
    if args.use_cache:
        cache_path = args.cache_file
        if not cache_path and args.output_file:
            # Use the specified output file as cache
            output_dir = args.output_dir or EvalConfig.OUTPUT_DIR
            cache_path = os.path.join(output_dir, args.output_file)
        elif not cache_path:
            # Try to find latest cache file matching eval mode
            output_dir = args.output_dir or EvalConfig.OUTPUT_DIR
            if args.eval_mode == "aesthetics_only":
                pattern = "aesthetics_results_*.json"
            else:
                pattern = "quantitative_results_*.json"
            
            cache_files = glob.glob(os.path.join(output_dir, pattern))
            if cache_files:
                cache_path = max(cache_files, key=os.path.getmtime)
                logger.info(f"Auto-detected latest cache file: {cache_path}")
        
        if cache_path:
            cached_count = evaluator.load_cache(cache_path)
            if cached_count > 0:
                print(f"Loaded {cached_count} cached results from: {cache_path}")
    
    # If a target folder is provided, process its subfolders directly
    results = []
    if args.target_folder:
        target = args.target_folder
        if not os.path.exists(target):
            logger.error(f"Target folder does not exist: {target}")
            return

        entries = sorted(os.listdir(target))
        logger.info(f"Processing target folder: {target} ({len(entries)} entries)")

        for name in entries:
            subpath = os.path.join(target, name)
            if not os.path.isdir(subpath):
                continue

            # Use the subfolder name as ppt_id and topic. product/difficulty set to generic values.
            ppt_id = name
            topic = name
            product = os.path.basename(target) or "target"
            difficulty = "unknown"

            try:
                res = evaluator.evaluate_ppt(
                    product=product,
                    difficulty=difficulty,
                    topic=topic,
                    ppt_id=ppt_id,
                    ppt_path=subpath,
                    quiz_data=None
                )
                if res:
                    results.append(res)
                else:
                    log_skipped_item("ppt", subpath, "evaluation_failed_or_no_images")
            except Exception as e:
                logger.exception(f"Failed to evaluate {subpath}: {e}")
                log_skipped_item("ppt", subpath, f"exception:{e}")

        # Save results for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_file:
            out_file = args.output_file
        else:
            if args.eval_mode == "aesthetics_only":
                out_file = f"aesthetics_results_target_{product}_{timestamp}.json"
            else:
                out_file = f"quantitative_results_target_{product}_{timestamp}.json"

        os.makedirs(evaluator.output_dir, exist_ok=True)
        if results:
            evaluator.save_results(results, out_file)
            logger.info(f"Saved {len(results)} results for target folder to: {os.path.join(evaluator.output_dir, out_file)}")
        else:
            logger.warning("No results produced for target folder processing")

    else:
        # Run normal discovery-based evaluation (with incremental saving)
        results = evaluator.evaluate_all(
            products=args.products,
            difficulties=args.difficulties,
            topics=args.topics,
            limit=args.limit,
            use_cache=args.use_cache,
            output_file=args.output_file  # Enable incremental saving
        )
    
    # Use the same output file from evaluate_all to avoid duplicates
    final_output_file = args.output_file if args.output_file else getattr(evaluator, '_last_output_file', None)
    
    if not final_output_file:
        # Fallback: create new filename (shouldn't happen if evaluate_all ran)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.eval_mode == "aesthetics_only":
            final_output_file = f"aesthetics_results_{timestamp}.json"
        else:
            final_output_file = f"quantitative_results_{timestamp}.json"
    
    # Construct full path for reporting
    output_dir = args.output_dir or EvalConfig.OUTPUT_DIR
    output_path = os.path.join(output_dir, final_output_file)
    
    # Only do a final save if results weren't incrementally saved
    # (e.g., all results came from cache, or no evaluations were performed)
    if results and not os.path.exists(output_path):
        output_path = evaluator.save_results(results, final_output_file)
        logger.info(f"Saved final results to: {output_path}")
    else:
        logger.info(f"Results already saved: {output_path}")
    
    print(f"\n{'='*60}")
    if args.eval_mode == "aesthetics_only":
        print(f"Aesthetics evaluation complete!")
    else:
        print(f"Quantitative evaluation complete!")
    print(f"  Eval mode: {args.eval_mode}")
    print(f"  Source mode: {args.source_mode}")
    print(f"  Evaluated: {len(results)} presentations")
    
    # Print summary by product (handle both EvaluationResult and dict)
    product_counts = {}
    for r in results:
        product = r.product if hasattr(r, 'product') else r.get('product', 'Unknown')
        product_counts[product] = product_counts.get(product, 0) + 1
    
    if product_counts:
        print(f"  By product:")
        for prod, count in sorted(product_counts.items()):
            print(f"    {prod}: {count}")
    
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
