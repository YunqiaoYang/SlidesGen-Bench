
# Evaluation Criteria Definitions
# Comprehensive PPT Evaluation Framework with Detailed Sub-criteria

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

class EvalConfig:
    """Central configuration for the evaluation system."""
    
    # Paths (can be overridden via environment variables)
    PPT_GEN_ROOT = os.getenv("PPT_GEN_ROOT", "./Generated_Slides")
    PPT_EVAL_ROOT = os.getenv("PPT_EVAL_ROOT", "/dataset")
    OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "results"))
    
    # Source content paths
    SOURCE_CONTENT_ROOT = os.getenv("SOURCE_CONTENT_ROOT", "./dataset")
    
    # VLM Configuration
    DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL", "gemini-3-flash-preview")
    VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "4096"))
    VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.1"))
    
    # ELO Configuration
    ELO_INITIAL_RATING = 1000
    ELO_K_FACTOR = 32
    
    # Evaluation Settings
    MAX_IMAGES_PER_PPT = int(os.getenv("MAX_IMAGES_PER_PPT", "40"))  # Reduced from 20 to avoid overload
    MAX_IMAGES_PER_VLM_CALL = int(os.getenv("MAX_IMAGES_PER_VLM_CALL", "8"))  # Max images in single VLM call
    SOURCE_CONTENT_MAX_CHARS = int(os.getenv("SOURCE_CONTENT_MAX_CHARS", "8000"))


# =============================================================================
# PRODUCT DEFINITIONS
# =============================================================================

# Supported PPT generation products
PRODUCTS = {
    "Gamma": {
        "directory": "Gamma",
        "input_format": "pptx",
        "description": "Gamma AI presentation generator",
        "structure": "difficulty/topic/name",
    },
    "NotebookLM": {
        "directory": "NotebookLM", 
        "input_format": "pdf",  # NotebookLM outputs PDF files
        "description": "Google NotebookLM presentation generator",
        "structure": "difficulty/topic",
    },
    "Kimi-Standard": {
        "directory": "Kimi/Standard",
        "input_format": "pptx",
        "description": "Kimi Standard mode presentation generator",
        "structure": "difficulty/topic/name",
        "has_slide_images": True,  # Already extracted
    },
    "Kimi-Smart": {
        "directory": "Kimi/Smart",
        "input_format": "pptx",
        "description": "Kimi Smart mode presentation generator",
        "structure": "difficulty/topic/name",
        "has_slide_images": True,  # Already extracted
    },
    "Kimi-Banana": {
        "directory": "Kimi/Banana",
        "input_format": "pptx",
        "description": "Kimi Banana mode presentation generator",
        "structure": "difficulty/topic/name",
        "has_slide_images": True,  # Already extracted
    },
    "Skywork": {
        "directory": "Skyworks",
        "input_format": "pptx",
        "description": "Skywork presentation generator",
        "structure": "difficulty/topic/name",
    },
    "Skywork-Banana": {
        "directory": "Skyworks/Banana",
        "input_format": "pptx",
        "description": "Skywork Banana mode presentation generator",
        "structure": "difficulty/topic",
    },
    "Zhipu": {
        "directory": "Zhipu",
        "input_format": "pptx",
        "description": "Zhipu AI presentation generator",
        "structure": "difficulty/topic/name",
    },
    "Quake": {
        "directory": "Quake",
        "input_format": "pptx",
        "description": "Quake presentation generator",
        "structure": "difficulty/topic/name",
    },
}

# Get list of all product names
def get_all_products() -> List[str]:
    """Get list of all supported product names."""
    return list(PRODUCTS.keys())

def get_product_path(product: str, ppt_gen_root: str = None) -> str:
    """Get full path to a product's directory."""
    root = ppt_gen_root or EvalConfig.PPT_GEN_ROOT
    if product in PRODUCTS:
        return os.path.join(root, PRODUCTS[product]["directory"])
    return os.path.join(root, product)

def get_product_config(product: str) -> Dict:
    """Get configuration for a specific product."""
    return PRODUCTS.get(product, {
        "directory": product,
        "input_format": "pptx",
        "description": f"Unknown product: {product}",
        "structure": "difficulty/topic/id",
    })

# =============================================================================
# DETAILED EVALUATION CRITERIA
# =============================================================================

CRITERIA = {
    "Content": {
    "description": "Evaluates the integrity, structure, and pacing of the presentation content, focusing on factual fidelity, logical progression, and cognitive load management.",
    "max_score": 10,
    "weight": 0.30,
    "sub_criteria": {
        "Accuracy_and_Completeness": {
            "description": "A unified metric assessing whether the presentation accurately represents the source material without factual errors while simultaneously covering all essential points and necessary depth. Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.40,
            "scoring_rubric": {
                "0-2": "Critical failure: Contains major factual errors or significant misrepresentations; omits the majority of key points.",
                "3-4": "Weak: Several noticeable errors exist; covers surface-level info but misses essential details or context.",
                "5-6": "Average: Generally accurate with only minor imprecisions; covers main points but lacks comprehensive depth.",
                "7-8": "Strong: Highly accurate with trivial discrepancies; comprehensive coverage with very few minor omissions.",
                "9-10": "Flawless: Perfectly accurate representation of source content; covers all essential points with appropriate depth and fidelity."
            }
        },
        "Logical_Flow": {
            "description": "Is the content organized in a logical sequence? Do slides build upon each other coherently to support a narrative? Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.30,
            "scoring_rubric": {
                "0-2": "Chaotic organization, no discernible logic or narrative structure.",
                "3-4": "Poor flow, frequent jumps between unrelated topics or confusing ordering.",
                "5-6": "Acceptable flow with some awkward transitions or ordering issues.",
                "7-8": "Good logical progression with smooth transitions between concepts.",
                "9-10": "Excellent narrative flow; a perfectly structured argument that guides the audience effortlessly."
            }
        },
        "Cognitive_Rhythm": {
            "description": "Evaluates the oscillation of information density (The 'Heartbeat'). Does the presentation vary between High Density (Complexity/New Info) and Low Density (Recovery/Status Quo) to manage Cognitive Load? Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.30,
            "scoring_rubric": {
                "0-2": "Flatline (Dead): Monotonous intensity. Either a 'flat high line' causing Cognitive Overload (every slide is dense/complex) or a 'flat low line' (no substance).",
                "3-4": "Arrhythmic: Little variation in density; fails to provide necessary 'recovery periods' after complex data.",
                "5-6": "Weak Pulse: Some variation in slide density, but the transition between complex and simple concepts feels accidental rather than strategic.",
                "7-8": "Healthy Pulse: Good 'Sawtooth' pattern; introduces complex information (Intrinsic Load) followed by simpler slides to aid memory transfer.",
                "9-10": "Resonant Sparkline: Perfect oscillation between 'What is' and 'What could be.' Strategic manipulation of information density prevents habituation and optimizes working memory capacity."
            }
        }
    }
    },
    "Visual_Design": {
    "description": "Evaluates the aesthetic quality, visual consistency, and professional appearance of the presentation.",
    "max_score": 10,
    "weight": 0.30,
    "sub_criteria": {
        "Color_Scheme": {
            "description": "Evaluates the application of color theory, focusing on the harmonic balance of the palette, contrast ratios, and how well the colors complement each other to create a unified aesthetic. Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.20,
            "scoring_rubric": {
                "0-2": "Dissonant or clashing colors that create visual tension; lack of harmonic balance.",
                "3-4": "Weak application of color theory; combinations feel accidental rather than designed.",
                "5-6": "Standard color usage; acceptable balance but lacks distinct harmonization.",
                "7-8": "Pleasing aesthetic with well-balanced, complementary tones that work together seamlessly.",
                "9-10": "Masterful application of color theory; sophisticated harmonies that elevate the content and evoke the correct mood."
            }
        },
        "Typography": {
            "description": "Are fonts readable, consistent, and appropriately sized? Is text hierarchy clear? Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.20,
            "scoring_rubric": {
                "0-2": "Unreadable fonts, inconsistent sizes, poor hierarchy",
                "3-4": "Readability issues, too many font styles",
                "5-6": "Adequate typography with minor issues",
                "7-8": "Clean typography with clear hierarchy",
                "9-10": "Excellent typography choices enhancing communication"
            }
        },
        "Visual_Consistency": {
            "description": "Examines the visual thread tying the presentation together, specifically focusing on the rigorous maintenance of color coherence, recurring design motifs, and layout stability across the entire narrative. Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.20,
            "scoring_rubric": {
                "0-2": "Fragmented visual identity; random color usage per slide with no logical thread.",
                "3-4": "Disjointed appearance; frequent breaks in color continuity or style shifting.",
                "5-6": "Generally unified, though minor deviations in color application or spacing exist.",
                "7-8": "Strong cohesive flow; color palette and design elements are strictly adhered to throughout.",
                "9-10": "Flawless holistic coherence; a unified visual system where every slide feels like part of an inseparable whole."
            }
        },
        "Image_Quality": {
            "description": "Are images high quality, relevant, and properly integrated? Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.20,
            "scoring_rubric": {
                "0-2": "Low quality, pixelated, or irrelevant images",
                "3-4": "Some image quality or relevance issues",
                "5-6": "Acceptable images that serve their purpose",
                "7-8": "High quality, relevant images well integrated",
                "9-10": "Exceptional imagery that significantly enhances the presentation"
            }
        },
        "Theme_Appropriateness": {
            "description": "Does the visual theme match the content topic and intended audience? Score must be an integer from 0-10.",
            "max_score": 10,
            "weight": 0.20,
            "scoring_rubric": {
                "0-2": "Theme completely inappropriate for content/audience",
                "3-4": "Theme partially mismatched",
                "5-6": "Theme is neutral/acceptable",
                "7-8": "Theme well-suited to content and audience",
                "9-10": "Theme perfectly captures the essence of the content"
            }
        }
    }
    },
    "Layout": {
        "description": "Evaluates the spatial organization, balance, and readability of slide layouts.",
        "max_score": 10,
        "weight": 0.20,
        "sub_criteria": {
            "Spatial_Balance": {
                "description": "Is whitespace used effectively? Are elements balanced on each slide? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.4,
                "scoring_rubric": {
                    "0-2": "Cluttered or overly sparse, poor balance",
                    "3-4": "Unbalanced layout, ineffective use of space",
                    "5-6": "Acceptable spacing and balance",
                    "7-8": "Well-balanced with effective whitespace",
                    "9-10": "Masterful use of space creating visual harmony"
                }
            },
            "Element_Alignment": {
                "description": "Are text, images, and other elements properly aligned? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.3,
                "scoring_rubric": {
                    "0-2": "Elements randomly placed, no alignment",
                    "3-4": "Noticeable misalignments",
                    "5-6": "Generally aligned with minor issues",
                    "7-8": "Clean alignment throughout",
                    "9-10": "Pixel-perfect alignment with professional grid system"
                }
            },
            "No_Overlapping": {
                "description": "Are there any elements that overlap, obscure each other, or extend beyond boundaries? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.3,
                "scoring_rubric": {
                    "0-2": "Major overlapping issues, content obscured",
                    "3-4": "Several elements overlap or are cut off",
                    "5-6": "Minor overlapping or boundary issues",
                    "7-8": "No significant overlapping issues",
                    "9-10": "Perfect element separation with clear boundaries"
                }
            },
        }
    },
    "Complexity": {
        "description": "Evaluates the sophistication and richness of visual and structural elements.",
        "max_score": 10,
        "weight": 0.20,
        "sub_criteria": {
            "Charts_and_Data": {
                "description": "Does the presentation include charts, graphs, or data visualizations where appropriate? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.25,
                "scoring_rubric": {
                    "0-2": "No data visualization despite data-heavy content",
                    "3-4": "Basic text-only data representation",
                    "5-6": "Simple charts present where needed",
                    "7-8": "Well-designed charts and data visualizations",
                    "9-10": "Sophisticated, insightful data visualizations"
                }
            },
            "Visual_Elements": {
                "description": "Are there icons, illustrations, diagrams, flowcharts, or infographics? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.25,
                "scoring_rubric": {
                    "0-2": "Text-only slides, no visual elements",
                    "3-4": "Minimal visual elements",
                    "5-6": "Adequate use of icons or simple graphics",
                    "7-8": "Rich visual elements enhancing understanding",
                    "9-10": "Exceptional use of visual elements including custom diagrams"
                }
            },
            "Advanced_Design": {
                "description": "Does the presentation use gradients, shadows, animations, or advanced design techniques? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.25,
                "scoring_rubric": {
                    "0-2": "Flat, basic design with no advanced techniques",
                    "3-4": "Minimal use of advanced design elements",
                    "5-6": "Some advanced design techniques applied",
                    "7-8": "Skillful use of gradients, shadows, depth",
                    "9-10": "Masterful advanced design creating visual depth and interest"
                }
            },
            "Layout_Variety": {
                "description": "Does the presentation use varied layouts appropriate for different content types? Score must be an integer from 0-10.",
                "max_score": 10,
                "weight": 0.25,
                "scoring_rubric": {
                    "0-2": "Same basic layout repeated throughout",
                    "3-4": "Minimal layout variety",
                    "5-6": "Some layout variations",
                    "7-8": "Good variety of layouts suited to content",
                    "9-10": "Creative, diverse layouts perfectly matched to content"
                }
            }
        }
    }
}

# Simplified criteria mapping for backward compatibility
CRITERIA_SIMPLE = {
    "Content": CRITERIA["Content"],
    "Style": CRITERIA["Visual_Design"],  # Renamed from Style to Visual_Design
    "Layout": CRITERIA["Layout"],
    "Complexity": CRITERIA["Complexity"],
}

# =============================================================================
# VLM PROMPTS - QUANTITATIVE EVALUATION
# =============================================================================

QUANTITATIVE_EVAL_PROMPT_TEMPLATE = """
You are an expert presentation designer and content evaluator with deep expertise in visual communication and presentation design.
Your task is to evaluate a PowerPoint presentation based on the provided images of its slides and the original source document.

**Source Document Summary/Content:**
{document_content}

**Presentation Topic:**
{topic}

**Number of Slides:** {num_slides}

---

## EVALUATION CRITERIA

Please evaluate the presentation on the following criteria. For each criterion, provide:
1. Detailed reasoning for your evaluation
2. Scores for each sub-criterion (0-10)

### 1. CONTENT (Weight: 30%)
{content_criteria}

### 2. VISUAL DESIGN (Weight: 30%)
{visual_design_criteria}

### 3. LAYOUT (Weight: 20%)
{layout_criteria}

### 4. COMPLEXITY (Weight: 20%)
{complexity_criteria}


## INSTRUCTIONS

1. Examine ALL slide images carefully in sequence.
2. Compare the presentation content against the source document.
3. Evaluate each sub-criterion objectively and assign an EXACT INTEGER score from 0-10 (NEVER use decimals).
4. Provide brief constructive feedback on overall strengths and areas for improvement.

---

## OUTPUT FORMAT (JSON)

**CRITICAL: All scores MUST be EXACT INTEGERS (whole numbers like 0, 1, 2, ..., 10). NEVER use decimals like 8.3, 7.5, etc. Round to the nearest whole number if needed.**

Return ONLY the JSON in the following format (no reasoning text):
{{
    "Content": {{
        "sub_scores": {{
            "Accuracy_and_Completeness": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Logical_Flow": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Cognitive_Rhythm": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }}
    }},
    "Visual_Design": {{
        "sub_scores": {{
            "Color_Scheme": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Typography": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Visual_Consistency": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Image_Quality": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Theme_Appropriateness": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }}
    }},
    "Layout": {{
        "sub_scores": {{
            "Spatial_Balance": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Element_Alignment": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "No_Overlapping": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }}
    }},
    "Complexity": {{
        "sub_scores": {{
            "Charts_and_Data": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Visual_Elements": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Advanced_Design": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Layout_Variety": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }}
    }},
    "Overall_Feedback": "<brief summary>",
    "Top_Strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "Areas_for_Improvement": ["<area 1>", "<area 2>", "<area 3>"]
}}
"""

def generate_criteria_description(criterion_name: str) -> str:
    """Generate detailed criteria description for prompts."""
    if criterion_name not in CRITERIA:
        return ""
    
    criterion = CRITERIA[criterion_name]
    lines = [f"**{criterion_name}**: {criterion['description']}\n"]
    
    if "sub_criteria" in criterion:
        lines.append("Sub-criteria:")
        for sub_name, sub_data in criterion["sub_criteria"].items():
            lines.append(f"  - **{sub_name}** (Weight: {sub_data['weight']*100:.0f}%): {sub_data['description']}")
            if "scoring_rubric" in sub_data:
                lines.append("    Scoring guide:")
                for score_range, description in sub_data["scoring_rubric"].items():
                    lines.append(f"      * {score_range}: {description}")
    
    return "\n".join(lines)

# =============================================================================
# VLM PROMPTS - ARENA COMPARISON
# =============================================================================

ARENA_COMPARISON_PROMPT_TEMPLATE = """
You are an expert presentation judge. Compare two presentations (PPT A and PPT B) on the same topic based on their slide images.

**Topic:** {topic}
**PPT A:** {num_slides_a} slides | **PPT B:** {num_slides_b} slides

## CRITERIA
1. **VISUAL DESIGN**: Color scheme, typography, consistency, image quality, theme
2. **LAYOUT**: Spatial balance, alignment, no overlapping, professional structure

## OUTPUT (JSON only)

{{
    "Visual_Design": {{"winner": "A"|"B"|"Tie", "score_difference": 1-5, "reason": "<brief reason>"}},
    "Layout": {{"winner": "A"|"B"|"Tie", "score_difference": 1-5, "reason": "<brief reason>"}},
    "Overall_Winner": "A"|"B"|"Tie",
    "Overall_Reason": "<brief overall comparison>",
    "Confidence": 1-5
}}
"""

# =============================================================================
# VLM PROMPTS - JUDGE AGENT
# =============================================================================

JUDGE_AGENT_PROMPT_TEMPLATE = """
You are a supreme judge for resolving inconsistencies in PPT evaluation.
We have a conflict between the Quantitative Evaluation scores and the Head-to-Head Arena result.

---

## CONTEXT

**Topic:** {topic}

**PPT A (Product: {product_a}):**
- Quantitative Total Score: {score_a}/40
- Content: {score_a_content}/10
- Visual Design: {score_a_visual}/10
- Layout: {score_a_layout}/10
- Complexity: {score_a_complexity}/10

**PPT B (Product: {product_b}):**
- Quantitative Total Score: {score_b}/40
- Content: {score_b_content}/10
- Visual Design: {score_b_visual}/10
- Layout: {score_b_layout}/10
- Complexity: {score_b_complexity}/10

**Arena Result:** {arena_winner} was judged better
**Arena Confidence:** {arena_confidence}/5
**Arena Reasoning:** {arena_reason}

---

## INCONSISTENCY

The Quantitative scores suggest **{score_winner}** is better (by {score_diff} points), 
but the Arena comparison chose **{arena_winner}**.

---

## YOUR TASK

1. Re-examine both presentations carefully.
2. Consider why the inconsistency might have occurred.
3. Make a final determination on which presentation is truly better.
4. Provide corrected scores if the quantitative evaluation was flawed.

---

## POTENTIAL REASONS FOR INCONSISTENCY
- Quantitative evaluation may have overlooked comparative advantages
- Arena may have focused too heavily on certain aspects
- One method may have been affected by bias or random factors
- The presentations may be genuinely very close in quality

---

## OUTPUT FORMAT (JSON)

{{
    "Final_Winner": "A" | "B" | "Tie",
    "Confidence": <1-5>,
    "Corrected_Scores": {{
        "A": {{
            "Content": <number>,
            "Visual_Design": <number>,
            "Layout": <number>,
            "Complexity": <number>,
            "Total": <number>
        }},
        "B": {{
            "Content": <number>,
            "Visual_Design": <number>,
            "Layout": <number>,
            "Complexity": <number>,
            "Total": <number>
        }}
    }},
    "Inconsistency_Analysis": "<why did the methods disagree>",
    "Reason": "<detailed justification for final decision>",
    "Recommendations": ["<recommendation 1>", "<recommendation 2>"]
}}
"""

SINGLE_SLIDE_EXTRACTION_PROMPT = """
Analyze the attached slide image for the specific purpose of a Content Quality & Accuracy Audit. I need to compare this slide against source documentation, so precision is paramount.

Please extract the content into the following structured concise Markdown file:

1. Textual Claims (Verbatim):
Headlines: Extract the exact Title and Subtitle.
Core Statements: List every distinct claim or bullet point found in the body text exactly as written. Do not summarize.
Callouts: Extract text from any bubbles, arrows, or highlight boxes.

2. Quantitative Data Extraction:
Chart/Table Data: For every chart or table, list the specific data points visible. (e.g., 'Q1 Revenue: $10m', 'Year-over-Year growth: 15%').
In-Text metrics: List any standalone numbers found in the text (e.g., '300+ employees', '50% reduction').

3. Visual Interpretation:

Describe important images or icons that contain information and explain if they convey a specific sentiment or data point (e.g., 'A green up-arrow indicating positive trend'). Ignore all decorative elements or irrelevant information.

Return the extracted content in a structured concise Markdown file.
"""

CONTENT_EVALUATION_PROMPT = """
You are an expert content evaluator. Evaluate the CONTENT QUALITY of a PowerPoint presentation by comparing the extracted slide contents against the source document.

**Source Document Summary/Content:**
{document_content}

**Presentation Topic:**
{topic}

**Number of Slides:** {num_slides}

**Extracted Slide Contents:**
{slide_contents}

---

## CONTENT EVALUATION CRITERIA

{content_criteria}

## INSTRUCTIONS

1. Review ALL extracted slide contents carefully.
2. Compare the presentation content against the source document for accuracy and completeness.
3. Evaluate each sub-criterion objectively and assign an EXACT INTEGER score from 0-10 (NEVER use decimals).
4. Provide detailed reasoning for your evaluation.

---

## OUTPUT FORMAT (JSON)

**CRITICAL: All scores MUST be EXACT INTEGERS (whole numbers like 0, 1, 2, ..., 10). NEVER use decimals.**

Return ONLY the JSON in the following format:
{{
    "Content": {{
        "sub_scores": {{
            "Accuracy_and_Completeness": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Logical_Flow": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Cognitive_Rhythm": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }},
        "reason": "<detailed reasoning for content evaluation>"
    }}
}}
"""

QUIZ_EVALUATION_PROMPT = """
You are an expert quiz evaluator. Answer the following multiple-choice questions based ONLY on the information presented in the extracted slide contents.

**Presentation Topic:**
{topic}

**Extracted Slide Contents:**
{slide_contents}

**Quiz Questions:**
{quiz_questions}

---

## INSTRUCTIONS

1. Read all extracted slide contents carefully.
2. For each question, select the best answer based ONLY on what's presented in the slides.
3. If the information is not covered in the slides, make your best inference or select "insufficient information".
4. Provide brief reasoning for each answer.

---

## OUTPUT FORMAT (JSON)

Return ONLY the JSON in the following format:
{{
    "answers": [
        {{
            "question_id": <number>,
            "selected_answer": "<A|B|C|D>",
            "reasoning": "<brief explanation>"
        }},
        ...
    ]
}}
"""

VISUAL_EVALUATION_PROMPT = """
You are an expert presentation designer evaluating the VISUAL DESIGN, LAYOUT, and COMPLEXITY of a PowerPoint presentation.
Your task is to evaluate the presentation based ONLY on the provided slide images.

**Presentation Topic:**
{topic}

**Number of Slides:** {num_slides}

---

## EVALUATION CRITERIA

Please evaluate the presentation on the following criteria. For each criterion, provide:
1. Detailed reasoning for your evaluation
2. Scores for each sub-criterion (0-10)

### 1. VISUAL DESIGN (Weight: 30%)
{visual_design_criteria}

### 2. LAYOUT (Weight: 20%)
{layout_criteria}

### 3. COMPLEXITY (Weight: 20%)
{complexity_criteria}

## INSTRUCTIONS

1. Examine ALL slide images carefully in sequence.
2. Evaluate each sub-criterion objectively and assign an EXACT INTEGER score from 0-10 (NEVER use decimals).
3. Provide brief constructive feedback on overall strengths and areas for improvement.

---

## OUTPUT FORMAT (JSON)

**CRITICAL: All scores MUST be EXACT INTEGERS (whole numbers like 0, 1, 2, ..., 10). NEVER use decimals like 8.3, 7.5, etc. Round to the nearest whole number if needed.**

Return ONLY the JSON in the following format (no reasoning text):
{{
    "Visual_Design": {{
        "sub_scores": {{
            "Color_Scheme": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Typography": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Visual_Consistency": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Image_Quality": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Theme_Appropriateness": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }},
        "reason": "<detailed reasoning>"
    }},
    "Layout": {{
        "sub_scores": {{
            "Spatial_Balance": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Element_Alignment": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "No_Overlapping": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }},
        "reason": "<detailed reasoning>"
    }},
    "Complexity": {{
        "sub_scores": {{
            "Charts_and_Data": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Visual_Elements": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Advanced_Design": <integer: 0|1|2|3|4|5|6|7|8|9|10>,
            "Layout_Variety": <integer: 0|1|2|3|4|5|6|7|8|9|10>
        }},
        "reason": "<detailed reasoning>"
    }},
    "Overall_Feedback": "<brief summary>",
    "Top_Strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "Areas_for_Improvement": ["<area 1>", "<area 2>", "<area 3>"]
}}
"""

# A lightweight variant that returns three-level grades instead of 0-10 scores
VISUAL_EVALUATION_GRADE_PROMPT = """
You are an expert presentation designer evaluating the VISUAL DESIGN, LAYOUT, and COMPLEXITY of a PowerPoint presentation.
Your task is to evaluate the presentation based ONLY on the provided slide images.

**Presentation Topic:**
{topic}

**Number of Slides:** {num_slides}

---

## EVALUATION CRITERIA

Please evaluate the presentation on the following criteria. For each criterion, provide:
1. Detailed reasoning for your evaluation
2. Grades for each sub-criterion using ONLY these labels: "bad", "moderate", "good"

### 1. VISUAL DESIGN
{visual_design_criteria}

### 2. LAYOUT
{layout_criteria}

### 3. COMPLEXITY
{complexity_criteria}

## GRADE DEFINITIONS
- "bad": serious issues or poor execution
- "moderate": acceptable but with noticeable issues
- "good": strong quality with minimal issues

## INSTRUCTIONS

1. Examine ALL slide images carefully in sequence.
2. Assign ONLY the allowed grades ("bad", "moderate", "good"). Do NOT use numbers.
3. Grade each criterion strictly; do not inflate scores to please someone.
4. Use the "moderate" rating only when the work meets expectations but is not outstanding, and justify it when possible.
5. Provide brief constructive feedback on overall strengths and areas for improvement.

---

## OUTPUT FORMAT (JSON)

Return ONLY the JSON in the following format (no reasoning text outside the JSON):
{{
    "Visual_Design": {{
        "sub_grades": {{
            "Color_Scheme": "<bad|moderate|good>",
            "Typography": "<bad|moderate|good>",
            "Visual_Consistency": "<bad|moderate|good>",
            "Image_Quality": "<bad|moderate|good>",
            "Theme_Appropriateness": "<bad|moderate|good>"
        }},
        "grade": "<bad|moderate|good>",
        "reason": "<detailed reasoning>"
    }},
    "Layout": {{
        "sub_grades": {{
            "Spatial_Balance": "<bad|moderate|good>",
            "Element_Alignment": "<bad|moderate|good>",
            "No_Overlapping": "<bad|moderate|good>"
        }},
        "grade": "<bad|moderate|good>",
        "reason": "<detailed reasoning>"
    }},
    "Complexity": {{
        "sub_grades": {{
            "Charts_and_Data": "<bad|moderate|good>",
            "Visual_Elements": "<bad|moderate|good>",
            "Advanced_Design": "<bad|moderate|good>",
            "Layout_Variety": "<bad|moderate|good>"
        }},
        "grade": "<bad|moderate|good>",
        "reason": "<detailed reasoning>"
    }},
    "Overall_Feedback": "<brief summary>",
    "Top_Strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "Areas_for_Improvement": ["<area 1>", "<area 2>", "<area 3>"]
}}
"""

# =============================================================================
# ADDITIONAL PROMPTS
# =============================================================================

SINGLE_SLIDE_ANALYSIS_PROMPT = """
Analyze this single slide and provide detailed feedback:

**Slide Number:** {slide_num}
**Presentation Topic:** {topic}

Evaluate:
1. Content clarity and relevance
2. Visual design quality
3. Layout effectiveness
4. Any issues or problems

Output JSON:
{{
    "slide_number": {slide_num},
    "content_score": <0-10>,
    "design_score": <0-10>,
    "layout_score": <0-10>,
    "issues": ["<issue 1>", "<issue 2>"],
    "strengths": ["<strength 1>", "<strength 2>"],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}
"""

COMPARATIVE_SLIDE_PROMPT = """
Compare these two slides from different presentations on the same topic:

**Topic:** {topic}
**Slide Position:** {slide_num}

**Slide A** is from presentation A.
**Slide B** is from presentation B.

Which slide is more effective and why?

Output JSON:
{{
    "winner": "A" | "B" | "Tie",
    "reason": "<detailed comparison>",
    "a_strengths": ["<strength>"],
    "b_strengths": ["<strength>"],
    "score_a": <0-10>,
    "score_b": <0-10>
}}
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_weighted_score(scores: Dict[str, float]) -> float:
    """Calculate weighted total score from individual criterion scores."""
    total = 0.0
    total_weight = 0.0
    for criterion_name, score in scores.items():
        if criterion_name in CRITERIA:
            weight = CRITERIA[criterion_name].get("weight", 0.2)
            total += score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    normalized_total = total / total_weight
    return round(normalized_total * 10, 2)  # Scale to 0-100

def get_criteria_names() -> List[str]:
    """Get list of all criterion names."""
    return list(CRITERIA.keys())

def get_sub_criteria_names(criterion: str) -> List[str]:
    """Get list of sub-criteria for a given criterion."""
    if criterion in CRITERIA and "sub_criteria" in CRITERIA[criterion]:
        return list(CRITERIA[criterion]["sub_criteria"].keys())
    return []

def validate_scores(result: dict) -> bool:
    """Validate that all scores are within valid ranges."""
    for criterion_name in CRITERIA:
        if criterion_name not in result:
            return False
        score = result[criterion_name].get("score", -1)
        if not (0 <= score <= 10):
            return False
    return True

