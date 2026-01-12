# PPT Evaluation System

This evaluation system provides comprehensive assessment of PowerPoint presentations using both Vision-Language Models (VLMs) and image-based aesthetics metrics.

## Overview

The system consists of two main components:

1. **Quantitative Evaluation** (`quantitative_eval.py`): Uses VLMs to evaluate content, visual design, layout, and complexity
2. **Aesthetics Metrics** (`aesthetics_metrics.py`): Computes image-based metrics like figure-ground contrast, color harmony, colorfulness, subband entropy, and visual HRV

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy opencv-python pillow scipy scikit-image tqdm python-dotenv
# For expensive metrics (feature congestion):
pip install pyrtools
```

## Environment Configuration

The system supports `.env` files for managing API keys and endpoints. This is the recommended way to configure your API credentials.

### Setup

1. **Copy the example configuration:**
   ```bash
   cd eval
   cp .env.example .env
   ```

2. **Edit `.env` with your API credentials:**
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=sk-your-openai-api-key
   # OPENAI_BASE_URL=https://api.openai.com/v1  # Optional custom endpoint
   
   # Anthropic Configuration
   ANTHROPIC_API_KEY=your-anthropic-api-key
   
   # Google Gemini Configuration
   GOOGLE_API_KEY=your-google-api-key
   GEMINI_API_KEY=your-gemini-api-key
   
   # Custom Google endpoint (for custom-google provider)
   CUSTOM_GOOGLE_API_KEY=your-custom-api-key
   GOOGLE_BASE_URL=https://gemini.visioncoder.cn
   
   # Default provider settings
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o
   # LLM_FALLBACK_MODELS=qwen-vl-max,claude-3-haiku-20240307
   ```

3. **The system will automatically load from `.env`:**
   - API keys and base URLs are read from `.env` first
   - Falls back to system environment variables if `.env` is not found
   - Command-line arguments override environment variables

### Supported Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_BASE_URL` | Custom OpenAI endpoint | `https://api.openai.com/v1` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GOOGLE_API_KEY` | Google Gemini API key | `AI...` |
| `CUSTOM_GOOGLE_API_KEY` | Custom Gemini endpoint key | `your-key` |
| `GOOGLE_BASE_URL` | Custom Gemini endpoint URL | `https://gemini.visioncoder.cn` |
| `LLM_PROVIDER` | Default provider | `openai`, `anthropic`, `google`, `custom-google` |
| `LLM_MODEL` | Default model name | `gemini-3-flash-preview`, `gpt4o` |
| `LLM_FALLBACK_MODELS` | Comma-separated fallback models | `qwen-vl-max,gemini-2.5-pro` |

## Quick Start

### 1. Aesthetics-Only Evaluation (Fast, No VLM)

Evaluate PPTs using only image-based aesthetics metrics:

```bash
# Basic aesthetics evaluation
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --source-mode selected \
    --products Gamma NotebookLM

# With custom aesthetics config
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --source-mode selected \
    --aesthetics-config aesthetics_config/aesthetics_config.json \
    --products Gamma

# Evaluate specific topics
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --topics "Machine Learning" "Climate Change" \
    --aesthetics-config aesthetics_config/aesthetics_config.json
```

### 2. Full VLM Evaluation

Complete evaluation using VLM for content + visual assessment:

```bash
# Full evaluation with OpenAI
python quantitative_eval.py \
    --provider openai \
    --model gpt-4o \
    --eval-mode full \
    --source-mode selected \
    --workers 4

# Full evaluation with custom aesthetics config
python quantitative_eval.py \
    --provider openai \
    --eval-mode full \
    --aesthetics-config aesthetics_config/aesthetics_config.json \
    --source-mode all
```

### 3. Content-Only or Visual-Only Evaluation

```bash
# Content evaluation only (uses LLM, no images)
python quantitative_eval.py \
    --provider openai \
    --eval-mode content_only \
    --source-mode selected

# Visual evaluation only (uses VLM with images)
python quantitative_eval.py \
    --provider openai \
    --eval-mode visual_only \
    --source-mode selected
```

### 4. Incremental Evaluation with Cache

Resume evaluation from a previous run:

```bash
# Use cache to skip already-evaluated PPTs
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --use-cache \
    --cache-file output/aesthetics_results_20260111.json

# Auto-detect latest cache file
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --use-cache
```

### 5. Evaluate Target Folder

Evaluate a specific folder containing PPT subdirectories:

```bash
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --target-folder /path/to/ppt_outputs \
    --aesthetics-config aesthetics_config/aesthetics_config.json
```

## Aesthetics Configuration

The aesthetics scoring can be customized via a JSON config file. See `aesthetics_config/aesthetics_config.json` for an example:

```json
{
  "parameters": {
    "weight_contrast": 7,
    "weight_colorfulness": 9,
    "weight_colorfulness_score": 0.65,
    "w_colorfulness_mean": 17,
    "w_colorfulness_std": 6,
    "weight_colorfulness_pacing": 0.6,
    "w_colorfulness_target": 9.5,
    "w_colorfulness_target_std": 4,
    "weight_harmony": 0.2,
    "weight_harmony_mean": 5,
    "weight_harmony_deck": 30,
    "harmony_sigma": 0.0005,
    "weight_subband_entropy": 0.1,
    "weight_subband_entropy_score": 50,
    "w_subband_entropy_mean": 4.9,
    "w_subband_entropy_std": 1.5,
    "weight_target_rmssd": 0.02,
    "weight_target_halfwidth": 0.1,
    "se_k": 1.7,
    "se_mu": 2.6
  }
}
```

### Parameter Descriptions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_contrast` | Weight for figure-ground contrast | 20.0 |
| `weight_colorfulness` | Weight for colorfulness metric | 1.0 |
| `w_colorfulness_mean` | Target mean colorfulness | 50.0 |
| `w_colorfulness_std` | Std dev for colorfulness Gaussian | 20.0 |
| `w_colorfulness_target` | Target colorfulness pacing (std) | 8.0 |
| `weight_harmony` | Weight for color harmony metric | 0.5 |
| `harmony_sigma` | Sigma for harmony score normalization | 0.05 |
| `weight_subband_entropy` | Weight for entropy metric | 1.0 |
| `w_subband_entropy_mean` | Target mean entropy | 3.5 |
| `weight_target_rmssd` | Target RMSSD for visual HRV | 0.1 |
| `se_k` | Sigmoid slope for entropy normalization | 1.5 |
| `se_mu` | Sigmoid midpoint for entropy | 5.5 |

## Using aesthetics_metrics.py Directly

You can also use the aesthetics metrics module directly:

```bash
# Evaluate a single image
python aesthetics_metrics.py /path/to/slide.png

# Evaluate a directory of images
python aesthetics_metrics.py /path/to/slide_images/ -o results.json

# Evaluate with specific metrics only
python aesthetics_metrics.py /path/to/slide_images/ \
    --metrics colorfulness,figure_ground_contrast,color_harmony

# Parallel processing
python aesthetics_metrics.py /path/to/slide_images/ -w 4

# With expensive metrics (feature congestion)
python aesthetics_metrics.py /path/to/slide_images/ -e
```

### Python API

```python
from aesthetics_metrics import (
    SlideAestheticsCalculator,
    ScoreParameters,
    load_parameters_from_config,
    calculate_total_aesthetics_score,
    FigureGroundContrastMetric,
    ColorfulnessMetric,
    ColorHarmonyMetric,
    SubbandEntropyMetric,
    VisualHRVMetric
)

# Load custom parameters
params = load_parameters_from_config("aesthetics_config/aesthetics_config.json")

# Create calculator
calculator = SlideAestheticsCalculator()

# Calculate metrics for a presentation
results = calculator.calculate_presentation_metrics(
    "/path/to/slide_images/",
    n_workers=4
)

# Get aggregated metrics
aggregated = results.get("aggregated", {})

# Calculate total aesthetics score with custom parameters
total_score, components = calculate_total_aesthetics_score(aggregated, params)
print(f"Total Score: {total_score:.2f}")
print(f"Components: {components}")

# Individual metric scoring
contrast_score, _ = FigureGroundContrastMetric.compute_score(0.5, params)
colorfulness_score, _ = ColorfulnessMetric.compute_score(17.0, 9.5, params)
harmony_score, _ = ColorHarmonyMetric.compute_score([10.0, 12.0, 8.0], params)
entropy_score, _ = SubbandEntropyMetric.compute_score(4.9, params)
rmssd_score, _ = VisualHRVMetric.compute_score([3.5, 4.0, 4.5], params)
```

## Command-Line Options Reference

### quantitative_eval.py

```
--provider          VLM provider (openai, anthropic, google, ollama, mock)
--model             Specific model name
--ppt-root          Root directory for generated PPTs
--output-dir        Directory to save results
--products          Products to evaluate
--difficulties      Difficulty levels to evaluate
--topics            Specific topics to evaluate
--limit             Maximum PPTs per product
--max-images        Maximum images per PPT
--workers           Parallel workers (default: 4)
--no-aesthetics     Disable aesthetics metrics
--aesthetics-metrics  Comma-separated metrics to compute
--aesthetics-config Path to aesthetics config JSON
--eval-mode         full|vlm_only|content_only|visual_only|aesthetics_only
--source-mode       selected|lite|scene|all
--use-cache         Use cached results
--cache-file        Path to cache file
--use-grids         Use slide grids for VLM
--visual-grade-mode Use grade-based visual evaluation
--target-folder     Process specific folder
--output-file       Output filename
--list-products     List available products
```

### aesthetics_metrics.py

```
input               Image file(s) or directory
-o, --output        Output JSON file path
-v, --verbose       Verbose output
-e, --expensive     Enable expensive metrics
--umsi-model        Path to UMSI model
-w, --workers       Number of parallel workers
--no-parallel       Disable parallel processing
--metrics           Comma-separated metrics list
```

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "evaluation_time": "2026-01-11T21:00:00",
    "vlm_model": "gpt-4o",
    "eval_mode": "full",
    "num_evaluations": 100
  },
  "results": [
    {
      "product": "Gamma",
      "difficulty": "high",
      "topic": "Machine Learning",
      "ppt_id": "ml_001",
      "num_slides": 10,
      "content_score": 8.5,
      "visual_design_score": 7.8,
      "layout_score": 8.2,
      "complexity_score": 7.5,
      "weighted_total": 8.0,
      "aesthetics_metrics": {
        "figure_ground_contrast": {"mean": 0.45},
        "colorfulness": {"mean": 25.3, "std": 8.2},
        "color_harmony": {"raw_distances": [...]},
        "subband_entropy": {"mean": 4.2},
        "visual_hrv": {"rmssd": 0.08}
      }
    }
  ],
  "summary": {
    "overall": {"mean": 7.8, "min": 5.2, "max": 9.5},
    "by_product": {"Gamma": 8.1, "NotebookLM": 7.5},
    "by_criterion": {...}
  }
}
```

## Tips

1. **For fast iteration**: Use `--eval-mode aesthetics_only` to skip VLM calls
2. **For large datasets**: Use `--use-cache` to resume interrupted runs
3. **For specific analysis**: Use `--aesthetics-metrics` to compute only needed metrics
4. **For custom scoring**: Create a JSON config file with your preferred weights
5. **For parallel speedup**: Increase `--workers` (recommended: 2-4 for API calls)

## Troubleshooting

### Missing pyrtools

```bash
# Install pyrtools for subband entropy and feature congestion
pip install pyrtools
```

### Rate Limiting

If you encounter API rate limits:
- Reduce `--workers` to 1-2
- Add delays between calls
- Use `--use-cache` to resume

### Memory Issues

For large presentations:
- Use `--max-images` to limit slides processed
- Disable expensive metrics with `--no-aesthetics` for VLM-only evaluation
