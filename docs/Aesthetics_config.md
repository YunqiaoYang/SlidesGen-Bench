# Aesthetics Evaluation Configuration Guide

This document provides detailed configuration options for the computational aesthetics evaluation in SlideGen-Bench.

---

## Overview

The aesthetics evaluation module computes four core metrics to assess visual quality:

| Metric | Description |
|--------|-------------|
| **Figure-Ground Contrast** | Measures contrast between foreground and background using WCAG standards |
| **Color Harmony** | Computes distance to harmonic color templates based on color theory |
| **Colorfulness** | Measures image colorfulness using Hasler & Süsstrunk's method |
| **Subband Entropy & Visual HRV** | Analyzes visual complexity and temporal consistency across slides |

---

## Quick Start

### Basic Aesthetics Evaluation

Evaluate presentations using image-based aesthetics metrics (no VLM required):

```bash
# Basic evaluation
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --products Gamma NotebookLM

# With custom configuration
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --aesthetics-config aesthetics_config/aesthetics_config.json \
    --products Gamma

# Evaluate specific topics
python quantitative_eval.py \
    --eval-mode aesthetics_only \
    --topics "Machine Learning" "Climate Change"
```

---

## Configuration File

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

### Parameter Reference

#### Figure-Ground Contrast

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_contrast` | Weight for figure-ground contrast in total score | 7.0 |

#### Colorfulness

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_colorfulness` | Weight for colorfulness metric | 9.0 |
| `weight_colorfulness_score` | Score multiplier | 0.65 |
| `w_colorfulness_mean` | Target mean colorfulness | 17.0 |
| `w_colorfulness_std` | Standard deviation for Gaussian scoring | 6.0 |
| `weight_colorfulness_pacing` | Pacing weight for slide-to-slide variation | 0.6 |
| `w_colorfulness_target` | Target colorfulness pacing (std) | 9.5 |
| `w_colorfulness_target_std` | Target pacing standard deviation | 4.0 |

#### Color Harmony

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_harmony` | Weight for color harmony metric | 0.2 |
| `weight_harmony_mean` | Mean harmony weight | 5.0 |
| `weight_harmony_deck` | Deck-level harmony weight | 30.0 |
| `harmony_sigma` | Sigma for harmony score normalization | 0.0005 |

#### Subband Entropy & Visual HRV

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_subband_entropy` | Weight for entropy metric | 0.1 |
| `weight_subband_entropy_score` | Entropy score multiplier | 50.0 |
| `w_subband_entropy_mean` | Target mean entropy | 4.9 |
| `w_subband_entropy_std` | Entropy standard deviation | 1.5 |
| `weight_target_rmssd` | Target RMSSD for visual HRV | 0.02 |
| `weight_target_halfwidth` | Half-width for RMSSD scoring | 0.1 |
| `se_k` | Sigmoid slope for entropy normalization | 1.7 |
| `se_mu` | Sigmoid midpoint for entropy | 2.6 |

---

## Standalone Usage

### Command Line

Use `aesthetics_metrics.py` directly for individual images or directories:

```bash
# Evaluate a single slide image
python aesthetics_metrics.py /path/to/slide.png

# Evaluate a directory of slide images
python aesthetics_metrics.py /path/to/slide_images/ -o results.json

# Select specific metrics
python aesthetics_metrics.py /path/to/slide_images/ \
    --metrics figure_ground_contrast,color_harmony,colorfulness,subband_entropy

# Parallel processing with 4 workers
python aesthetics_metrics.py /path/to/slide_images/ -w 4
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

# Calculate total aesthetics score
total_score, components = calculate_total_aesthetics_score(aggregated, params)
print(f"Total Score: {total_score:.2f}")
print(f"Components: {components}")
```

#### Individual Metric Computation

```python
# Compute individual metric scores
contrast_score, _ = FigureGroundContrastMetric.compute_score(0.5, params)
colorfulness_score, _ = ColorfulnessMetric.compute_score(17.0, 9.5, params)
harmony_score, _ = ColorHarmonyMetric.compute_score([10.0, 12.0, 8.0], params)
entropy_score, _ = SubbandEntropyMetric.compute_score(4.9, params)
rmssd_score, _ = VisualHRVMetric.compute_score([3.5, 4.0, 4.5], params)
```

---

## Command-Line Options

### quantitative_eval.py

| Option | Description |
|--------|-------------|
| `--eval-mode` | Evaluation mode: `aesthetics_only` for metrics only |
| `--products` | Products to evaluate |
| `--topics` | Specific topics to evaluate |
| `--aesthetics-config` | Path to aesthetics config JSON |
| `--aesthetics-metrics` | Comma-separated metrics to compute |
| `--workers` | Number of parallel workers (default: 4) |
| `--output-dir` | Directory to save results |
| `--output-file` | Output filename |

### aesthetics_metrics.py

| Option | Description |
|--------|-------------|
| `input` | Image file(s) or directory |
| `-o, --output` | Output JSON file path |
| `-v, --verbose` | Verbose output |
| `-w, --workers` | Number of parallel workers |
| `--no-parallel` | Disable parallel processing |
| `--metrics` | Comma-separated metrics list |
| `-c, --config` | Path to JSON config file |
| `--compute-score` | Compute total aesthetics score |

---

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "metadata": {
    "evaluation_time": "2026-01-11T21:00:00",
    "eval_mode": "aesthetics_only",
    "num_evaluations": 100
  },
  "results": [
    {
      "product": "Gamma",
      "topic": "Machine Learning",
      "ppt_id": "ml_001",
      "num_slides": 10,
      "aesthetics_metrics": {
        "figure_ground_contrast": {"mean": 0.45},
        "colorfulness": {"mean": 25.3, "std": 8.2},
        "color_harmony": {"raw_distances": [...]},
        "subband_entropy": {"mean": 4.2},
        "visual_hrv": {"rmssd": 0.08}
      },
      "aesthetics_score": 72.5
    }
  ],
  "summary": {
    "by_product": {"Gamma": 8.1, "NotebookLM": 7.5}
  }
}
```

---

## Troubleshooting

### Missing pyrtools

The subband entropy metric requires `pyrtools`:

```bash
pip install pyrtools
```

### Memory Issues

For large presentations, reduce the number of parallel workers:

```bash
python aesthetics_metrics.py /path/to/slides/ -w 2
```

