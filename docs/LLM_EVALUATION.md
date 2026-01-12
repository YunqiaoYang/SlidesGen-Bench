# LLM-as-Judge Evaluation Guide

This document provides detailed instructions for using LLM-based evaluation methods in SlideGen-Bench.

## Overview

In addition to computational aesthetics metrics, we provide two LLM-based evaluation approaches:

1. **LLM Rating**: Direct scoring using Vision-Language Models
2. **LLM Arena**: Pairwise comparison with ELO ranking system

---

## LLM Rating Evaluation

### Description

LLM Rating uses Vision-Language Models (VLMs) to directly score slide presentations based on visual quality criteria.

### Usage

```bash
python eval/quantitative_eval.py [OPTIONS]
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | VLM provider (`openai`, `anthropic`, `google`, `ollama`, `mock`, `custom-google`) | `openai` |
| `--model` | Specific model name | Provider default |
| `--ppt-root` | Root directory for generated PPTs | Config default |
| `--output-dir` | Directory to save results | Config default |
| `--products` | Products to evaluate (`Gamma`, `NotebookLM`, `Kimi-Standard`, etc.) | All |
| `--difficulties` | Difficulties to evaluate | All |
| `--topics` | Specific topics to evaluate | All |
| `--limit` | Maximum number of PPTs to evaluate | None |
| `--max-images` | Maximum images per PPT | Config default |
| `--max-images-per-call` | Maximum images per VLM call (for batching) | Config default |
| `--source-mode` | Source content mode: `topic`, `scene`, or `all` | `all` |
| `--workers` | Number of parallel workers | 4 |
| `--use-cache` | Use cached results if available | False |
| `--cache-file` | Path to cache file for incremental evaluation | Auto-detected |
| `--output-file` | Output filename | Auto-generated |

### Source Mode Options

- `topic`: Evaluates topic_introduction difficulty only
- `scene`: Evaluates scene-based difficulties (work_report, business_plan, brand_promote, personal_statement, product_launch, course_preparation)
- `all`: Evaluates all difficulties

### Evaluation Mode Options

| Mode | Description |
|------|-------------|
| `full` | VLM evaluation + aesthetics metrics |
| `vlm_only` | Legacy content + visual evaluation |
| `content_only` | Text-based VLM evaluation only |
| `visual_only` | Image-only evaluation |
| `aesthetics_only` | Image-based metrics only, no VLM calls |

### Visual Evaluation Options

| Option | Description |
|--------|-------------|
| `--use-grids` | Use slide grids to reduce VLM token consumption |
| `--visual-grade-mode` | Request bad/moderate/good grades for visual evaluation |

### Target Folder Mode

| Option | Description |
|--------|-------------|
| `--target-folder` | Process subfolders of specified path as individual PPTs |

### Examples

**Basic evaluation:**
```bash
python eval/quantitative_eval.py --provider openai --products Gamma NotebookLM
```

**Evaluate with caching:**
```bash
python eval/quantitative_eval.py --use-cache --cache-file results/cached_results.json
```

**Visual-only evaluation:**
```bash
python eval/quantitative_eval.py --eval-mode visual_only --provider anthropic --model claude-3-opus-20240229
```

**Evaluate specific topics:**
```bash
python eval/quantitative_eval.py --source-mode topic --topics "Machine Learning" "Data Science"
```

**Process a target folder:**
```bash
python eval/quantitative_eval.py --target-folder /path/to/ppts --eval-mode visual_only
```

---

## LLM Arena Evaluation

### Description

LLM Arena performs pairwise comparisons between presentations from different products/tools using an ELO-based ranking system.

### Features

- Head-to-head VLM-based comparisons
- ELO rating system for ranking
- Supports multiple comparison strategies
- Detailed per-criterion comparison results

### Usage

```bash
python eval/arena_eval.py [OPTIONS]
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | VLM provider | `openai` |
| `--model` | Specific model name | Provider default |
| `--products` | Products to compare | All available |
| `--comparisons` | Number of comparisons to perform | Auto |
| `--output-dir` | Directory to save results | Config default |

### Example

```bash
python eval/arena_eval.py --provider openai --products Gamma NotebookLM Kimi-Standard
```

---

## VLM Interface Configuration

### Supported Providers

| Provider | Models | Notes |
|----------|--------|-------|
| `openai` | GPT-4V, GPT-4o | Requires `OPENAI_API_KEY` |
| `anthropic` | Claude 3 family | Requires `ANTHROPIC_API_KEY` |
| `google` | Gemini Pro Vision | Requires `GOOGLE_API_KEY` |
| `ollama` | LLaVA, etc. | Local deployment |
| `custom-google` | Custom endpoint | For custom deployments |

### Environment Setup

Create a `.env` file in the `eval/` directory (see `.env.example`):

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

---

## Output Format

### Rating Results

Results are saved as JSON files containing:
- Individual PPT evaluation scores
- Per-criterion scores (content, layout, design, etc.)
- Aggregated metrics per product/difficulty
- Summary statistics

### Arena Results

Arena results include:
- Pairwise comparison outcomes
- ELO ratings for each product
- Win/loss/tie statistics
- Confidence intervals

---

## Best Practices

1. **Consistency**: Use the same VLM provider and model across all evaluations for fair comparison
2. **Caching**: Enable caching (`--use-cache`) for large-scale evaluations to save API costs
3. **Batching**: Use `--max-images-per-call` to optimize token usage
4. **Grids**: Enable `--use-grids` to reduce token consumption while maintaining quality

---

## Configuration

Additional configuration options can be set in `eval/eval_config.py`.
