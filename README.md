# SlideGen-Bench

A comprehensive benchmark for evaluating AI-generated PowerPoint presentations.

## Overview

SlideGen-Bench provides quantitative evaluation tools for assessing PowerPoint presentations using Vision-Language Models (VLMs) and image-based aesthetics metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Evaluation Module

### Quantitative Evaluation

The `quantitative_eval.py` script performs detailed evaluation of PowerPoint presentations.

#### Usage

```bash
python eval/quantitative_eval.py [OPTIONS]
```

#### Key Options

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

#### Source Mode Options

- `topic`: Evaluates topic_introduction difficulty only
- `scene`: Evaluates scene-based difficulties (work_report, business_plan, brand_promote, personal_statement, product_launch, course_preparation)
- `all`: Evaluates all difficulties

#### Evaluation Mode Options

| Mode | Description |
|------|-------------|
| `full` | VLM evaluation + aesthetics metrics |
| `vlm_only` | Legacy content + visual evaluation |
| `content_only` | Text-based VLM evaluation only |
| `visual_only` | Image-only evaluation |
| `aesthetics_only` | Image-based metrics only, no VLM calls |

#### Aesthetics Options

| Option | Description |
|--------|-------------|
| `--no-aesthetics` | Disable aesthetics metrics computation |
| `--aesthetics-metrics` | Comma-separated list of metrics to compute |
| `--aesthetics-config` | Path to JSON config for aesthetics score parameters |

Available aesthetics metrics:
- `figure_ground_contrast`
- `lab`
- `hsv`
- `color_harmony`
- `visual_complexity`
- `colorfulness`
- `subband_entropy`

#### Visual Evaluation Options

| Option | Description |
|--------|-------------|
| `--use-grids` | Use slide grids to reduce VLM token consumption |
| `--visual-grade-mode` | Request bad/moderate/good grades for visual evaluation |

#### Target Folder Mode

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

**Aesthetics-only evaluation:**
```bash
python eval/quantitative_eval.py --eval-mode aesthetics_only --aesthetics-metrics lab,hsv,colorfulness
```

**Evaluate specific topics:**
```bash
python eval/quantitative_eval.py --source-mode topic --topics "Machine Learning" "Data Science"
```

**Process a target folder:**
```bash
python eval/quantitative_eval.py --target-folder /path/to/ppts --eval-mode aesthetics_only
```

### Aesthetics Metrics Module

The `aesthetics_metrics.py` module can also be run standalone:

```bash
python eval/aesthetics_metrics.py IMAGE_PATH [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Output JSON file path |
| `--verbose, -v` | Verbose output |
| `--workers, -w` | Number of parallel workers (default: 2) |
| `--no-parallel` | Disable parallel processing |
| `--metrics` | Comma-separated list of metrics to compute |
| `--config, -c` | Path to JSON config file for score parameters |
| `--compute-score` | Compute total aesthetics score from aggregated metrics |

## Output Format

Results are saved as JSON files containing:
- Individual PPT evaluation scores
- Aggregated metrics per product/difficulty
- Summary statistics

## Configuration

Configuration options can be set in `eval/eval_config.py` or via command-line arguments.

## License

See LICENSE file for details.
