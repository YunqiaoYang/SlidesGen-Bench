<p align="center">
  <img src="images/logo.png" alt="SlideGen-Bench Logo" width="400">
</p>

<p align="center">
  <b>🎯 A Comprehensive Benchmark for Evaluating AI-Generated Presentations</b>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/Yqy6/SlideGen-Align">
    <img src="https://img.shields.io/badge/🤗%20Dataset-SlideGen--Align-yellow" alt="Dataset">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  </a>
  <a href="#installation">
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome">
  </a>
</p>

<p align="center">
  <a href="#-abstract">Abstract</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-evaluation-pipeline">Evaluation</a> •
  <a href="#-slidegen-align-dataset">Dataset</a> •
  <a href="#-license">License</a>
</p>

---

## 📖 Abstract

The rapid evolution of Large Language Models (LLMs) has fostered diverse paradigms for automated slide generation, ranging from code-driven layouts to image-centric synthesis. However, evaluating these heterogeneous systems remains challenging, as existing protocols often struggle to provide comparable scores across architectures or rely on uncalibrated judgments.

In this paper, we introduce **SlideGen-Bench**, a benchmark designed to evaluate slide generation through a lens of three core principles:

| Principle | Description |
|:---------:|:------------|
| 🌐 **Universality** | Unified visual-domain evaluation framework agnostic to generation methods |
| 📊 **Quantification** | Reproducible metrics across *Content*, *Aesthetics*, and *Editability* |
| ✅ **Reliability** | High correlation with human preference via the SlideGen-Align dataset |

<p align="center">
  <img src="images/main-pipeline.pdf" alt="Main Pipeline" width="800">
</p>

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yunqiaoyang/SlideGen-Bench.git
cd SlideGen-Bench

# Install dependencies
pip install -r requirements.txt
```

### 📦 Additional Setup

Configure **PaddleOCR DocLayout Detection** for layout analysis:
- 📖 [PaddleOCR Documentation](https://www.paddleocr.ai/latest/version3.x/module_usage/layout_detection.html#_4)
- We use the `PP-DocLayout_plus-L` model

---

## 🔬 Evaluation Pipeline

### 📋 Step 1: Slide Generation & Preprocessing

Convert all slide formats into images to ensure a unified evaluation framework.

<details>
<summary><b>🖼️ Image Conversion</b></summary>

We provide a converting script for preprocessing:

```bash
python eval/pre_process.py --input /path/to/slides --output /path/to/images
```

For pipelines that do not directly output images:

```bash
python eval/process_zhipu.py  # Example script - adapt to your pipeline
```

</details>

<details>
<summary><b>📑 PPTX to Image Conversion</b></summary>

For PPTX files, we use **LibreOffice** for conversion:
- 📖 [Official LibreOffice Documentation](https://www.libreoffice.org/get-help/documentation/)

</details>

---

### 📝 Step 2: Content Evaluation (QuizBank)

Evaluate content quality using the **QuizBank** methodology:

```bash
# Run content evaluation
python eval/quantitative_eval.py --eval-mode content_only --provider openai

# Calculate quiz accuracy and generate results
python eval/calculate_quiz_accuracy.py --input results/content_eval.json --output results/accuracy_table.csv
```

---

### 🎨 Step 3: Aesthetics Evaluation

#### 📐 Core Aesthetics Metrics

Computational aesthetics metrics for objective evaluation:

```bash
python eval/aesthetics_metrics.py IMAGE_PATH [OPTIONS]
```

| Metric | Description |
|:-------|:------------|
| `figure_ground_contrast` | 🔲 Measures foreground/background contrast using WCAG standards |
| `color_harmony` | 🎨 Computes distance to harmonic color templates |
| `colorfulness` | 🌈 Measures colorfulness using Hasler & Süsstrunk's method |
| `subband_entropy` | 📊 Analyzes visual complexity via subband decomposition |
| `visual_hrv` | 💓 Visual Heart Rate Variability for temporal consistency from subband entropy |

**Example Usage:**

```bash
python eval/quantitative_eval.py --eval-mode aesthetics_only \
    --aesthetics-metrics figure_ground_contrast,color_harmony,colorfulness,subband_entropy
```

> 📖 For detailed configuration, see [Aesthetics Configuration Guide](docs/Aesthetics_config.md)

#### 🤖 LLM-as-Judge Methods

We also provide LLM-based evaluation methods:

| Method | Description |
|:-------|:------------|
| **LLM Rating** | Direct scoring by language models |
| **LLM Arena** | Pairwise comparison with ELO ranking |

> 📖 See [LLM Evaluation Guide](docs/LLM_EVALUATION.md) for detailed documentation

---

### ✏️ Step 4: Presentation Editability Intelligence (PEI)

Evaluate presentation editability using a **knock-out evaluation strategy** — assessing how well generated presentations can be edited and modified after creation.

> 📄 **Reference:** [PEI Evaluation Protocol](eval/pei.md)

---

## 📋 Quick Reference

| Dimension | Method | Script | Description |
|:----------|:-------|:-------|:------------|
| 📝 **Content** | QuizBank | `quantitative_eval.py --eval-mode content_only` | Quiz-based content accuracy |
| 🎨 **Aesthetics** | Computational | `aesthetics_metrics.py` | Objective visual metrics |
| 🎨 **Aesthetics** | LLM Rating | `quantitative_eval.py --eval-mode visual_only` | LLM-based scoring |
| 🎨 **Aesthetics** | LLM Arena | `arena_eval.py` | Pairwise ELO ranking |
| ✏️ **Editability** | PEI Knock-out | [PEI Protocol](docs/PEI(2).pdf) | Edit capability assessment |

---

## ⚙️ Configuration

Configuration options can be set via:
- 📝 Config file: `eval/eval_config.py`
- 💻 Command-line arguments

---

## 📊 SlideGen-Align Dataset

<p align="center">
  <a href="https://huggingface.co/datasets/Yqy6/SlideGen-Align">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg" alt="Dataset on HF">
  </a>
</p>

We release **SlideGen-Align**, a human preference dataset for evaluating AI-generated slide presentations.

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/Yqy6/SlideGen-Align"><b>huggingface.co/datasets/Yqy6/SlideGen-Align</b></a>
</p>

### 📈 Dataset Statistics

<table align="center">
  <tr>
    <td align="center"><b>📊 Total Rankings</b><br>1,326</td>
    <td align="center"><b>🏢 Products</b><br>9</td>
    <td align="center"><b>📂 Categories</b><br>7</td>
    <td align="center"><b>💡 Topics</b><br>187</td>
  </tr>
</table>

### 🏢 Products Evaluated

| Product | Provider | Description |
|:--------|:---------|:------------|
| **Gamma** | Gamma.app | 🎨 AI presentation maker |
| **NotebookLM** | Google | 📓 AI notebook with presentation generation |
| **Kimi-Standard** | Moonshot AI | 🌙 Kimi (standard mode) |
| **Kimi-Smart** | Moonshot AI | 🧠 Kimi (smart mode) |
| **Kimi-Banana** | Moonshot AI | 🍌 Kimi (Banana template) |
| **Skywork** | Kunlun Tech | 🌤️ Skywork AI |
| **Skywork-Banana** | Kunlun Tech | 🍌 Skywork (Banana template) |
| **Zhipu** | Zhipu AI | 🤖 Presentation generator |
| **Quake** | ByteDance | ⚡ Quake presentation tool |

### 📂 Scenario Categories

| Category | Topics | Description |
|:---------|:------:|:------------|
| `topic_introduction` | 93 | 📚 General topic introductions (AI, Climate Change, 5G, etc.) |
| `product_launch` | 23 | 🚀 Product launch announcements |
| `personal_statement` | 20 | 👤 Personal statements and self-introductions |
| `brand_promote` | 15 | 📢 Brand promotion and marketing |
| `course_preparation` | 15 | 🎓 Educational course materials |
| `work_report` | 13 | 📊 Work progress reports |
| `business_plan` | 8 | 💼 Business plan presentations |

### 📝 Annotation Format

<details>
<summary>Click to expand annotation example</summary>

```json
{
    "results": [
        {
            "product": "NotebookLM",
            "difficulty": "topic_introduction",
            "topic": "FinTech",
            "rank": 1
        },
        ...
    ]
}
```

</details>

### 💻 Usage

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("Yqy6/SlideGen-Align")

# Access the data
for item in dataset['train']:
    print(f"{item['product']} - {item['topic']}: Rank {item['rank']}")
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>If you find SlideGen-Bench useful, please consider giving us a ⭐!</i>
</p>
