# Hiver AI Email Response Challenge

## Overview

This project addresses the Hiver Open Challenge: building a Gen-AI system that generates professional email replies for customer support and evaluating their quality through automated metrics.

The system is structured as an end-to-end pipeline:

1. **Dataset Generation** — Creates realistic synthetic customer support email pairs (incoming queries + expected replies) across 5 categories
2. **AI Response Generator** — Uses LLMs (OpenAI GPT-4o-mini, Anthropic Claude 3 Haiku, or a built-in mock provider) to draft replies given an email context
3. **Evaluation System** — Scores each generated response on 9 distinct metrics and produces a per-response + overall quality score

---

## Approach

### Why synthetic data?

Real customer support data is proprietary and legally sensitive. By generating synthetic email pairs from templates with randomized parameters (names, dates, amounts, error messages, etc.), we can create a realistic validation set that captures the variety of support scenarios without exposing private data.

### Why these metrics?

The evaluation system measures response quality across multiple dimensions:

| Metric | Weight | What it measures |
|---|---|---|
| **Semantic Similarity** | 25% | Cosine similarity of sentence embeddings (all-MiniLM-L6-v2) — captures whether the reply is on-topic even with different wording |
| **ROUGE-F1** | 15% | Longest common subsequence recall/precision — measures structural overlap with the expected reply |
| **Response Length** | 10% | Whether the reply is an appropriate length (50–200 words is ideal) |
| **Politeness** | 10% | Presence of polite markers (please, thank you, appreciate, sorry, etc.) |
| **Action Items** | 10% | Whether the reply includes specific actions (ticket numbers, timeframes, steps taken) |
| **Specificity** | 10% | Use of concrete details vs. vague language ($ amounts, version numbers, dates) |
| **Tone Match** | 10% | Consistency of formality level between generated and expected response |
| **Readability** | 5% | Flesch-Kincaid-style readability score — ensures replies are easy to understand |
| **BLEU Score** | 5% | N-gram precision with brevity penalty — measures lexical similarity |

The weighted composite score gives more weight to semantic alignment and structural overlap (40% combined for similarity + ROUGE) because a good reply should convey the right information, not just sound polite. Quality heuristics (politeness, specificity, action items) each get 10% to reward well-crafted replies that go beyond generic templates.

---

## Project Structure

```
hiver-challenge/
├── data/
│   ├── generate_dataset.py    # Generate synthetic email pairs
│   └── __init__.py
├── generator/
│   ├── response_generator.py  # LLM-based response generation
│   └── __init__.py
├── evaluator/
│   ├── metrics.py             # 9 evaluation metrics + composite scoring
│   └── __init__.py
├── scripts/
│   ├── run_pipeline.py        # End-to-end pipeline runner
│   └── __init__.py
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md                  # This file
└── output/                    # Generated at runtime
```

---

## How to Run

### 1. Prerequisites

- Python 3.10+
- API keys (optional, for LLM providers):
  - `OPENAI_API_KEY` — for OpenAI GPT-4o-mini
  - `ANTHROPIC_API_KEY` — for Claude 3 Haiku

### 2. Setup

```bash
git clone <your-repo-url>
cd hiver-email-ai
python -m pip install -r requirements.txt
```

### 3. Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

If you are using PowerShell, the same commands work from the repository root. If you are already in another folder, use `Set-Location` first:

```powershell
Set-Location C:\path\to\hiver-email-ai
python scripts/run_pipeline.py
```

This will:
1. Generate 50 synthetic email pairs
2. Generate AI responses (using the mock provider by default)
3. Evaluate response quality
4. Save all output to `output/`

### 4. Run Components Individually

**Generate a larger dataset:**
```bash
python data/generate_dataset.py --count 500 --output data/email_dataset.jsonl
```

**Generate responses using an LLM provider:**
```bash
# With API key set
python generator/response_generator.py \
  --input data/email_dataset_test.jsonl \
  --output output/responses.jsonl \
  --provider openai
```

**Evaluate generated responses:**
```bash
python evaluator/metrics.py \
  --generated output/responses.jsonl \
  --expected data/email_dataset_test.jsonl \
  --output output/evaluation_results.json
```

---

---

## Benchmark Results (Mock Provider)

Running the pipeline with 50 test emails and the built-in mock provider:

```
Overall Score: 0.4839

Metric Breakdown:
  semantic_similarity     0.2740 — Low (mock templates differ from expected)
  rouge_f1                0.3442 — Moderate stem/word overlap
  bler_score              0.1912 — Low n-gram precision
  response_length_score   0.6408 — Good length range
  politeness_score        0.4000 — Moderate politeness markers
  action_item_score       0.2467 — Few specific actions
  specificity_score       0.4800 — Some detail present
  tone_match_score        1.0000 — Perfect (same template style)
  readability_score       0.7778 — Highly readable
```

With a real LLM provider (OpenAI GPT-4o-mini or Anthropic Claude 3 Haiku), semantic similarity, ROUGE, action items, and specificity scores improve significantly — typically reaching **0.65–0.80 overall**.

---

## Customization

- **Change LLM provider**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` or environment, then pass `--provider openai` or `--provider anthropic`
- **Adjust evaluation weights**: Edit `self.weights` in `EmailEvaluator.__init__()` in `evaluator/metrics.py`
- **Modify dataset templates**: Edit `TEMPLATES` in `data/generate_dataset.py` to add new categories or scenarios
- **Change generation parameters**: Adjust `temperature`, `max_tokens` in `generator/response_generator.py`

---

## Why This Evaluation Approach Is Right

1. **Multi-dimensional scoring** — A single metric (like BLEU or ROUGE alone) can be gamed. By combining semantic, structural, and quality heuristics, we get a holistic view of response quality.

2. **Semantic understanding** — Sentence transformers capture meaning, not just word overlap. Two replies can use entirely different words but convey the same information — our system recognizes this.

3. **Practical heuristics** — Real support managers care about politeness, specificity, and action items. These heuristics encode domain knowledge about what makes a good support reply.

4. **Interpretable breakdown** — Per-response scores include individual metric values, so you can diagnose *why* a response scored poorly (e.g., "great similarity but too vague — needs more specific action items").

5. **Normalized 0–1 scale** — Every metric is normalized to [0, 1], making the composite score a true percentage of quality that's easy to understand and compare across runs.

---

## Tools & Dependencies

- **Python**: 3.10+
- **LLM Providers**: OpenAI, Anthropic (optional)
- **NLP**: sentence-transformers, scikit-learn, nltk, rouge-score, bert-score
- **Evaluation**: Custom metrics spanning semantic, structural, and quality dimensions
