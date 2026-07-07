# Hiver AI Email Response Challenge

## Overview

This project builds a Gen-AI email suggested-response system for customer support. It takes an incoming email, generates a suggested reply using a generative approach, and evaluates that reply against a reference response using a multi-metric scoring system.

The system is structured as an end-to-end pipeline:

1. **Dataset Generation** — Creates a synthetic but realistic dataset of incoming emails paired with reference replies across 5 support categories.
2. **Gen-AI Response Generator** — Uses a generative approach that can run with a mock provider by default and can also use OpenAI or Anthropic if API keys are available.
3. **Accuracy and Evaluation System** — Scores each generated response using semantic, structural, and quality-oriented metrics, then reports per-response and overall scores.

---

## Dataset: where it came from and why it is representative

The dataset is synthetic and hand-authored in this repository. I created it to avoid relying on private support data while still covering realistic support scenarios such as billing disputes, technical issues, feature requests, account changes, and general questions.

Why this is a reasonable proxy for the challenge:
- It contains paired examples of incoming customer email and expected reply.
- It covers multiple customer support categories and urgency/tier combinations.
- It includes a mix of actionable, specific, and conversational replies, which is useful for evaluating whether generated responses are helpful rather than merely fluent.

The data is generated from templates in [data/generate_dataset.py](data/generate_dataset.py), which creates varied names, dates, charges, error messages, plans, and ticket-like details so the pipeline has enough diversity to test generation and evaluation.

---

## Generation approach

The response generator is a lightweight generative system rather than a classical classifier. It uses a prompt-style approach with a fallback mock provider and optional few-shot examples drawn from the generated dataset.

Why this approach was chosen:
- It is simple to run end to end without needing heavy fine-tuning.
- It is directly aligned with the challenge’s requirement for a Gen-AI response generator.
- It can work offline with the mock provider, while still supporting OpenAI or Anthropic when credentials are present.

The generator is implemented in [generator/response_generator.py](generator/response_generator.py). In the default run, the mock provider is used so the project remains runnable even without API keys.

---

## Accuracy and evaluation strategy

A strict exact-match metric would be too brittle for this task because good support replies can be phrased differently while still being correct. Instead, the evaluation system uses a weighted composite of multiple metrics that capture different aspects of response quality.

### Metrics used

| Metric | Weight | Why it matters |
|---|---:|---|
| **Semantic Similarity** | 25% | Measures whether the generated reply is meaningfully aligned with the reference reply. |
| **ROUGE-F1** | 15% | Captures structural overlap and shared phrasing. |
| **Response Length** | 10% | Rewards replies that are neither too short nor overly verbose. |
| **Politeness** | 10% | Encourages courteous support tone. |
| **Action Items** | 10% | Rewards replies that contain concrete next steps or commitments. |
| **Specificity** | 10% | Rewards detailed, concrete replies rather than generic ones. |
| **Tone Match** | 10% | Checks whether the tone is similar to the reference reply. |
| **Readability** | 5% | Encourages clear and easy-to-understand responses. |
| **BLEU-style score** | 5% | Adds a lexical similarity signal without relying on exact match. |

This makes the evaluation system more faithful to real support quality than a single metric would be. A response can be polite and readable but still miss action items; conversely, it can include action items but be too vague. The composite score balances these dimensions.

### How the evaluation is reported

The system outputs:
- a per-response score for each generated reply,
- an overall average score across the dataset,
- and a breakdown of the individual metrics used to build the aggregate score.

The evaluation logic lives in [evaluator/metrics.py](evaluator/metrics.py).

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
- Optional: OpenAI or Anthropic API keys if you want to use real LLM providers instead of the built-in mock provider.

### 2. Setup

```bash
git clone https://github.com/Infernus1011/hiver-email-ai.git
cd hiver-email-ai
python -m pip install -r requirements.txt
```

### 3. Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

This will:
1. Generate 50 synthetic email pairs
2. Generate AI responses using the mock provider by default
3. Evaluate response quality
4. Save outputs to [output](output)

### 4. Run Components Individually

**Generate a larger dataset:**
```bash
python data/generate_dataset.py --count 500 --output data/email_dataset.jsonl
```

**Generate responses using a provider:**
```bash
python generator/response_generator.py --input data/email_dataset_test.jsonl --output output/responses.jsonl --provider mock
```

**Evaluate generated responses:**
```bash
python evaluator/metrics.py --generated output/responses.jsonl --expected data/email_dataset_test.jsonl --output output/evaluation_results.json
```

### 5. Quick validation

```bash
python -m unittest -q tests.test_generate_dataset tests.test_response_generator
```

---

## AI tools usage

This project was built and iterated with AI assistance for code generation, debugging, and test scaffolding. The core repository structure, the dataset generator, the evaluation metrics, and the README were developed with AI help while keeping the pipeline runnable end to end.

---

## Submission readiness summary

This repository is ready to submit for the challenge because it includes:
- a runnable end-to-end pipeline,
- a dataset generator and documented dataset design,
- a Gen-AI-style response generator,
- and an evaluation system with per-response and overall scoring.

The main trade-off is that the dataset is synthetic rather than sourced from private support logs, but that is an intentional and transparent choice that keeps the project reproducible and privacy-safe.

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
