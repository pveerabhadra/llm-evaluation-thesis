# LLM Evaluation for Scientific QA and MMLU

Master's thesis project evaluating three large language models across two benchmarks under different retrieval conditions.

## Models

| Model | Provider | Parameters |
|-------|----------|-----------|
| Gemma-4 31B FP8 | RedHatAI | 31B |
| GPT-OSS 120B | OpenAI | 120B |
| Qwen 3.5 122B FP8 | Qwen | 122B |

---

## Benchmarks

### 1. SciDQA — Scientific Document Question Answering

Evaluates 2,937 questions from the [SciDQA dataset](https://github.com/yale-nlp/SciDQA) across **five retrieval conditions**:

| Condition | Description |
|-----------|-------------|
| `no_retrieval` | Closed-book — model's parametric knowledge only |
| `rag_top3` | BM25 sparse retrieval, top-3 chunks (replicates SciDQA paper baseline) |
| `rag_top5` | BM25 sparse retrieval, top-5 chunks (extended baseline) |
| `rag_dense` | Dense semantic retrieval via `all-MiniLM-L6-v2`, top-3 chunks (thesis contribution) |
| `long_context` | Full paper text up to 140,000 characters |

**Chunking**: Paragraph-aware sentence-level sliding window (10 sentences, 1 overlap) matching SciDQA paper Algorithm 1.

**Inline metrics per record**: ROUGE-1/2/L/avg, n-gram grounding score, no-answer signal, response length, latency, token counts.

### 2. MMLU — Massive Multitask Language Understanding

Evaluates 570 questions across 57 subjects in both **English** and **German** (translated dataset). Structured JSON-output prompt forces a single-letter answer (A/B/C/D).

---

## Repository Structure

```
Thesis/
├── SciDQA/
│   ├── scidqa_gemma4.py          # Full evaluation — Gemma-4
│   ├── scidqa_gptoss.py          # Full evaluation — GPT-OSS
│   ├── scidqa_qwen3.5.py         # Full evaluation — Qwen 3.5
│   ├── run_batches_scidqa_*.py   # Per-model batch runners (500-question batches)
│   ├── run_all_scidqa.py         # Master orchestration (runs all 3 models sequentially)
│   ├── retry_errors_gemma4.py    # Retries 502 errors for Gemma-4
│   ├── retry_bad_qwen_v1v4.py    # Retries null-content records for Qwen (v1–v4)
│   ├── scidqa_pilot_gptoss.py    # Pilot script (100–500 questions, used for development)
│   └── data/                     # SciDQADataset.xlsx + papers_fulltext_nougat.pkl (not in repo)
│
├── MMLU/
│   ├── mmlu_english_gemma4.py    # MMLU English — Gemma-4
│   ├── mmlu_english_gptoss.py    # MMLU English — GPT-OSS
│   ├── mmlu_english_qwen3.5.py   # MMLU English — Qwen 3.5
│   └── run_batches.py            # Batch runner for English MMLU
│
├── MMLU German/
│   ├── german_gemma4.py          # MMLU German — Gemma-4
│   ├── german_gptoss.py          # MMLU German — GPT-OSS
│   ├── german_qwen3.5.py         # MMLU German — Qwen 3.5
│   └── run_batches_german.py     # Batch runner for German MMLU
│
└── analysis/
    └── combine_results.py        # Merges batches, deduplicates, generates accuracy report
```

---

## Setup

```bash
pip install openai pandas rank-bm25 rouge-score sentence-transformers nltk tqdm openpyxl datasets
```

Set your API key:
```bash
export LITELLM_API_KEY=your_key_here
```

---

## Running SciDQA

### Single batch (500 questions)
```bash
python3 SciDQA/scidqa_gemma4.py --offset 0 --n 500
```

### Full dataset via batch runner
```bash
caffeinate python3 SciDQA/run_batches_scidqa_gemma4.py
```

### All three models sequentially (overnight)
```bash
caffeinate python3 SciDQA/run_all_scidqa.py
```

### Retry failed records
```bash
python3 SciDQA/retry_errors_gemma4.py        # Gemma-4 502 errors
python3 SciDQA/retry_bad_qwen_v1v4.py        # Qwen null-content records
```

---

## Running MMLU

```bash
python3 MMLU/mmlu_english_gemma4.py          # English
python3 "MMLU German/german_gemma4.py"       # German
```

---

## Key Design Decisions

- **Rate limiting**: Token-bucket rate limiter shared across threads (200–300 req/min depending on model)
- **Concurrency**: `ThreadPoolExecutor` with 40 workers; threads block on the rate limiter
- **State management**: Batch runners persist progress to JSON state files and resume automatically
- **Error handling**: `sys.exit(1)` on 401/404 errors; 502s are retried with exponential backoff
- **Qwen thinking mode**: Qwen 3.5 returns answers in `reasoning_content` when `content` is null; the script falls back gracefully and records are flagged with `is_retry=True` after targeted reruns
