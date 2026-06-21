# LLM Evaluation for Scientific QA and MMLU

Empirical evaluation conducted as part of an M.Sc. thesis at Universität Osnabrück. The study benchmarks three open-source large language models across two tasks, comparing response quality under different retrieval configurations and measuring cross-language generalization.

## Models

| Model | Parameters |
|-------|-----------|
| Gemma-4 31B (FP8) | 31B |
| GPT-OSS 120B | 120B |
| Qwen 3.5 122B (FP8) | 122B |

All models are accessed through a LiteLLM-compatible API gateway.

## Benchmarks

### SciDQA — Scientific Document QA

2,937 questions from the [SciDQA dataset](https://github.com/yale-nlp/SciDQA), each requiring answers grounded in a specific scientific paper. Models are evaluated across five retrieval conditions:

| Condition | Description |
|-----------|-------------|
| `no_retrieval` | Closed-book — model's parametric knowledge only |
| `rag_top3` | BM25 sparse retrieval, top-3 chunks |
| `rag_top5` | BM25 sparse retrieval, top-5 chunks |
| `rag_dense` | Dense semantic retrieval (`all-MiniLM-L6-v2`), top-3 chunks |
| `long_context` | Full paper text (~140k characters) |

Chunking uses a paragraph-aware sliding window (10 sentences, 1 sentence overlap).

**Evaluation metrics per response:** ROUGE-1/2/L, BERTScore (RoBERTa and SciBERT), NLI faithfulness, LLM-as-judge scores (cross-reference design), and a mismatch condition measuring hallucination behavior under deliberately incorrect context.

### MMLU — Multitask Language Understanding

570 questions across 57 subjects in English and German (translated via the [MMMLU dataset](https://huggingface.co/datasets/openai/MMMLU)). Tests factual knowledge and cross-language consistency.

## Repository Structure

```
Thesis/
├── SciDQA/
│   ├── scidqa_gemma4.py           # Evaluation — Gemma-4
│   ├── scidqa_gptoss.py           # Evaluation — GPT-OSS
│   ├── scidqa_qwen3.5.py          # Evaluation — Qwen 3.5
│   ├── scidqa_mismatch.py         # Hallucination/mismatch condition
│   ├── dataset.py                 # Dataset loading and chunking
│   ├── run_batches_scidqa_*.py    # Batch orchestration per model
│   └── run_all_scidqa.py          # Runs all three models sequentially
│
├── MMLU/
│   ├── mmlu_english_gemma4.py     # English MMLU — Gemma-4
│   ├── mmlu_english_gptoss.py     # English MMLU — GPT-OSS
│   ├── mmlu_english_qwen3.5.py    # English MMLU — Qwen 3.5
│   └── run_batches.py             # Batch orchestration
│
├── MMLU German/
│   ├── german_gemma4.py           # German MMLU — Gemma-4
│   ├── german_gptoss.py           # German MMLU — GPT-OSS
│   ├── german_qwen3.5.py          # German MMLU — Qwen 3.5
│   └── run_batches_german.py      # Batch orchestration
│
└── analysis/
    ├── combine_scidqa.py          # Merges SciDQA batches, deduplicates
    ├── combine_results.py         # Merges MMLU batches, deduplicates
    ├── bertscore_eval.py          # BERTScore (RoBERTa + SciBERT)
    ├── nli_faithfulness.py        # NLI faithfulness (RAG conditions)
    ├── llm_judge.py               # LLM-as-judge cross-reference evaluation
    ├── cross_language_analysis.py # EN/DE accuracy comparison
    └── thesis_tables.py           # Summary tables from all results
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API credentials:

```bash
cp .env.example .env
```

```
LITELLM_BASE_URL=https://your-litellm-gateway/v1
LITELLM_API_KEY=your_api_key_here
```

## Running Evaluations

### SciDQA

```bash
python3 SciDQA/run_batches_scidqa_gemma4.py   # full dataset, single model
python3 SciDQA/run_all_scidqa.py              # all three models
```

### MMLU

```bash
python3 MMLU/mmlu_english_gemma4.py
python3 "MMLU German/german_gemma4.py"
```

### Analysis

```bash
cd analysis
python3 combine_results.py        # merge and deduplicate raw outputs
python3 bertscore_eval.py         # compute BERTScore
python3 nli_faithfulness.py       # NLI faithfulness (RAG conditions only)
python3 llm_judge.py              # LLM-as-judge (cross-reference)
python3 thesis_tables.py --save   # generate summary tables
```

## Design Notes

- **Concurrency**: `ThreadPoolExecutor` with per-model token-bucket rate limiters to stay within API quotas
- **Fault tolerance**: All scripts resume from the last successful record; transient errors are retried with exponential backoff
- **LLM-as-judge design**: Cross-reference only — each model is scored by the other two, never itself, to reduce self-preference bias
- **Qwen thinking mode**: Qwen 3.5 is prompted with `/no_think` during judging to suppress chain-of-thought output and stay within the token budget
