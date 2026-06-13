"""
dataset.py
----------
Downloads and loads the SciDQA Reading Comprehension dataset from OSF.
Source: https://osf.io/3g2hn/overview  (shared by the SciDQA authors)

Dataset structure (once downloaded to ./data/):
───────────────────────────────────────────────
SciDQADataset.xlsx          2937 QA pairs with paper references
  Columns: id, year, venue, rid, pid, decision, que, ans, version

papers_fulltext_nougat.pkl  Full paper texts (Nougat OCR, Markdown format)
  Structure: dict['initial' | 'final'][pid] → str (paper text)
  'initial' = pre-review version, 'final' = accepted version

relevant_pft.pkl            Structured paper metadata + sections per paper
  Structure: dict[pid] → {'name', 'metadata': {sections, refs, ...}, 'year', 'conf'}

relevant_ptabs.pkl          Relevant tables per paper
  Structure: dict[pid] → {...}

SciDQA_MuliDocQA.xlsx       316 multi-document QA pairs (separate task)

Linking questions to paper text:
  version_key = 'initial' if row['version'] == 'Initial' else 'final'
  paper_text  = fulltext[version_key][row['pid']]

Usage:
  python3 dataset.py                # download + inspect structure
  python3 dataset.py --no-download  # inspect already-downloaded files only
  python3 dataset.py --sample 3     # show 3 example QA + paper text pairs
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import textwrap
import urllib.request

import pandas as pd

# ── OSF download URLs ──────────────────────────────────────────────────────────
OSF_FILES = {
    "SciDQADataset.xlsx"         : "https://osf.io/download/bvpjw/",
    "SciDQA_MuliDocQA.xlsx"      : "https://osf.io/download/h9uym/",
    "papers_fulltext_nougat.pkl" : "https://osf.io/download/62jhw/",
    "relevant_pft.pkl"           : "https://osf.io/download/8ezjw/",
    "relevant_ptabs.pkl"         : "https://osf.io/download/5v8bm/",
    # model_len_rag_chunks.pkl is 301 MB — add manually if needed for RAG
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

# Maps dataset 'version' column → fulltext dict key
VERSION_MAP = {"Initial": "initial", "Revised": "final"}


# ── Download ───────────────────────────────────────────────────────────────────

def _progress_hook(filename: str):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb  = downloaded / 1_048_576
            sys.stdout.write(f"\r  {filename}: {pct:3d}%  ({mb:.1f} MB)")
            sys.stdout.flush()
        if total_size > 0 and block_num * block_size >= total_size:
            print()
    return hook


def download_files(skip_existing: bool = True) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Saving to: {DATA_DIR}\n")
    for fname, url in OSF_FILES.items():
        dest = os.path.join(DATA_DIR, fname)
        if skip_existing and os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1_048_576
            print(f"  [skip] {fname} already exists ({size_mb:.1f} MB)")
            continue
        print(f"  Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(fname))
        except Exception as e:
            print(f"  [ERROR] Failed to download {fname}: {e}")


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """Load the main SciDQA QA pairs (2937 rows)."""
    return pd.read_excel(os.path.join(DATA_DIR, "SciDQADataset.xlsx"))


def load_fulltext() -> dict:
    """
    Load paper full texts (Nougat OCR, Markdown).
    Returns: dict['initial'|'final'][pid] → str
    """
    with open(os.path.join(DATA_DIR, "papers_fulltext_nougat.pkl"), "rb") as f:
        return pickle.load(f)


def load_relevant_pft() -> dict:
    """
    Load structured paper metadata + sections.
    Returns: dict[pid] → {'name', 'metadata': {sections, references, ...}, 'year', 'conf'}
    """
    with open(os.path.join(DATA_DIR, "relevant_pft.pkl"), "rb") as f:
        return pickle.load(f)


def load_relevant_ptabs() -> dict:
    """
    Load relevant tables per paper.
    Returns: dict[pid] → {...}
    """
    with open(os.path.join(DATA_DIR, "relevant_ptabs.pkl"), "rb") as f:
        return pickle.load(f)


def load_multidoc() -> pd.DataFrame:
    """Load the multi-document QA variant (316 rows)."""
    return pd.read_excel(os.path.join(DATA_DIR, "SciDQA_MuliDocQA.xlsx"))


# ── Main accessor: get paper text for a question row ──────────────────────────

def get_paper_text(row: pd.Series, fulltext: dict) -> str | None:
    """
    Given a dataset row and the fulltext dict, return the paper text.
    Returns None if the paper is not found.
    """
    version_key = VERSION_MAP.get(row["version"], "initial")
    return fulltext.get(version_key, {}).get(row["pid"])


def get_structured_sections(row: pd.Series, pft: dict) -> list:
    """
    Return the structured sections list for a question's paper from relevant_pft.
    Each section: {'heading': str | None, 'text': str}
    """
    entry = pft.get(row["pid"])
    if entry is None:
        return []
    return entry.get("metadata", {}).get("sections", [])


# ── Inspection ─────────────────────────────────────────────────────────────────

def inspect_all() -> None:
    df       = load_dataset()
    fulltext = load_fulltext()
    pft      = load_relevant_pft()

    print("\n" + "═" * 65)
    print("  SciDQADataset.xlsx")
    print("═" * 65)
    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {', '.join(df.columns.tolist())}")
    print(f"  Versions: {df['version'].value_counts().to_dict()}")
    print(f"  Venues  : {df['venue'].value_counts().to_dict()}")

    print("\n" + "═" * 65)
    print("  papers_fulltext_nougat.pkl")
    print("═" * 65)
    for v in ["initial", "final"]:
        print(f"  version='{v}': {len(fulltext[v])} papers")
    sample_pid = list(fulltext["initial"].keys())[0]
    print(f"  Text format : plain Markdown string (from Nougat OCR)")
    print(f"  Sample text : {fulltext['initial'][sample_pid][:150]}…")

    print("\n" + "═" * 65)
    print("  relevant_pft.pkl  (structured sections per paper)")
    print("═" * 65)
    print(f"  Papers    : {len(pft)}")
    sample_pft = pft[list(pft.keys())[0]]
    sections   = sample_pft.get("metadata", {}).get("sections", [])
    print(f"  Sample keys     : {list(sample_pft.keys())}")
    print(f"  Sections in sample paper : {len(sections)}")
    if sections:
        print(f"  First section heading : {sections[0].get('heading')}")
        print(f"  First section text    : {sections[0].get('text', '')[:120]}…")

    print("\n" + "═" * 65)
    print("  Linking test: question → paper text")
    print("═" * 65)
    sample_row = df.iloc[0]
    text = get_paper_text(sample_row, fulltext)
    print(f"  Question id  : {sample_row['id']}")
    print(f"  pid          : {sample_row['pid']}")
    print(f"  version      : {sample_row['version']} → '{VERSION_MAP[sample_row['version']]}'")
    print(f"  Paper found  : {text is not None}")
    if text:
        print(f"  Paper length : {len(text):,} chars")
        print(f"  Question     : {sample_row['que']}")
        print(f"  Answer       : {sample_row['ans']}")


def show_samples(n: int = 3) -> None:
    df       = load_dataset()
    fulltext = load_fulltext()

    print(f"\nShowing {n} sample QA + paper context pairs\n")
    for _, row in df.head(n).iterrows():
        text = get_paper_text(row, fulltext)
        print("─" * 65)
        print(f"  id      : {row['id']}  |  venue: {row['venue']} {row['year']}  |  pid: {row['pid']}")
        print(f"  Q: {row['que']}")
        print(f"  A: {row['ans']}")
        if text:
            snippet = text[:400].replace("\n", " ")
            print(f"  Paper snippet: {snippet}…")
        else:
            print("  Paper: [not found]")
    print("─" * 65)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading; inspect already-present files only")
    parser.add_argument("--sample", type=int, default=0,
                        help="Show N example QA + paper text pairs (default: 0 = skip)")
    args = parser.parse_args()

    if not args.no_download:
        print("Downloading SciDQA RC dataset files from OSF...\n")
        download_files()

    missing = [f for f in ["SciDQADataset.xlsx", "papers_fulltext_nougat.pkl", "relevant_pft.pkl"]
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"\n[ERROR] Missing files: {missing}")
        print("Run without --no-download to fetch them.")
        return

    print("\nInspecting downloaded files...")
    inspect_all()

    if args.sample > 0:
        show_samples(args.sample)

    print("\n" + "═" * 65)
    print("  Ready. Import this module in your evaluation scripts:")
    print("    from dataset import load_dataset, load_fulltext, get_paper_text")
    print("═" * 65)


if __name__ == "__main__":
    main()
