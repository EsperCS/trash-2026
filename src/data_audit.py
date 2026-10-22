#!/usr/bin/env python3
"""Comprehensive data audit script for the jailbreak detection dataset.

Loads train.csv and test.csv, inspects schema, distributions, duplicates,
and text-length statistics. Saves a structured Markdown report to
reports/data_audit.md.

Usage
-----
    python src/data_audit.py --data_dir data
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name in *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _section(title: str, level: int = 2) -> str:
    """Return a Markdown heading."""
    return f"\n{'#' * level} {title}\n"


def _basic_info(df: pd.DataFrame, name: str) -> str:
    """Return a Markdown block with basic DataFrame info."""
    lines: list[str] = [_section(f"{name} — Basic Info")]
    lines.append(f"- **Rows:** {len(df):,}")
    lines.append(f"- **Columns:** {len(df.columns)}")
    lines.append(f"- **Column names:** `{'`, `'.join(df.columns)}`")
    lines.append("")

    # dtypes table
    lines.append("| Column | Dtype | Nulls | Null % |")
    lines.append("|--------|-------|-------|--------|")
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = null_count / len(df) * 100 if len(df) > 0 else 0.0
        lines.append(
            f"| `{col}` | {df[col].dtype} | {null_count:,} | {null_pct:.2f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _duplicate_info(df: pd.DataFrame, text_col: Optional[str], name: str) -> str:
    """Exact and near-duplicate analysis."""
    lines: list[str] = [_section(f"{name} — Duplicate Analysis")]

    # Exact row duplicates
    exact_dups = int(df.duplicated().sum())
    lines.append(f"- **Exact duplicate rows:** {exact_dups:,}")

    if text_col and text_col in df.columns:
        text_dups = int(df[text_col].duplicated().sum())
        lines.append(f"- **Duplicate `{text_col}` values:** {text_dups:,}")

        # Near-duplicates: same text after lowering + stripping whitespace
        normed = df[text_col].astype(str).str.lower().str.strip()
        near_dups = int(normed.duplicated().sum())
        lines.append(
            f"- **Near-duplicate `{text_col}` (lowered+stripped):** {near_dups:,}"
        )
    lines.append("")
    return "\n".join(lines)


def _class_distribution(df: pd.DataFrame, label_col: Optional[str], name: str) -> str:
    """Class distribution analysis."""
    lines: list[str] = [_section(f"{name} — Class Distribution")]
    if label_col is None or label_col not in df.columns:
        lines.append("_No label column detected._\n")
        return "\n".join(lines)

    counts: Counter = Counter(df[label_col].dropna())
    total = sum(counts.values())
    lines.append("| Label | Count | Proportion |")
    lines.append("|-------|-------|------------|")
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{label}` | {cnt:,} | {cnt / total:.4f} |")
    lines.append("")
    return "\n".join(lines)


def _text_length_stats(df: pd.DataFrame, text_col: Optional[str], name: str) -> str:
    """Character- and word-level length statistics."""
    lines: list[str] = [_section(f"{name} — Text Length Statistics")]
    if text_col is None or text_col not in df.columns:
        lines.append("_No text column detected._\n")
        return "\n".join(lines)

    texts = df[text_col].astype(str)
    char_lens = texts.str.len()
    word_lens = texts.str.split().str.len()

    for stat_name, series in [("Char length", char_lens), ("Word count", word_lens)]:
        lines.append(f"**{stat_name}**\n")
        lines.append(f"- mean: {series.mean():.1f}")
        lines.append(f"- std:  {series.std():.1f}")
        lines.append(f"- min:  {series.min()}")
        lines.append(f"- 25%:  {series.quantile(0.25):.0f}")
        lines.append(f"- 50%:  {series.quantile(0.50):.0f}")
        lines.append(f"- 75%:  {series.quantile(0.75):.0f}")
        lines.append(f"- 95%:  {series.quantile(0.95):.0f}")
        lines.append(f"- max:  {series.max()}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def run_audit(data_dir: str = "data") -> str:
    """Run the full data audit and return the Markdown report string.

    Parameters
    ----------
    data_dir : str
        Directory containing ``train.csv`` and ``test.csv``.

    Returns
    -------
    str
        Full Markdown report.
    """
    data_path = Path(data_dir)
    report_parts: list[str] = ["# Data Audit Report\n"]

    text_candidates = ["prompt", "text", "content", "query", "message"]
    label_candidates = ["label", "target", "is_jailbreak", "class"]

    for csv_name in ("train.csv", "test.csv"):
        csv_path = data_path / csv_name
        if not csv_path.exists():
            logger.warning("File not found: %s — skipping.", csv_path)
            report_parts.append(f"\n## {csv_name}\n\n_File not found._\n")
            continue

        logger.info("Loading %s …", csv_path)
        df = pd.read_csv(csv_path)
        logger.info("Loaded %s — shape %s", csv_name, df.shape)

        text_col = _detect_column(df, text_candidates)
        label_col = _detect_column(df, label_candidates)

        logger.info("Detected text column: %s", text_col)
        logger.info("Detected label column: %s", label_col)

        report_parts.append(_basic_info(df, csv_name))
        report_parts.append(_duplicate_info(df, text_col, csv_name))
        report_parts.append(_class_distribution(df, label_col, csv_name))
        report_parts.append(_text_length_stats(df, text_col, csv_name))

    return "\n".join(report_parts)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run data audit on CSV files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing train.csv and test.csv (default: data).",
    )
    args = parser.parse_args()

    report = run_audit(data_dir=args.data_dir)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_audit.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Audit report saved to %s", out_path)


if __name__ == "__main__":
    main()
