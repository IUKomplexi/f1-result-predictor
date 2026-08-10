"""Small shared helpers for the text reports (markdown rendering, ranking).

Kept dependency-free (pandas only) so ``predict``, ``model.evaluate`` and
``model.calibrate`` can all render and rank identically instead of each
re-implementing the same two functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without the `tabulate` package."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep, *body])


def rank_by(df: pd.DataFrame, score_col: str, tiebreak_col: str) -> pd.Series:
    """Rank rows by descending score, ascending tiebreak.

    Non-positive tiebreak values (e.g. ``grid=0`` for pit-lane starts, or a
    missing position) are treated as the worst — they sort last, never first.

    The returned Series is aligned to ``df``'s index (rank 1 = best), so it
    can be boolean-indexed against ``df`` and compared across rows.
    """
    tiebreak = (
        df[tiebreak_col]
        .replace(0, np.inf)
        .fillna(np.inf)
    )
    ranked = (
        df.assign(_tiebreak=tiebreak)
        .sort_values([score_col, "_tiebreak"], ascending=[False, True])
    )
    ranks = pd.Series(range(1, len(df) + 1), index=ranked.index)
    return ranks.reindex(df.index)
