"""Grouped holdout and K-fold assignment.

freMTPL2 has no policy dates, so out-of-time validation is impossible. The substitute is a random
holdout plus K-fold CV on the remainder, drawn at the level of *profile groups* (rows with identical
rating factors) so that near-duplicate policies cannot sit on both sides of a split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLDOUT_FOLD = -1


def assign_split(group_ids: np.ndarray | pd.Series, holdout_fraction: float, n_folds: int, seed: int) -> pd.DataFrame:
    """Return one row per distinct group: group_id, is_holdout, fold (-1 for holdout groups)."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    ids = np.sort(pd.unique(np.asarray(group_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    n_holdout = round(len(perm) * holdout_fraction)
    holdout, train = perm[:n_holdout], perm[n_holdout:]
    folds = np.arange(len(train)) % n_folds
    out = pd.concat(
        [
            pd.DataFrame({"group_id": holdout, "is_holdout": True, "fold": HOLDOUT_FOLD}),
            pd.DataFrame({"group_id": train, "is_holdout": False, "fold": folds}),
        ],
        ignore_index=True,
    )
    out["group_id"] = out["group_id"].astype("int64")
    out["fold"] = out["fold"].astype("int64")
    return out.sort_values("group_id", ignore_index=True)
