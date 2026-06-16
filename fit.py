"""
fit.py — estimate zero-inflated Gamma parameters per player via method of moments.

Quick start
-----------
    from data import pull_weekly_stats
    from fit import fit_player_params

    weekly = pull_weekly_stats([2022, 2023, 2024])
    params = fit_player_params(weekly)
    # Returns one row per player with columns:
    # player_id, player_name, position, alpha, beta, pi, mu, sigma,
    # n_games, n_nonzero, fallback_used
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# NFL regular seasons from 2022 onward span 18 scheduled weeks, but every team
# has exactly one bye week, so a player can appear in at most 17 games per
# season.  We use 17 as the per-season denominator when computing π so that bye
# weeks are excluded from the zero-inflation probability.
#
# Why exclude byes from π?  Because π is meant to capture "given the team has
# a game, what is the probability this player scores 0?" — bye weeks don't
# satisfy that condition.  The simulator (matchup.py) must zero-out players on
# bye using the schedule directly, NOT by drawing from the π Bernoulli.
# Including byes in π would cause the simulator to double-count them.
_GAMES_PER_SEASON: int = 17

# Minimum non-zero game weeks needed to fit Gamma parameters directly.
# With n_nonzero = 2 you can compute μ and σ, but σ is extremely noisy.
# CLAUDE.md specifies 4 as the fallback threshold; this is a pragmatic floor,
# not a mathematical one.
_MIN_NONZERO: int = 4


def fit_player_params(
    weekly_df: pd.DataFrame,
    min_nonzero: int = _MIN_NONZERO,
    cache: bool = True,
) -> pd.DataFrame:
    """Estimate (α, β, π) per player via method of moments on non-zero weeks.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Output of ``data.pull_weekly_stats()``. Required columns:
        ``player_id``, ``player_name``, ``position``, ``season``, ``ppr_score``.
    min_nonzero : int, optional
        Minimum non-zero-score weeks required for a direct Gamma fit.
        Players below this threshold (or with σ=0) fall back to the
        positional-median parameters.  Default: 4.
    cache : bool, optional
        If True, write the result to ``data/player_params.parquet``.

    Returns
    -------
    pd.DataFrame
        One row per unique ``player_id``. Columns:

        ============  ===========================================================
        player_id     Unique player identifier from nfl_data_py
        player_name   Most recent display name (handles mid-season trades)
        position      Most recent position (handles position changes)
        alpha         Gamma shape parameter  α = μ² / σ²
        beta          Gamma scale parameter  β = σ² / μ
        pi            Zero-inflation probability  π = (n_possible − n_nonzero) / n_possible
        mu            Mean of non-zero weekly scores
        sigma         Std dev of non-zero weekly scores (ddof=1)
        n_games       Total active weeks in the data (including 0-score weeks)
        n_nonzero     Weeks with ppr_score > 0 (used to fit Gamma)
        fallback_used True if positional-median values were substituted
        ============  ===========================================================

    Notes
    -----
    Statistical model
    -----------------
    Each player's weekly PPR score is modeled as a zero-inflated Gamma:

        X = 0              with probability π           (zero week)
        X ~ Gamma(α, β)    with probability 1 − π       (productive week)

    Zero weeks include: injury/IR, inactive/DNP decisions, mid-season cuts,
    and active-but-0-production games.  Bye weeks are *not* included (see
    _GAMES_PER_SEASON above).

    Method of moments
    -----------------
    For X ~ Gamma(α, β) with scale β:

        E[X]   = αβ  →  μ̂  = sample mean of non-zero scores
        Var[X] = αβ² →  σ̂² = sample variance of non-zero scores

    Solving:  β̂ = σ̂² / μ̂     α̂ = μ̂² / σ̂²

    We fit only on non-zero weeks because including zeros in the Gamma fit
    would artificially inflate σ̂², pushing β upward and α downward —
    producing a heavier-tailed distribution than the player's true productive-
    game output.

    π estimation
    ------------
    π̂ = (n_possible − n_nonzero) / n_possible

    n_possible = 17 × (number of seasons the player appeared in).
    n_nonzero  = number of rows with ppr_score > 0.

    Limitation: n_possible cannot distinguish a player who debuted in week 5
    from one who tore their ACL in week 5 — both show the same "missed weeks"
    count.  This is an inherent limitation of weekly-stats-only data.

    σ = 0 edge case
    ---------------
    If all n_nonzero scores are identical, σ = 0 → β = 0, α = ∞ (degenerate).
    These players are routed to the positional-median fallback identically to
    the n_nonzero < min_nonzero case.

    Positional median fallback
    --------------------------
    Median α, β, π, μ, σ are computed from the well-fit cohort (n_nonzero ≥
    min_nonzero AND σ > 0) only, to avoid circular contamination by the same
    low-quality fits we are trying to repair.
    """
    # Sort so iloc[-1] captures the most recent name/position per player
    # (handles mid-season trades and position changes, e.g. RB→WR)
    df = weekly_df.sort_values(["player_id", "season", "week"])

    records: list[dict] = []

    for player_id, grp in df.groupby("player_id", sort=False):
        player_name = grp["player_name"].iloc[-1]
        position    = grp["position"].iloc[-1]

        n_seasons  = grp["season"].nunique()
        n_possible = _GAMES_PER_SEASON * n_seasons

        nonzero   = grp.loc[grp["ppr_score"] > 0, "ppr_score"]
        n_nonzero = len(nonzero)
        n_games   = len(grp)   # active weeks, including 0-score rows

        # Compute μ and σ even for low-n players so we can detect σ=0
        if n_nonzero >= 2:
            mu    = float(nonzero.mean())
            # ddof=1 (unbiased): with n=4 games, ddof=0 would underestimate
            # σ² by a factor of 3/4, inflating α and shrinking β — understating
            # how spread out the player's scores really are.
            sigma = float(nonzero.std(ddof=1))
        else:
            mu    = np.nan
            sigma = np.nan

        # π̂: fraction of possible weeks with zero production.
        # Clip to [0, 1] as a guard against any floating-point surprise.
        pi = float(np.clip((n_possible - n_nonzero) / n_possible, 0.0, 1.0))

        # Flag for fallback: too few data points, or degenerate Gamma (σ=0)
        needs_fallback = (n_nonzero < min_nonzero) or (sigma == 0.0)

        if needs_fallback:
            alpha, beta = np.nan, np.nan
        else:
            beta  = sigma ** 2 / mu
            alpha = mu ** 2 / sigma ** 2

        records.append({
            "player_id":    player_id,
            "player_name":  player_name,
            "position":     position,
            "alpha":        alpha,
            "beta":         beta,
            "pi":           pi,
            "mu":           mu,
            "sigma":        sigma,
            "n_games":      n_games,
            "n_nonzero":    n_nonzero,
            "fallback_used": needs_fallback,
        })

    params = pd.DataFrame(records)

    # --- Positional median fallback -------------------------------------------
    # Compute medians from the well-fit cohort only.  Including fallback players
    # in the median would contaminate the reference distribution with the same
    # low-data / degenerate cases we are trying to repair.
    well_fit = params[~params["fallback_used"]]
    pos_medians = (
        well_fit
        .groupby("position")[["alpha", "beta", "pi", "mu", "sigma"]]
        .median()
        .add_suffix("_med")
        .reset_index()
    )

    params = params.merge(pos_medians, on="position", how="left")

    for col in ["alpha", "beta", "pi", "mu", "sigma"]:
        med_col = f"{col}_med"
        mask = params["fallback_used"] & params[med_col].notna()
        params.loc[mask, col] = params.loc[mask, med_col]

    params = params.drop(columns=[c for c in params.columns if c.endswith("_med")])

    if cache:
        DATA_DIR.mkdir(exist_ok=True)
        params.to_parquet(DATA_DIR / "player_params.parquet", index=False)

    return params.reset_index(drop=True)
