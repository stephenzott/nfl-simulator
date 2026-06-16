"""
matchup.py — simulate a single fantasy matchup and return win probability.

Quick start
-----------
    from matchup import simulate_matchup

    result = simulate_matchup(roster_a, roster_b, player_params, n=10_000)
    print(f"P(A wins) = {result['p_win_a']:.3f} ± {result['se']:.3f}")

Bye-week handling
-----------------
Pass ``bye_teams`` as a list of team abbreviations (e.g. ``["KC", "MIA"]``)
to automatically zero out players whose ``"team"`` field matches.  Players on
bye are *removed* before calling ``simulate_team_score`` so that π is not
double-counted (π was estimated excluding bye weeks — see fit.py and the
simulate.py module docstring).

Roster dict schema
------------------
Each element of roster_a / roster_b must be a dict with at least:

    {"player_id": str, "team": str}   # "team" only required if bye_teams used

Extra keys (player_name, position, slot, etc.) are silently ignored.
"""

from typing import Union

import numpy as np
import pandas as pd

from simulate import simulate_team_score


def simulate_matchup(
    roster_a: list[dict],
    roster_b: list[dict],
    player_params: pd.DataFrame,
    n: int = 10_000,
    bye_teams: Union[list[str], None] = None,
    rng: Union[int, np.random.Generator, None] = None,
) -> dict:
    """Simulate N matchup trials and return win probability for Team A.

    Parameters
    ----------
    roster_a : list[dict]
        Players on Team A.  Each dict must contain ``"player_id"`` and, if
        ``bye_teams`` is provided, ``"team"``.
    roster_b : list[dict]
        Players on Team B.  Same schema as ``roster_a``.
    player_params : pd.DataFrame
        Output of ``fit.fit_player_params()``.  Required columns:
        ``player_id``, ``alpha``, ``beta``, ``pi``.
    n : int, optional
        Number of Monte Carlo trials.  Default 10 000 gives SE < 0.005 for
        win probabilities in (0.1, 0.9).
    bye_teams : list[str] or None, optional
        Team abbreviations on bye this matchup week (e.g. ``["KC", "MIA"]``).
        Players whose ``"team"`` field matches will be removed before
        simulation.  If None, no bye filtering is applied.
    rng : int, Generator, or None, optional
        Random-number source.  Pass an integer for reproducible results.
        The same RNG instance is shared between Team A and Team B draws so
        that the two score arrays are statistically independent (different
        RNG state) but the overall simulation is reproducible from a single
        seed.

    Returns
    -------
    dict
        Keys:

        ``p_win_a`` : float
            Estimated probability that Team A's total score exceeds Team B's.
            Ties count as a loss for A (strict ``>``), matching standard
            fantasy scoring tie-break rules (higher score wins).
        ``se`` : float
            Binomial standard error of the win-probability estimate:
            ``sqrt(p̂(1 − p̂) / n)``.  At n=10 000 and p̂=0.5 this is ≈0.005.
        ``mean_score_a`` : float
            Expected total PPR score for Team A across all trials.
        ``mean_score_b`` : float
            Expected total PPR score for Team B across all trials.
        ``score_dist_a`` : np.ndarray
            Shape ``(n,)``.  Full score distribution for Team A — use this
            to compute percentiles, plot histograms, or chain into a season
            simulation.
        ``score_dist_b`` : np.ndarray
            Shape ``(n,)``.  Full score distribution for Team B.

    Notes
    -----
    Shared RNG
    ----------
    Both team draws use the *same* ``np.random.Generator`` instance, passed
    sequentially.  This is correct: Team A exhausts some RNG state, then Team
    B draws from the *next* state.  The two score arrays are therefore
    statistically independent draws from the same underlying stream —
    equivalent to drawing from two separate seeded generators.

    Why a single seed still makes the full result reproducible: given the
    same seed, ``np.random.default_rng(seed)`` always produces the same
    sequence of floats.  Consuming that sequence for A first, then B, yields
    the same A and B arrays every time.

    Tie handling
    ------------
    ``scores_a > scores_b`` is strict.  A tie (identical floats) is a loss
    for A.  In practice, ties in Gamma-draw sums are vanishingly rare (the
    distributions are continuous), but the strict inequality matches how most
    fantasy platforms resolve ties (higher score wins; if truly tied, the
    platform typically awards the win to neither or uses a secondary rule).

    Bye filtering
    -------------
    Players in ``bye_teams`` are dropped *before* the simulation call, not
    zeroed out inside the simulation.  This is intentional: π already excludes
    bye weeks, so passing a bye player into ``simulate_team_score`` would give
    them a Gamma draw with probability (1 − π) rather than a guaranteed 0,
    overcounting their contribution.

    Standard error interpretation
    ------------------------------
    SE = sqrt(p̂(1 − p̂) / n).  This is the standard error of a sample
    proportion under the binomial model.  It tells you how much p̂ would
    vary across repeated independent simulations of the same n trials.
    At n=10 000 and p̂=0.6: SE ≈ 0.0049 → 95% CI ≈ [0.590, 0.610].
    """
    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif rng is None:
        rng = np.random.default_rng()

    # Filter out players on bye before simulation.  π excludes byes, so
    # passing a bye player into simulate_team_score would inflate their
    # expected contribution instead of zeroing it.
    if bye_teams:
        bye_set = set(bye_teams)
        roster_a = [p for p in roster_a if p.get("team") not in bye_set]
        roster_b = [p for p in roster_b if p.get("team") not in bye_set]

    scores_a = simulate_team_score(roster_a, player_params, n=n, rng=rng)
    scores_b = simulate_team_score(roster_b, player_params, n=n, rng=rng)

    p_hat = float((scores_a > scores_b).mean())
    se = float(np.sqrt(p_hat * (1.0 - p_hat) / n))

    return {
        "p_win_a":     p_hat,
        "se":          se,
        "mean_score_a": float(scores_a.mean()),
        "mean_score_b": float(scores_b.mean()),
        "score_dist_a": scores_a,
        "score_dist_b": scores_b,
    }
