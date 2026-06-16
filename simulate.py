"""
simulate.py — Monte Carlo team-score engine.

Quick start
-----------
    from simulate import simulate_team_score

    scores = simulate_team_score(roster, player_params, n=10_000, rng=42)
    # returns np.ndarray of shape (n,) — one total team score per trial

Bye-week contract
-----------------
π was estimated excluding bye weeks (see fit.py).  Players who are on bye
this matchup week must be removed from `roster` *before* calling this
function.  If a bye-week player is passed in, they will receive a Gamma draw
with probability (1 − π) instead of the correct deterministic 0, inflating
the team's simulated score.  matchup.py is responsible for the filter.
"""

from typing import Union

import numpy as np
import pandas as pd


def simulate_team_score(
    roster: list[dict],
    player_params: pd.DataFrame,
    n: int = 10_000,
    rng: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
    """Draw N simulated total team scores via zero-inflated Gamma sampling.

    Parameters
    ----------
    roster : list[dict]
        Players on this team for the matchup week.  Each dict must contain
        at least ``"player_id"``.  Extra keys (name, position, slot) are
        ignored.  Players on bye must be excluded by the caller (see module
        docstring).
    player_params : pd.DataFrame
        Output of ``fit.fit_player_params()``.  Required columns:
        ``player_id``, ``alpha``, ``beta``, ``pi``.
    n : int, optional
        Number of Monte Carlo trials.  Default 10 000 gives SE < 0.005 for
        all win probabilities in (0.1, 0.9).  Increase to 50 000 for tighter
        tail estimates (e.g. playoff bracket odds).
    rng : int, Generator, or None, optional
        Random-number source.  Pass an integer seed for reproducible results,
        an existing ``np.random.Generator`` to share state with a caller, or
        ``None`` (default) for a fresh unpredictable generator.

        Uses the modern PCG64 generator (``np.random.default_rng``), not the
        legacy global ``np.random.*`` functions.  This is thread-safe and
        ~20% faster than the legacy interface.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``.  Trial ``i`` is the sum of all roster players'
        simulated PPR scores in that trial.  The two team arrays produced
        by calling this function twice (once per team) are independent draws,
        which is correct under the current model (no inter-team correlation).
        Intra-team player correlations (QB–WR stack effects) are the Phase 2
        extension; at that point, replace the independent draws here with a
        Gaussian copula step.

    Raises
    ------
    ValueError
        If any ``player_id`` in ``roster`` is missing from ``player_params``
        or has NaN values for ``alpha``, ``beta``, or ``pi``.  This is raised
        rather than silently zeroing the player because a missing model fit
        would typically understate the team's expected score by 8–20 pts,
        producing quietly wrong win probabilities.

    Notes
    -----
    Sampling mechanism
    ------------------
    For each player j and trial i:

        active_{j,i} ~ Bernoulli(1 − π_j)          (did the player score?)
        gamma_{j,i}  ~ Gamma(α_j, β_j)              (how much, if active)
        x_{j,i}      = active_{j,i} × gamma_{j,i}

    where β_j is the *scale* parameter (numpy's ``scale`` argument).  The
    numpy convention is ``np.random.Generator.gamma(shape=α, scale=β)``,
    which matches our parameterization: E[X] = αβ, Var[X] = αβ².

    Vectorization
    -------------
    All player draws are computed in a single ``(n_players, n)`` NumPy
    operation rather than a Python loop over players.  At a typical roster
    size of 9 players × 10 000 trials, the performance difference is
    negligible, but the vectorised form scales cleanly to batch simulations
    (e.g. simulating all Week 1 matchups simultaneously).

    Independence assumption
    -----------------------
    Player scores are drawn independently — no covariance between teammates
    or opponents.  Known implication: QB–WR stack value is understated, and
    correlated busts (e.g. a blizzard suppressing an entire offense) are not
    captured.  See CLAUDE.md Phase 2 notes for the copula extension.
    """
    if not roster:
        return np.zeros(n)

    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Parameter lookup
    # ------------------------------------------------------------------
    param_index = player_params.set_index("player_id")

    missing, nan_params = [], []
    alphas, betas, pis = [], [], []

    for player in roster:
        pid = player["player_id"]
        if pid not in param_index.index:
            missing.append(pid)
            continue
        row = param_index.loc[pid]
        if any(np.isnan([row["alpha"], row["beta"], row["pi"]])):
            nan_params.append(pid)
            continue
        alphas.append(float(row["alpha"]))
        betas.append(float(row["beta"]))
        pis.append(float(row["pi"]))

    errors = []
    if missing:
        errors.append(f"player_ids not found in player_params: {missing}")
    if nan_params:
        errors.append(
            f"player_ids have NaN alpha/beta/pi (no model fit — "
            f"check fit.py fallback coverage): {nan_params}"
        )
    if errors:
        raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Vectorized zero-inflated Gamma draws
    # ------------------------------------------------------------------
    alpha_arr = np.array(alphas)   # shape (n_players,)
    beta_arr  = np.array(betas)
    pi_arr    = np.array(pis)

    n_players = len(alpha_arr)

    # Gamma draw for every player × trial: shape (n_players, n)
    # numpy Gamma(shape=α, scale=β) matches our scale parameterization.
    gamma_draws = rng.gamma(
        shape=alpha_arr[:, None],
        scale=beta_arr[:, None],
        size=(n_players, n),
    )

    # Zero-inflation mask: True = player is active this trial (scores > 0).
    # We draw Uniform(0,1) and compare to π (the zero probability) rather than
    # using rng.binomial so that we only need one RNG call for the mask.
    active = rng.random((n_players, n)) >= pi_arr[:, None]

    # Team score per trial: sum across players
    return (active * gamma_draws).sum(axis=0)
