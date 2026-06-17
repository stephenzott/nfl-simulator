# NFL Fantasy Monte Carlo Simulator — Project Context

> Park this file (`CLAUDE.md`) in the root of the project folder.
> Claude Code will automatically read it as project context.

---

## Project Goal

Build a Python-based Monte Carlo simulator for the 2026 NFL season to estimate
**week-by-week fantasy matchup win probabilities** for a Full PPR redraft league.

---

## Statistical Model

### Player Scoring Distribution

Each player's weekly PPR score is modeled as a **zero-inflated Gamma distribution**:

```
X ~ 0           with probability π          (DNP / bye / injury)
X ~ Gamma(α, β) with probability (1 - π)    (active week)
```

Parameters are estimated via **method of moments** on non-zero weeks only:

```
β̂ = σ̂² / μ̂        (scale)
α̂ = μ̂² / σ̂²       (shape)
π̂ = (n_possible − n_nonzero) / n_possible
    where n_possible = 17 × n_seasons  (bye weeks excluded)
```

### Simulation Step (one trial)

For a matchup between Team A and Team B with N simulation trials:

```
S_A^(i) = Σ_{j ∈ A} x_j^(i),    x_j^(i) ~ ZeroInflatedGamma(α̂_j, β̂_j, π̂_j)
S_B^(i) = Σ_{k ∈ B} x_k^(i),    x_k^(i) ~ ZeroInflatedGamma(α̂_k, β̂_k, π̂_k)

P̂(A wins) = (1/N) Σ 1[S_A^(i) > S_B^(i)]
SE = sqrt(p̂(1 - p̂) / N)
```

Recommended N = 10,000. At p̂ = 0.6, SE ≈ ±0.005.

### Planned Extensions (not yet implemented)

- **Correlated draws**: Model intra-team player correlations via a covariance
  matrix + Gaussian copula (e.g. QB–WR stack effects). When this is added,
  replace independent Gamma draws with a correlated multivariate draw mapped
  back to Gamma marginals.

---

## Scoring System

**Full PPR** — standard scoring plus 1 point per reception.

```python
PPR_SCORING = {
    "passing_yards":      0.04,   # per yard
    "passing_tds":        4.0,
    "interceptions":     -2.0,
    "rushing_yards":      0.1,    # per yard
    "rushing_tds":        6.0,
    "receiving_yards":    0.1,    # per yard
    "receiving_tds":      6.0,
    "receptions":         1.0,    # full PPR
    "fumbles_lost":      -2.0,
    "two_pt_conversions": 2.0,
}
```

---

## Data Sources

Two libraries are used depending on the season year. Both require no API key.

```bash
pip install nfl_data_py nflreadpy pandas numpy scipy matplotlib
```

### nfl_data_py — seasons 2022–2024

| Table | Call | Purpose |
|---|---|---|
| Weekly player stats | `nfl.import_weekly_data([year])` | Fit α, β, π per player |
| Season rosters | `nfl.import_seasonal_rosters([year])` | Map players to teams |
| Schedule | `nfl.import_schedules([year])` | Week-by-week matchups |

### nflreadpy — seasons 2025+

`data.py` dispatches to `nflreadpy.load_player_stats(seasons=year)` for any
year > 2024 and normalizes two column name differences before applying the same
PPR scoring pipeline:

| nflreadpy column | nfl_data_py equivalent |
|---|---|
| `passing_interceptions` | `interceptions` |
| `team` | `recent_team` |

### Historical range

**2022–2025** (4 seasons) for parameter estimation; production parameters are
written to `data/player_params.parquet` after running `exploration.ipynb`.
2026 preseason projections can override μ̂ while keeping historical σ̂².

---

## File Structure

```
nfl-simulator/
│
├── CLAUDE.md               ← you are here
│
├── data.py                 ← pull + clean data, compute PPR scores, cache per-year parquet
├── fit.py                  ← estimate (α, β, π) per player via method of moments
├── simulate.py             ← MC engine: vectorized zero-inflated Gamma draws
├── matchup.py              ← two rosters → P(win), SE, score distributions
│
├── data/
│   ├── weekly_stats_{year}.parquet  ← per-year cache (2022–2025)
│   ├── player_params.parquet        ← fitted (α, β, π, μ, σ) per player
│   └── schedules_{year}.parquet     ← schedule cache
│
└── notebooks/
    └── exploration.ipynb   ← parameter validation: Gamma fit, holdout calibration,
                               simulation sanity check, matchup demo
```

---

## Module Contracts

### `data.py`

```python
pull_weekly_stats(
    years: list[int],
    season_type: str = "REG",       # filter; "" = all game types
    force_refresh: bool = False,
) -> pd.DataFrame
# One row per player-week. Columns: season, week, season_type, player_id,
# player_name, position, recent_team, passing_yards, passing_tds,
# interceptions, rushing_yards, rushing_tds, receiving_yards, receiving_tds,
# receptions, rushing_fumbles_lost, receiving_fumbles_lost, sack_fumbles_lost,
# passing_2pt_conversions, rushing_2pt_conversions, receiving_2pt_conversions,
# fantasy_points_ppr, ppr_score.
# Dispatches to nflreadpy for year > 2024; normalizes to same schema.

pull_seasonal_rosters(year: int, force_refresh: bool = False) -> pd.DataFrame
pull_schedule(year: int, game_type: str = "REG", force_refresh: bool = False) -> pd.DataFrame
```

### `fit.py`

```python
fit_player_params(
    weekly_df: pd.DataFrame,
    min_nonzero: int = 4,           # fallback threshold
    cache: bool = True,             # writes data/player_params.parquet
) -> pd.DataFrame
# One row per player. Columns: player_id, player_name, position,
# alpha, beta, pi, mu, sigma, n_games, n_nonzero, fallback_used.
# Fitting on non-zero weeks only. Players with n_nonzero < 4 or σ = 0
# fall back to positional-median parameters.
# π excludes bye weeks: n_possible = 17 × n_seasons.
```

### `simulate.py`

```python
simulate_team_score(
    roster: list[dict],             # each dict needs "player_id"; bye players must be pre-filtered
    player_params: pd.DataFrame,
    n: int = 10_000,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray                     # shape (n,) — one total PPR score per trial
```

### `matchup.py`

```python
simulate_matchup(
    roster_a: list[dict],           # each dict needs "player_id" and "team" (if bye_teams used)
    roster_b: list[dict],
    player_params: pd.DataFrame,
    n: int = 10_000,
    bye_teams: list[str] | None = None,   # NFL team abbreviations on bye this week
    rng: int | np.random.Generator | None = None,
) -> dict
# Returns:
# {
#     "p_win_a": float,             # P(Team A score > Team B score)
#     "se": float,                  # sqrt(p̂(1-p̂)/n)
#     "mean_score_a": float,
#     "mean_score_b": float,
#     "score_dist_a": np.ndarray,   # shape (n,)
#     "score_dist_b": np.ndarray,
# }
```

---

## Key Design Decisions

- **Gamma not Gaussian**: Weekly scores are non-negative and right-skewed.
  Gaussian would assign positive probability to negative scores.
- **Zero-inflation handled explicitly**: DNP weeks are modeled as a Bernoulli
  draw on π rather than being included in the Gamma fit, which would bias α and β downward.
- **π excludes bye weeks**: π is estimated as `(n_possible − n_nonzero) / n_possible`
  where `n_possible = 17 × n_seasons`. `matchup.py` filters bye players before
  simulation via the `bye_teams` parameter; passing a bye player into
  `simulate_team_score` would give them a non-zero Gamma draw, inflating their score.
- **Method of moments over MLE**: Closed-form, fast, and sufficient given
  typical per-player sample sizes (16–51 games across 3 seasons). MLE would give
  marginally better estimates but adds complexity.
- **Per-year parquet cache**: Each season is cached separately so adding a new
  year only fetches the delta rather than re-fetching everything.
- **N = 10,000 default**: Gives SE < 0.005 for all p̂ ∈ (0.1, 0.9). Increase
  to 50,000 for tighter tail estimates (e.g. playoff bracket odds).
- **PCG64 RNG**: Uses `np.random.default_rng()` (not legacy `np.random.*`).
  Thread-safe and reproducible via integer seed.

---

## Not In Scope (Yet)

- Player correlations / covariance structure (Phase 2 — Gaussian copula)
- Injury simulation / week-to-week availability modeling
- Trade evaluation or waiver wire optimizer
- Multi-week season simulation (playoff bracket odds)
- UI — all output is DataFrames / dicts, visualize in notebook

---

## Teaching Style

The user is a **data scientist** — fluent in Python and statistics, but new to
building MC simulators. When writing or explaining code:

- **Explain the math behind every implementation decision**, not just what the
  code does. E.g. don't just say "we fit a Gamma here" — explain why that
  parameterization, what the method of moments is actually doing, and what
  could go wrong.
- **Call out non-obvious statistical choices** inline as comments: why Gamma
  and not Log-Normal, why we exclude zeros before fitting, what happens to the
  estimate if sample size is small, etc.
- **Flag assumptions explicitly** — every model has them. Surface them so the
  user can decide whether they hold.
- **Prefer depth over brevity** when explaining. The user wants to understand
  the simulator well enough to extend it themselves (e.g. adding correlations
  in Phase 2).

---

## Style Notes for Claude Code

- All functions should have **numpy-style docstrings** with Parameters, Returns,
  and a brief Notes section explaining the math.
- Use **type hints** throughout.
- Prefer **pandas** for data wrangling, **numpy** for the simulation loop
  (avoid Python loops over players — vectorize where possible).
- Cache all data pulls to `data/` as per-year `.parquet` files.
- No Jupyter magic in `.py` files — keep notebooks separate.
