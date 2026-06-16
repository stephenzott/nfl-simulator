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
π̂ = (zero-score weeks) / (total weeks)
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
    "passing_yards":    0.04,   # per yard
    "passing_tds":      4.0,
    "interceptions":   -2.0,
    "rushing_yards":    0.1,    # per yard
    "rushing_tds":      6.0,
    "receiving_yards":  0.1,    # per yard
    "receiving_tds":    6.0,
    "receptions":       1.0,    # full PPR
    "fumbles_lost":    -2.0,
    "two_pt_conversions": 2.0,
}
```

---

## Data Source

**nfl_data_py** (wraps nflfastR — no API key required)

```bash
pip install nfl_data_py pandas numpy scipy
```

Primary tables used:

| Table | nfl_data_py call | Purpose |
|---|---|---|
| Weekly player stats | `nfl.import_weekly_data([year])` | Fit α, β, π per player |
| Season rosters | `nfl.import_rosters([year])` | Map players to teams |
| Schedule | `nfl.import_schedules([year])` | Week-by-week matchups |

Historical range: **2022–2024** (3 seasons for stable parameter estimates).
2026 preseason projections can override μ̂ while keeping historical σ̂².

---

## File Structure

```
nfl_mc_simulator/
│
├── CLAUDE.md               ← you are here
│
├── data.py                 ← pull + clean nfl_data_py data, compute PPR scores
├── fit.py                  ← estimate (α, β, π) per player via method of moments
├── simulate.py             ← MC engine: draw from distributions, sum team scores
├── matchup.py              ← takes two rosters → returns P(win), SE, score dist
│
├── data/
│   ├── weekly_stats.parquet    ← cached raw pulls (avoid re-fetching)
│   ├── player_params.parquet   ← fitted (α, β, π, μ, σ) per player-season
│   └── schedules.parquet       ← 2026 schedule
│
└── notebooks/
    └── exploration.ipynb   ← scratch space for parameter sanity checks
```

---

## Module Contracts

### `data.py`
- `pull_weekly_stats(years: list[int]) -> pd.DataFrame`
  Returns tidy DataFrame with one row per player-week, columns include raw stat
  fields and a computed `ppr_score` column.
- `pull_rosters(year: int) -> pd.DataFrame`
- `pull_schedule(year: int) -> pd.DataFrame`

### `fit.py`
- `fit_player_params(weekly_df: pd.DataFrame) -> pd.DataFrame`
  Returns one row per player with columns: `player_id`, `player_name`, `alpha`,
  `beta`, `pi`, `mu`, `sigma`, `n_games`.
  Fitting is done on non-zero weeks only. Players with < 4 non-zero weeks get
  flagged and fall back to positional median parameters.

### `simulate.py`
- `simulate_team_score(roster: list[dict], player_params: pd.DataFrame, n: int) -> np.ndarray`
  Returns array of shape `(n,)` — one total team score per trial.
  Each player draw: `0` with prob `π`, else `Gamma(α, β)` sample.

### `matchup.py`
- `simulate_matchup(roster_a, roster_b, player_params, n=10_000) -> dict`
  Returns:
  ```python
  {
      "p_win_a": float,       # P(Team A wins)
      "se": float,            # standard error of estimate
      "mean_score_a": float,
      "mean_score_b": float,
      "score_dist_a": np.ndarray,   # shape (n,)
      "score_dist_b": np.ndarray,
  }
  ```

---

## Key Design Decisions

- **Gamma not Gaussian**: Weekly scores are non-negative and right-skewed.
  Gaussian would assign positive probability to negative scores.
- **Zero-inflation handled explicitly**: DNP weeks are modeled as a Bernoulli
  draw on π rather than being included in the Gamma fit, which would bias α and β downward.
- **Method of moments over MLE**: Closed-form, fast, and sufficient given
  typical per-player sample sizes (16–34 games). MLE would give marginally
  better estimates but adds complexity.
- **N = 10,000 default**: Gives SE < 0.005 for all p̂ ∈ (0.1, 0.9). Increase
  to 50,000 only if you need tighter tails (e.g. playoff odds).

---

## Not In Scope (Yet)

- Player correlations / covariance structure (Phase 2)
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
- Cache all nfl_data_py pulls to `data/` as `.parquet` to avoid redundant
  network calls.
- No Jupyter magic in `.py` files — keep notebooks separate.
