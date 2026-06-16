"""
data.py — pull and cache nfl_data_py tables; compute PPR fantasy scores.

Quick start
-----------
    from data import pull_weekly_stats, pull_seasonal_rosters, pull_schedule

    stats    = pull_weekly_stats([2022, 2023, 2024])
    rosters  = pull_seasonal_rosters(2024)
    schedule = pull_schedule(2026)

Caching
-------
Each function writes one parquet file per year under data/ on first call and
reads from cache on subsequent calls.  Per-year files (weekly_stats_2022.parquet,
weekly_stats_2023.parquet, …) are preferred over a monolithic weekly_stats.parquet
because adding a new year only fetches the delta rather than re-fetching
everything.
"""

from pathlib import Path

import pandas as pd
import nfl_data_py as nfl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Scoring system
# ---------------------------------------------------------------------------

# Full PPR weights.  Stored as a module-level constant so downstream code can
# import and inspect the scoring rules without re-parsing anything.
PPR_SCORING: dict[str, float] = {
    "passing_yards":      0.04,
    "passing_tds":        4.0,
    "interceptions":     -2.0,   # QB turnovers, not defensive INT
    "rushing_yards":      0.1,
    "rushing_tds":        6.0,
    "receiving_yards":    0.1,
    "receiving_tds":      6.0,
    "receptions":         1.0,   # full PPR
    "fumbles_lost":      -2.0,
    "two_pt_conversions": 2.0,
}

# ---------------------------------------------------------------------------
# Column manifest
# ---------------------------------------------------------------------------

# nfl_data_py splits fumbles_lost and two_pt_conversions by action type.
# We aggregate them to scalar columns before applying PPR_SCORING weights.
# See _compute_ppr_score() for details.
_FUMBLE_LOST_COLS = [
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "sack_fumbles_lost",
]
_TWO_PT_COLS = [
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
]

# Columns retained in the cached parquet files.  Keeping the raw stat fields
# (not just ppr_score) lets downstream code recompute scoring under different
# rules without re-fetching.  fit.py needs position for the fallback-to-
# positional-median logic; simulate.py needs player_id and recent_team.
_KEEP_COLS: list[str] = [
    "season", "week", "season_type",
    "player_id", "player_name", "position", "recent_team",
    # raw stats that feed the PPR formula
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "receptions",
    *_FUMBLE_LOST_COLS,
    *_TWO_PT_COLS,
    # library's column — kept only as a validation reference
    "fantasy_points_ppr",
    # our computed score
    "ppr_score",
]

# ---------------------------------------------------------------------------
# PPR score computation
# ---------------------------------------------------------------------------


def _compute_ppr_score(df: pd.DataFrame) -> pd.Series:
    """Compute Full PPR score row-wise from raw stat columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weekly stats DataFrame from nfl_data_py, with NaN for missing
        stat fields.

    Returns
    -------
    pd.Series
        PPR fantasy score for each row, floored implicitly at 0 by the
        scoring math (negative totals are theoretically possible but rare;
        we don't clip them here because fit.py should handle zero-score weeks
        explicitly via the π parameter).

    Notes
    -----
    Why we recompute instead of using ``fantasy_points_ppr``
    --------------------------------------------------------
    The library's column is correct for most players, but recomputing from
    scratch with an explicit weight dict makes the math auditable and lets
    us swap the scoring system (e.g. half-PPR, superflex) in one place.

    Why we fill NaN with 0
    ----------------------
    nfl_data_py emits NaN for stat categories a player never touched (e.g.
    rushing_yards for a pure pocket QB).  NaN * weight = NaN, which would
    contaminate the row sum.  Zero is the correct semantic value here: the
    player truly had 0 rushing yards, not a missing observation.

    Why fumbles_lost and two_pt_conversions are split
    -------------------------------------------------
    The raw data stores these per-action-type (rushing, receiving, sack /
    passing, rushing, receiving) rather than as a single column.  We sum the
    sub-columns first so the final weight application is clean.
    """
    fumbles_lost = sum(df[c].fillna(0) for c in _FUMBLE_LOST_COLS)
    two_pt = sum(df[c].fillna(0) for c in _TWO_PT_COLS)

    return (
        df["passing_yards"].fillna(0)   * PPR_SCORING["passing_yards"]
        + df["passing_tds"].fillna(0)   * PPR_SCORING["passing_tds"]
        + df["interceptions"].fillna(0) * PPR_SCORING["interceptions"]
        + df["rushing_yards"].fillna(0) * PPR_SCORING["rushing_yards"]
        + df["rushing_tds"].fillna(0)   * PPR_SCORING["rushing_tds"]
        + df["receiving_yards"].fillna(0) * PPR_SCORING["receiving_yards"]
        + df["receiving_tds"].fillna(0) * PPR_SCORING["receiving_tds"]
        + df["receptions"].fillna(0)    * PPR_SCORING["receptions"]
        + fumbles_lost                  * PPR_SCORING["fumbles_lost"]
        + two_pt                        * PPR_SCORING["two_pt_conversions"]
    )


def _validate_ppr_score(df: pd.DataFrame, tolerance: float = 2.0, max_bad_pct: float = 0.01) -> None:
    """Assert our ppr_score agrees with the library's fantasy_points_ppr.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain both ``ppr_score`` and ``fantasy_points_ppr``.
    tolerance : float
        Maximum allowed absolute difference per row before a row is flagged.
        Defaults to 2.0 because the library includes special_teams_tds (6 pts)
        which we intentionally omit.  A single ST TD creates a 6-pt gap, so
        any gap > 2 on a non-kicker row is a signal of a column-mapping error.
    max_bad_pct : float
        Fraction of rows allowed to exceed tolerance before raising.

    Notes
    -----
    The library's ``fantasy_points_ppr`` includes special_teams_tds (kick/punt
    return touchdowns) scored at 6 pts each.  We omit this stat because ST
    return TDs are too rare to model per-player and most fantasy leagues track
    them differently.  Expect ~0.5–1% of rows to show a 6-pt gap for the rare
    returner.  The 1% cap on bad rows catches systematic column mismatches
    while tolerating the known ST discrepancy.
    """
    gap = (df["ppr_score"] - df["fantasy_points_ppr"].fillna(0)).abs()
    bad = gap[gap > tolerance]
    bad_pct = len(bad) / len(df)
    if bad_pct > max_bad_pct:
        raise ValueError(
            f"ppr_score diverges from fantasy_points_ppr for {len(bad)} rows "
            f"({100 * bad_pct:.1f}% > {100 * max_bad_pct:.1f}% threshold). "
            "Check column mapping in _compute_ppr_score().\n"
            f"Worst offenders:\n{df.loc[bad.nlargest(5).index, ['player_name', 'week', 'ppr_score', 'fantasy_points_ppr']]}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pull_weekly_stats(
    years: list[int],
    season_type: str = "REG",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull per-player weekly stats for the given seasons.

    Fetches from nfl_data_py on first call; reads from
    ``data/weekly_stats_{year}.parquet`` on subsequent calls.

    Parameters
    ----------
    years : list[int]
        NFL seasons to include (e.g. ``[2022, 2023, 2024]``).
    season_type : str, optional
        Filter to this value of the ``season_type`` column.  ``"REG"``
        (default) keeps only regular-season weeks.

        **Assumption — why REG only by default**: Postseason rows are
        produced by fewer teams (only playoff participants), at a different
        point in the season (smaller sample per player), and against
        systematically stronger opponents than a random regular-season week.
        Including them would inject selection bias into the Gamma fit: players
        who make the playoffs are typically better than average, so their
        postseason scores would pull μ̂ upward.  Pass ``season_type=""`` to
        include postseason rows if you want to model playoff matchups
        specifically.
    force_refresh : bool, optional
        Re-fetch from nfl_data_py even if a cached file exists.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame, one row per player-week, with columns::

            season, week, season_type,
            player_id, player_name, position, recent_team,
            passing_yards, passing_tds, interceptions,
            rushing_yards, rushing_tds,
            receiving_yards, receiving_tds, receptions,
            rushing_fumbles_lost, receiving_fumbles_lost, sack_fumbles_lost,
            passing_2pt_conversions, rushing_2pt_conversions, receiving_2pt_conversions,
            fantasy_points_ppr,   # library's value, for reference
            ppr_score             # our computed value

    Notes
    -----
    Missing weeks (bye, injury, DNP, IR) produce **no row** in this output —
    nfl_data_py only emits rows for weeks where a player appeared in play-by-
    play data.  fit.py accounts for this when estimating π (the zero-inflation
    probability) by comparing each player's n_active_weeks against the total
    number of possible weeks in each season.  Do not pre-pad missing weeks
    with zeros here.

    All players (all positions) are returned.  fit.py handles position-based
    filtering and fallbacks.
    """
    frames: list[pd.DataFrame] = []
    for year in years:
        cache_path = DATA_DIR / f"weekly_stats_{year}.parquet"
        if not force_refresh and cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            raw = nfl.import_weekly_data([year])
            raw["ppr_score"] = _compute_ppr_score(raw)
            _validate_ppr_score(raw)
            df = raw[[c for c in _KEEP_COLS if c in raw.columns]].copy()
            df.to_parquet(cache_path, index=False)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if season_type:
        combined = combined[combined["season_type"] == season_type].copy()

    return combined.reset_index(drop=True)


def pull_seasonal_rosters(
    year: int,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull and cache the end-of-season roster for a given year.

    Parameters
    ----------
    year : int
        NFL season year.
    force_refresh : bool, optional
        Bypass cache if True.

    Returns
    -------
    pd.DataFrame
        One row per player.  Key columns: ``player_id``, ``player_name``,
        ``position``, ``team``, ``season``.

    Notes
    -----
    Uses ``import_seasonal_rosters`` (not the non-existent ``import_rosters``).
    The seasonal roster represents the final end-of-season snapshot; use
    ``pull_weekly_rosters`` if you need week-level roster moves.
    """
    cache_path = DATA_DIR / f"rosters_{year}.parquet"
    if not force_refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = nfl.import_seasonal_rosters([year])
    df.to_parquet(cache_path, index=False)
    return df


def pull_schedule(
    year: int,
    game_type: str = "REG",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull and cache the schedule for a given NFL season.

    Parameters
    ----------
    year : int
        NFL season year.
    game_type : str, optional
        Filter to ``"REG"`` (default), ``"WC"``, ``"DIV"``, ``"CON"``,
        ``"SB"``, or ``""`` (all game types).
    force_refresh : bool, optional
        Bypass cache if True.

    Returns
    -------
    pd.DataFrame
        One row per game.  Key columns: ``game_id``, ``season``, ``week``,
        ``game_type``, ``away_team``, ``home_team``, ``gameday``.
        Also includes betting lines, weather, and QB fields from nfl_data_py.
    """
    cache_path = DATA_DIR / f"schedules_{year}.parquet"
    if not force_refresh and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = nfl.import_schedules([year])
        df.to_parquet(cache_path, index=False)

    if game_type:
        df = df[df["game_type"] == game_type].copy()

    return df.reset_index(drop=True)
