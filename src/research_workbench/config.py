from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path





@dataclass(frozen=True)
class RefreshSchedule:
    """Centralised refresh-frequency configuration (in hours)."""
    history_interval_hours: int = 24
    summary_interval_hours: int = 168  # weekly
    signals_interval_hours: int = 24
    events_interval_hours: int = 168  # weekly


@dataclass(frozen=True)
class AppSettings:
    repo_root: Path
    data_dir: Path
    docs_dir: Path
    input_dir: Path
    summary_path: Path
    history_path: Path
    radar_path: Path
    watchlist_path: Path
    manual_events_path: Path
    signals_snapshot_path: Path = field(default=Path(""))

    refresh: RefreshSchedule = field(default_factory=RefreshSchedule)


def default_settings(repo_root: Path) -> AppSettings:
    data_dir = repo_root / "outputs" / "research_data"
    input_dir = repo_root / "research_inputs"
    return AppSettings(
        repo_root=repo_root,
        data_dir=data_dir,
        docs_dir=repo_root / "docs",
        input_dir=input_dir,
        summary_path=data_dir / "three_month_summary.csv",
        history_path=data_dir / "three_month_close_history.csv",
        radar_path=data_dir / "equity_opportunity_radar.csv",
        watchlist_path=input_dir / "watchlist_cn.json",
        manual_events_path=input_dir / "manual_events.csv",
        signals_snapshot_path=data_dir / "signals_snapshot.csv",

    )
