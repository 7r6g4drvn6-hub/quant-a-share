from __future__ import annotations

import pandas as pd

from quant_app.backtest import run_backtest
from quant_app.strategy import build_panel


def test_run_backtest_handles_duplicate_daily_rows() -> None:
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    bars = pd.DataFrame(
        {
            "date": dates.tolist() + [dates[-1]],
            "open": [10.0 + idx * 0.1 for idx in range(80)] + [18.0],
            "close": [10.1 + idx * 0.1 for idx in range(80)] + [18.2],
            "high": [10.2 + idx * 0.1 for idx in range(80)] + [18.4],
            "low": [9.9 + idx * 0.1 for idx in range(80)] + [17.8],
            "volume": [1000 + idx for idx in range(80)] + [2000],
            "amount": [200000000.0 for _ in range(80)] + [250000000.0],
        }
    )
    stock_list = pd.DataFrame(
        {
            "code": ["600000", "600000"],
            "name": ["测试股份", "测试股份"],
            "list_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "is_st": [False, False],
        }
    )

    panel = build_panel({"600000": bars}, stock_list, ma_short=5, ma_long=20, momentum_window=5)

    assert not panel.duplicated(subset=["code", "date"]).any()
    equity, holdings, metrics = run_backtest(
        panel,
        top_n=1,
        min_avg_amount=1.0,
        min_listed_days=1,
        cost_bps=0.0,
    )

    assert not equity.empty
    assert not holdings.empty
    assert metrics["final_equity"] > 0
