from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_app.strategy import candidates_for_date


def _metrics(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    final_equity = float(equity["equity"].iloc[-1])
    total_return = final_equity - 1
    days = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days, 1)
    years = days / 365.25
    annual_return = final_equity ** (1 / years) - 1 if final_equity > 0 else -1
    daily_returns = equity["daily_return"].fillna(0)
    annual_volatility = daily_returns.std(ddof=0) * math.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility else np.nan
    drawdown = equity["equity"] / equity["equity"].cummax() - 1
    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((daily_returns > 0).mean()),
        "avg_turnover": float(equity["turnover"].mean()),
    }


def run_backtest(
    panel: pd.DataFrame,
    top_n: int,
    min_avg_amount: float,
    min_listed_days: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    dates = sorted(panel["date"].dropna().unique())
    weights: dict[str, float] = {}
    equity_value = 1.0
    rows = []
    holdings_rows = []
    cost_rate = cost_bps / 10000

    for trade_date in dates:
        day = panel[panel["date"] == trade_date].set_index("code")
        gross_return = 0.0
        for code, weight in list(weights.items()):
            if code in day.index and pd.notna(day.loc[code, "daily_return"]):
                gross_return += weight * float(day.loc[code, "daily_return"])

        equity_value *= 1 + gross_return

        candidates = candidates_for_date(
            panel,
            trade_date,
            min_avg_amount=min_avg_amount,
            min_listed_days=min_listed_days,
            top_n=top_n,
        )
        target_codes = candidates["code"].tolist() if not candidates.empty else []
        target_weight = 1 / len(target_codes) if target_codes else 0
        target_weights = {code: target_weight for code in target_codes}

        turnover = sum(
            abs(target_weights.get(code, 0.0) - weights.get(code, 0.0))
            for code in set(weights) | set(target_weights)
        )
        cost = turnover * cost_rate
        equity_value *= max(0.0, 1 - cost)

        rows.append(
            {
                "date": pd.to_datetime(trade_date),
                "equity": equity_value,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "positions": len(target_codes),
            }
        )

        for _, row in candidates.iterrows():
            holdings_rows.append(
                {
                    "date": pd.to_datetime(trade_date),
                    "code": row["code"],
                    "name": row.get("name", ""),
                    "rank": row["rank"],
                    "weight": target_weight,
                    "close": row["close"],
                    "momentum": row["momentum"],
                    "avg_amount_20": row["avg_amount_20"],
                }
            )

        weights = target_weights

    equity = pd.DataFrame(rows)
    if equity.empty:
        return equity, pd.DataFrame(holdings_rows), {}
    equity["daily_return"] = equity["equity"].pct_change().fillna(equity["equity"] - 1)
    return equity, pd.DataFrame(holdings_rows), _metrics(equity)


def normalized_benchmark(index_bars: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    if index_bars.empty:
        return pd.DataFrame(columns=["date", "benchmark"])
    dates = pd.to_datetime(dates)
    df = index_bars[index_bars["date"].isin(dates)].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "benchmark"])
    df["benchmark"] = df["close"] / df["close"].iloc[0]
    return df[["date", "benchmark"]]

