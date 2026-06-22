from __future__ import annotations

from itertools import product
from math import sqrt

import pandas as pd

from quant_app.backtest import run_backtest
from quant_app.strategy import build_panel, latest_indicator_snapshot


FACTOR_LABELS = {
    "momentum": "动量",
    "trend_strength": "趋势强度",
    "liquidity_score": "流动性",
    "low_vol_score": "低波动",
    "value_score": "估值",
    "factor_score": "综合因子",
}


def market_segment(code: str | int) -> str:
    text = str(code).zfill(6)
    if text.startswith(("600", "601", "603", "605")):
        return "上证主板"
    if text.startswith(("000", "001")):
        return "深证主板"
    if text.startswith("002"):
        return "中小板"
    if text.startswith(("300", "301")):
        return "创业板"
    if text.startswith(("688", "689")):
        return "科创板"
    if text.startswith(("8", "4", "92")):
        return "北交所"
    if text.startswith(("5", "15", "16", "18")):
        return "基金/ETF"
    return "其他"


def preferred_group_column(stock_list: pd.DataFrame) -> str:
    for column in ["industry", "申万行业", "industry_name", "板块"]:
        if column in stock_list and stock_list[column].notna().any():
            return column
    return "risk_group"


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    return numeric.rank(pct=True, ascending=ascending)


def build_factor_snapshot(panel: pd.DataFrame, stock_list: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    history = panel.sort_values(["code", "date"]).copy()
    grouped = history.groupby("code", group_keys=False)
    history["return_5"] = grouped["close"].pct_change(5)
    history["volatility_20"] = grouped["daily_return"].transform(lambda value: value.rolling(20).std()) * sqrt(252)
    history["volatility_60"] = grouped["daily_return"].transform(lambda value: value.rolling(60).std()) * sqrt(252)
    history["max_close_60"] = grouped["close"].transform(lambda value: value.rolling(60).max())
    history["drawdown_60"] = history["close"] / history["max_close_60"] - 1

    latest = latest_indicator_snapshot(history)
    if latest.empty:
        return latest

    info_columns = [
        "code",
        "name",
        "pe",
        "pb",
        "turnover",
        "float_market_cap",
        "total_market_cap",
        "is_st",
        "list_date",
    ]
    extra_columns = [column for column in ["industry", "申万行业", "industry_name", "板块"] if column in stock_list]
    info = stock_list[[column for column in info_columns + extra_columns if column in stock_list]].copy()
    info["code"] = info["code"].astype(str).str.zfill(6)
    latest = latest.drop(columns=[column for column in info.columns if column in latest.columns and column != "code"], errors="ignore")
    latest = latest.merge(info, on="code", how="left")

    latest["risk_group"] = latest["code"].map(market_segment)
    group_column = preferred_group_column(latest)
    if group_column != "risk_group":
        latest["risk_group"] = latest[group_column].fillna(latest["risk_group"])

    latest["trend_strength"] = latest["close"] / latest["ma_long"] - 1
    latest["liquidity_score"] = _rank_pct(latest["avg_amount_20"], ascending=True)
    latest["low_vol_score"] = _rank_pct(latest["volatility_20"], ascending=False)
    latest["value_score"] = _rank_pct(latest["pb"].where(pd.to_numeric(latest.get("pb"), errors="coerce") > 0), ascending=False) if "pb" in latest else pd.NA
    latest["momentum_score"] = _rank_pct(latest["momentum"], ascending=True)
    latest["trend_score"] = _rank_pct(latest["trend_strength"], ascending=True)
    latest["factor_score"] = (
        latest["momentum_score"].fillna(0.5) * 0.35
        + latest["trend_score"].fillna(0.5) * 0.25
        + latest["liquidity_score"].fillna(0.5) * 0.20
        + latest["low_vol_score"].fillna(0.5) * 0.10
        + latest["value_score"].fillna(0.5) * 0.10
    )
    return latest.reset_index(drop=True)


def add_group_neutral_rank(candidates: pd.DataFrame, factor_snapshot: pd.DataFrame, max_per_group: int, top_n: int) -> pd.DataFrame:
    if candidates.empty or factor_snapshot.empty:
        return pd.DataFrame()
    groups = factor_snapshot[["code", "risk_group", "factor_score"]].copy()
    groups["code"] = groups["code"].astype(str).str.zfill(6)
    result = candidates.copy()
    result["code"] = result["code"].astype(str).str.zfill(6)
    result = result.merge(groups, on="code", how="left", suffixes=("", "_factor"))
    sort_columns = [column for column in ["score", "signal_strength", "momentum"] if column in result]
    if "factor_score" in result:
        sort_columns.append("factor_score")
    if not sort_columns:
        sort_columns = ["rank"] if "rank" in result else ["code"]
    result = result.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
    result["group_rank"] = result.groupby("risk_group").cumcount() + 1
    result = result[result["group_rank"] <= max(1, int(max_per_group))].head(top_n).copy()
    result["neutral_rank"] = range(1, len(result) + 1)
    return result


def _position_weights(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(columns=["code", "weight"])
    frame = positions.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "target_position_pct" in frame:
        frame["weight"] = pd.to_numeric(frame["target_position_pct"], errors="coerce").fillna(0.0)
    elif "weight" in frame:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    elif "target_value" in frame:
        value_sum = pd.to_numeric(frame["target_value"], errors="coerce").fillna(0.0).sum()
        frame["weight"] = pd.to_numeric(frame["target_value"], errors="coerce").fillna(0.0) / value_sum if value_sum else 0.0
    else:
        frame["weight"] = 1 / len(frame)
    return frame[["code", "weight"]]


def group_exposure(positions: pd.DataFrame, factor_snapshot: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or factor_snapshot.empty:
        return pd.DataFrame()
    weights = _position_weights(positions)
    universe = factor_snapshot[["code", "risk_group"]].copy()
    universe["code"] = universe["code"].astype(str).str.zfill(6)
    merged = weights.merge(universe, on="code", how="left")
    merged["risk_group"] = merged["risk_group"].fillna("未分类")

    exposure = merged.groupby("risk_group", as_index=False).agg(position_weight=("weight", "sum"), position_count=("code", "count"))
    universe_weight = universe.groupby("risk_group", as_index=False).agg(universe_count=("code", "count"))
    total_universe = max(len(universe), 1)
    universe_weight["universe_weight"] = universe_weight["universe_count"] / total_universe
    result = exposure.merge(universe_weight, on="risk_group", how="outer").fillna(0)
    result["active_weight"] = result["position_weight"] - result["universe_weight"]
    return result.sort_values("position_weight", ascending=False).reset_index(drop=True)


def factor_exposure(positions: pd.DataFrame, factor_snapshot: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or factor_snapshot.empty:
        return pd.DataFrame()
    weights = _position_weights(positions)
    factors = factor_snapshot.copy()
    factors["code"] = factors["code"].astype(str).str.zfill(6)
    merged = weights.merge(factors, on="code", how="left")
    total_weight = merged["weight"].sum()
    if total_weight <= 0:
        return pd.DataFrame()

    rows = []
    for column, label in FACTOR_LABELS.items():
        if column not in factors:
            continue
        portfolio_value = (pd.to_numeric(merged[column], errors="coerce") * merged["weight"]).sum() / total_weight
        universe_value = pd.to_numeric(factors[column], errors="coerce").mean()
        rows.append(
            {
                "factor": label,
                "portfolio_value": portfolio_value,
                "universe_value": universe_value,
                "active_value": portfolio_value - universe_value,
                "interpretation": "高于股票池" if portfolio_value > universe_value else "低于股票池",
            }
        )
    return pd.DataFrame(rows)


def run_parameter_scan(
    bars_by_code: dict[str, pd.DataFrame],
    stock_list: pd.DataFrame,
    ma_short_values: list[int],
    ma_long_values: list[int],
    momentum_windows: list[int],
    top_n_values: list[int],
    min_avg_amount: float,
    min_listed_days: int,
    cost_bps: float,
) -> pd.DataFrame:
    if not bars_by_code:
        return pd.DataFrame()

    rows = []
    for ma_short, ma_long, momentum_window, top_n in product(
        ma_short_values,
        ma_long_values,
        momentum_windows,
        top_n_values,
    ):
        if ma_short >= ma_long:
            continue
        panel = build_panel(
            bars_by_code,
            stock_list,
            ma_short=ma_short,
            ma_long=ma_long,
            momentum_window=momentum_window,
        )
        equity, _, metrics = run_backtest(
            panel,
            top_n=top_n,
            min_avg_amount=min_avg_amount,
            min_listed_days=min_listed_days,
            cost_bps=cost_bps,
        )
        if not metrics:
            continue
        rows.append(
            {
                "ma_short_param": ma_short,
                "ma_long_param": ma_long,
                "momentum_window_param": momentum_window,
                "top_n_param": top_n,
                "trading_days": len(equity),
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["sharpe", "annual_return"], ascending=False, na_position="last").reset_index(drop=True)
