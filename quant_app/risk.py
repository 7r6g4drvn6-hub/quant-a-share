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


BOARD_STATUS_ORDER = {
    "强势领涨": 0,
    "板块活跃": 1,
    "资金分化": 2,
    "板块承压": 3,
    "普跌弱势": 4,
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


def stock_group_map(stock_list: pd.DataFrame) -> pd.DataFrame:
    if stock_list.empty:
        return pd.DataFrame(columns=["code", "risk_group"])

    frame = stock_list.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["risk_group"] = frame["code"].map(market_segment)
    group_column = preferred_group_column(frame)
    if group_column != "risk_group":
        frame["risk_group"] = frame[group_column].fillna(frame["risk_group"])
    return frame[["code", "risk_group"]].drop_duplicates("code")


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

    latest = latest.drop(columns=["risk_group"], errors="ignore")
    latest = latest.merge(stock_group_map(latest), on="code", how="left")
    latest["risk_group"] = latest["risk_group"].fillna(latest["code"].map(market_segment))

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


def _board_status(weighted_pct_change: float, up_ratio: float) -> str:
    if weighted_pct_change >= 1.5 and up_ratio >= 0.65:
        return "强势领涨"
    if weighted_pct_change >= 0.5 and up_ratio >= 0.55:
        return "板块活跃"
    if weighted_pct_change <= -1.5 and up_ratio <= 0.35:
        return "普跌弱势"
    if weighted_pct_change <= -0.5 and up_ratio <= 0.45:
        return "板块承压"
    return "资金分化"


def board_market_judgment(
    realtime_quotes: pd.DataFrame,
    stock_list: pd.DataFrame,
    min_members: int = 5,
) -> tuple[dict[str, object], pd.DataFrame]:
    if realtime_quotes.empty or stock_list.empty:
        return {}, pd.DataFrame()

    quote_cols = ["code", "name", "latest", "pct_change", "amount", "quote_time", "quote_datetime"]
    quotes = realtime_quotes[[column for column in quote_cols if column in realtime_quotes]].copy()
    quotes["code"] = quotes["code"].astype(str).str.zfill(6)
    quotes["pct_change"] = pd.to_numeric(quotes.get("pct_change"), errors="coerce")
    quotes["amount"] = pd.to_numeric(quotes.get("amount"), errors="coerce").fillna(0.0)
    quotes = quotes[quotes["pct_change"].notna()].copy()
    if quotes.empty:
        return {}, pd.DataFrame()

    grouped_quotes = quotes.merge(stock_group_map(stock_list), on="code", how="left")
    grouped_quotes["risk_group"] = grouped_quotes["risk_group"].fillna(grouped_quotes["code"].map(market_segment))
    total_amount = grouped_quotes["amount"].sum()
    total_amount = total_amount if total_amount > 0 else 1.0

    rows = []
    for board_name, group in grouped_quotes.groupby("risk_group", dropna=False):
        quote_count = len(group)
        if quote_count < min_members:
            continue

        amount = float(group["amount"].sum())
        avg_pct_change = float(group["pct_change"].mean())
        median_pct_change = float(group["pct_change"].median())
        weighted_pct_change = (
            float((group["pct_change"] * group["amount"]).sum() / amount)
            if amount > 0
            else avg_pct_change
        )
        up_count = int((group["pct_change"] > 0).sum())
        down_count = int((group["pct_change"] < 0).sum())
        flat_count = int((group["pct_change"] == 0).sum())
        up_ratio = up_count / quote_count if quote_count else 0.0
        strong_count = int((group["pct_change"] >= 5.0).sum())
        limit_up_count = int((group["pct_change"] >= 9.7).sum())
        limit_down_count = int((group["pct_change"] <= -9.7).sum())
        leader = group.sort_values(["pct_change", "amount"], ascending=False).iloc[0]
        laggard = group.sort_values(["pct_change", "amount"], ascending=[True, False]).iloc[0]
        board_score = weighted_pct_change * 0.45 + (up_ratio - 0.5) * 6.0 + min(amount / total_amount, 0.2) * 4.0

        rows.append(
            {
                "board_name": str(board_name),
                "market_status": _board_status(weighted_pct_change, up_ratio),
                "board_score": board_score,
                "quote_count": quote_count,
                "up_count": up_count,
                "down_count": down_count,
                "flat_count": flat_count,
                "up_ratio": up_ratio,
                "avg_pct_change": avg_pct_change,
                "median_pct_change": median_pct_change,
                "weighted_pct_change": weighted_pct_change,
                "amount": amount,
                "amount_yi": amount / 100000000,
                "amount_share": amount / total_amount,
                "strong_count": strong_count,
                "limit_up_count": limit_up_count,
                "limit_down_count": limit_down_count,
                "leader_code": leader.get("code", ""),
                "leader_name": leader.get("name", ""),
                "leader_pct_change": leader.get("pct_change", pd.NA),
                "laggard_code": laggard.get("code", ""),
                "laggard_name": laggard.get("name", ""),
                "laggard_pct_change": laggard.get("pct_change", pd.NA),
                "quote_time": group["quote_time"].dropna().max() if "quote_time" in group else pd.NA,
            }
        )

    if not rows:
        return {}, pd.DataFrame()

    boards = pd.DataFrame(rows)
    boards["status_order"] = boards["market_status"].map(BOARD_STATUS_ORDER).fillna(9)
    boards = boards.sort_values(
        ["status_order", "board_score", "weighted_pct_change", "amount"],
        ascending=[True, False, False, False],
    ).drop(columns=["status_order"]).reset_index(drop=True)
    boards["rank"] = range(1, len(boards) + 1)

    overall_amount = float(grouped_quotes["amount"].sum())
    overall_weighted = (
        float((grouped_quotes["pct_change"] * grouped_quotes["amount"]).sum() / overall_amount)
        if overall_amount > 0
        else float(grouped_quotes["pct_change"].mean())
    )
    overall_up_ratio = float((grouped_quotes["pct_change"] > 0).mean())
    strong_boards = int(boards["market_status"].isin(["强势领涨", "板块活跃"]).sum())
    weak_boards = int(boards["market_status"].isin(["板块承压", "普跌弱势"]).sum())

    if strong_boards >= max(2, len(boards) * 0.35) and overall_up_ratio >= 0.55:
        market_view = "板块扩散偏强"
    elif weak_boards >= max(2, len(boards) * 0.35) and overall_up_ratio <= 0.45:
        market_view = "板块退潮偏弱"
    elif strong_boards > 0 and weak_boards > 0:
        market_view = "结构分化"
    elif overall_weighted >= 0.3 and overall_up_ratio >= 0.5:
        market_view = "温和修复"
    elif overall_weighted <= -0.3 and overall_up_ratio <= 0.5:
        market_view = "震荡承压"
    else:
        market_view = "中性震荡"

    summary = {
        "market_view": market_view,
        "board_count": len(boards),
        "strong_board_count": strong_boards,
        "weak_board_count": weak_boards,
        "overall_up_ratio": overall_up_ratio,
        "overall_weighted_pct_change": overall_weighted,
        "total_amount_yi": overall_amount / 100000000,
        "leading_board": boards.iloc[0]["board_name"],
        "leading_status": boards.iloc[0]["market_status"],
        "leading_pct_change": boards.iloc[0]["weighted_pct_change"],
        "weakest_board": boards.sort_values("weighted_pct_change").iloc[0]["board_name"],
    }
    return summary, boards


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
