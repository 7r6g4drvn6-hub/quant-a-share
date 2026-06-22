from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "config"
UI_DEFAULTS_PATH = CONFIG_DIR / "ui_defaults.json"

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_app.backtest import normalized_benchmark, run_backtest
from quant_app.data import (
    INDEX_MAP,
    fetch_index_bars,
    fetch_realtime_index_quotes,
    fetch_realtime_quotes,
    load_cached_bars_for_codes,
    load_stock_list,
    normalize_code,
    update_daily_cache,
)
from quant_app.strategy import (
    build_panel,
    intraday_candidates,
    latest_candidates,
    market_index_predictions,
    next_day_predictions,
    optimize_trade_plan,
    operation_recommendations,
)
from quant_app.risk import (
    add_group_neutral_rank,
    board_market_judgment,
    build_factor_snapshot,
    factor_exposure,
    group_exposure,
    run_parameter_scan,
)


st.set_page_config(page_title="A股量化研究台", page_icon="Q", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1480px;}
      h1, h2, h3 {letter-spacing: 0;}
      [data-testid="stMetric"] {
        border: 1px solid #e6e8ef;
        border-radius: 8px;
        padding: 10px 12px;
        background: #ffffff;
      }
      .muted {color: #667085; font-size: 0.88rem;}
      .risk-note {
        border-left: 4px solid #d92d20;
        padding: 8px 12px;
        background: #fff5f4;
        color: #344054;
        margin-top: 12px;
        border-radius: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def number_yi(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value / 100000000:.1f}亿"


def money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    if abs(value) < 0.5:
        value = 0.0
    if abs(value) >= 100000000:
        return f"{value / 100000000:.2f}亿"
    if abs(value) >= 10000:
        return f"{value / 10000:.1f}万"
    return f"{value:,.0f}"


def load_ui_defaults() -> dict:
    if not UI_DEFAULTS_PATH.exists():
        return {}
    try:
        with UI_DEFAULTS_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_ui_defaults(defaults: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    with UI_DEFAULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(defaults, file, ensure_ascii=False, indent=2)


def clamp_int(value, lower: int, upper: int, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(lower, min(upper, number))


def parse_date_default(value, fallback: date) -> date:
    if not value:
        return fallback
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return fallback
    return parsed.date()


COLUMN_LABELS = {
    "rank": "排名",
    "index_name": "指数",
    "code": "代码",
    "name": "名称",
    "symbol": "标识",
    "date": "日期",
    "open": "开盘价",
    "close": "收盘价",
    "high": "最高价",
    "low": "最低价",
    "latest": "最新价",
    "prev_close": "昨收价",
    "pct_change": "涨跌幅%",
    "pct_change_rt": "盘中涨幅%",
    "volume": "成交量",
    "amount": "成交额",
    "amount_yi": "成交额(亿)",
    "amount_rt": "盘中成交额",
    "turnover": "换手率%",
    "pe": "市盈率",
    "pb": "市净率",
    "list_date": "上市日期",
    "weight": "权重",
    "momentum": "日线动量",
    "avg_amount_20": "20日均成交额",
    "ma_short": "短均线",
    "ma_long": "长均线",
    "signal_strength": "信号强度",
    "action": "操作建议",
    "action_score": "操作强度",
    "recommendation": "交易动作",
    "trade_shares": "买卖股数",
    "trade_value": "买卖金额",
    "current_shares": "当前股数",
    "current_value": "当前市值",
    "target_value": "目标市值",
    "target_position_pct": "目标仓位",
    "expected_profit_amount": "期望收益额",
    "max_risk_amount": "最大风险额",
    "expected_value": "期望收益",
    "reward_risk": "收益风险比",
    "risk_pct": "止损风险",
    "gain_pct": "止盈空间",
    "stop_loss": "止损参考",
    "take_profit": "止盈参考",
    "suggested_position": "策略仓位",
    "amount_ratio": "放量倍数",
    "amount_ratio_rt": "放量倍数",
    "distance_to_ma_short": "偏离短均线",
    "score": "评分",
    "neutral_rank": "中性排名",
    "risk_group": "风险分组",
    "board_name": "板块",
    "market_status": "板块状态",
    "board_score": "板块评分",
    "quote_count": "样本数",
    "up_count": "上涨家数",
    "down_count": "下跌家数",
    "flat_count": "平盘家数",
    "up_ratio": "上涨占比",
    "avg_pct_change": "平均涨跌幅%",
    "median_pct_change": "中位涨跌幅%",
    "weighted_pct_change": "成交额加权涨跌幅%",
    "amount_share": "成交额占比",
    "strong_count": "强势股数",
    "limit_up_count": "涨停数",
    "limit_down_count": "跌停数",
    "leader_code": "领涨代码",
    "leader_name": "领涨股",
    "leader_pct_change": "领涨涨幅%",
    "laggard_code": "领跌代码",
    "laggard_name": "领跌股",
    "laggard_pct_change": "领跌跌幅%",
    "group_rank": "组内排名",
    "factor_score": "综合因子",
    "trend_strength": "趋势强度",
    "liquidity_score": "流动性因子",
    "low_vol_score": "低波动因子",
    "value_score": "估值因子",
    "momentum_score": "动量因子",
    "volatility_20": "20日波动率",
    "volatility_60": "60日波动率",
    "drawdown_60": "60日回撤",
    "position_weight": "持仓权重",
    "position_count": "持仓数量",
    "universe_weight": "股票池权重",
    "universe_count": "股票池数量",
    "active_weight": "主动偏离",
    "factor": "因子",
    "portfolio_value": "组合值",
    "universe_value": "股票池均值",
    "active_value": "主动暴露",
    "interpretation": "解释",
    "ma_short_param": "短均线",
    "ma_long_param": "长均线",
    "momentum_window_param": "动量窗口",
    "top_n_param": "持仓数量",
    "final_equity": "最终净值",
    "total_return": "累计收益",
    "annual_return": "年化收益",
    "annual_volatility": "年化波动",
    "sharpe": "夏普",
    "max_drawdown": "最大回撤",
    "win_rate": "胜率",
    "avg_turnover": "平均换手",
    "trading_status": "交易状态",
    "t1_locked": "T+1锁定",
    "sellable_shares": "可卖股数",
    "limit_up": "涨停价",
    "limit_down": "跌停价",
    "limit_pct": "涨跌停幅度",
    "can_buy": "可买",
    "can_sell": "可卖",
    "quote_time": "行情时间",
    "quote_datetime": "行情日期时间",
    "direction": "明日方向",
    "probability_up": "上涨概率",
    "confidence": "置信度",
    "expected_next_return": "历史期望",
    "base_probability": "基础概率",
    "sample_count": "样本数",
    "sample_source": "样本来源",
    "reason": "原因",
    "error": "错误",
    "amplitude": "振幅%",
    "price_change": "涨跌额",
    "daily_return": "日收益",
    "trading_days": "交易天数",
    "is_st": "ST标记",
}


def table_view(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    if columns:
        existing_columns = [column for column in columns if column in df.columns]
        view = df[existing_columns].copy()
    else:
        view = df.copy()
    return view.rename(columns={column: COLUMN_LABELS.get(column, column) for column in view.columns})


def parse_holdings_text(text: str) -> pd.DataFrame:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part for part in re.split(r"[,\s，]+", line) if part]
        if len(parts) < 2 or not re.search(r"\d{6}", parts[0]):
            continue
        try:
            code = normalize_code(parts[0])
            shares = float(parts[1])
            cost_price = float(parts[2]) if len(parts) >= 3 else pd.NA
            buy_date = pd.to_datetime(parts[3], errors="coerce") if len(parts) >= 4 else pd.NaT
        except ValueError:
            continue
        if shares <= 0:
            continue
        rows.append({"code": code, "shares": shares, "cost_price": cost_price, "buy_date": buy_date})
    return pd.DataFrame(rows, columns=["code", "shares", "cost_price", "buy_date"])


@st.cache_data(ttl=300)
def cached_stock_list(refresh: bool = False, min_rows: int = 1000) -> pd.DataFrame:
    return load_stock_list(refresh=refresh, min_rows=min_rows)


@st.cache_data(ttl=5)
def cached_realtime_quotes(codes: tuple[str, ...]) -> pd.DataFrame:
    return fetch_realtime_quotes(codes)


@st.cache_data(ttl=5)
def cached_realtime_index_quotes() -> pd.DataFrame:
    return fetch_realtime_index_quotes()


@st.cache_data(ttl=1800)
def cached_index_history(start, end) -> dict[str, pd.DataFrame]:
    return {
        index_name: fetch_index_bars(index_name, start, end, refresh=False)
        for index_name in INDEX_MAP
    }


def liquid_universe(stock_list: pd.DataFrame, count: int) -> pd.DataFrame:
    df = stock_list.copy()
    df = df[(~df["is_st"]) & df["latest"].notna() & df["amount"].notna()]
    df = df[~df["name"].str.contains("退", na=False)]
    return df.sort_values("amount", ascending=False).head(count).reset_index(drop=True)


def realtime_state(
    selected_codes: list[str],
    panel: pd.DataFrame,
    min_avg_amount: float,
    min_listed_days: int,
    min_realtime_amount: float,
    min_intraday_pct: float,
    max_intraday_pct: float,
    top_n: int,
    prediction_min_samples: int,
) -> tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        realtime_quotes = cached_realtime_quotes(tuple(selected_codes))
    except Exception as exc:  # noqa: BLE001 - keep the rest of the app usable.
        realtime_quotes = pd.DataFrame()
        realtime_error = str(exc)
    else:
        realtime_error = ""

    realtime_picks = intraday_candidates(
        panel,
        realtime_quotes,
        min_avg_amount=min_avg_amount,
        min_listed_days=min_listed_days,
        min_realtime_amount=min_realtime_amount,
        min_intraday_pct=min_intraday_pct,
        max_intraday_pct=max_intraday_pct,
        top_n=top_n,
    )
    predictions = next_day_predictions(
        panel,
        realtime_quotes,
        top_n=top_n,
        min_samples=prediction_min_samples,
    )
    operations = operation_recommendations(panel, realtime_quotes, predictions, top_n=top_n)
    return realtime_quotes, realtime_error, realtime_picks, predictions, operations


def make_equity_chart(equity: pd.DataFrame, benchmark: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["equity"],
            name="策略净值",
            line=dict(color="#1570ef", width=2.4),
        )
    )
    if not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark["date"],
                y=benchmark["benchmark"],
                name="沪深300",
                line=dict(color="#12b76a", width=1.8),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=8, r=8, t=24, b=8),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="归一化净值",
        hovermode="x unified",
    )
    return fig


def make_price_chart(bars: pd.DataFrame, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bars["date"], y=bars["close"], name="收盘价", line=dict(color="#101828")))
    if "ma_short" in bars:
        fig.add_trace(go.Scatter(x=bars["date"], y=bars["ma_short"], name="短均线", line=dict(color="#1570ef")))
    if "ma_long" in bars:
        fig.add_trace(go.Scatter(x=bars["date"], y=bars["ma_long"], name="长均线", line=dict(color="#f79009")))
    fig.update_layout(
        title=name,
        height=420,
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig


st.title("A股量化研究台")

ui_defaults = load_ui_defaults()
sample_defaults = ui_defaults.get("sample", {}) if isinstance(ui_defaults.get("sample", {}), dict) else {}
default_end_date = parse_date_default(sample_defaults.get("end_date"), date.today())
default_start_date = parse_date_default(
    sample_defaults.get("start_date"),
    default_end_date - timedelta(days=730),
)
if default_start_date > default_end_date:
    default_start_date = default_end_date - timedelta(days=730)
max_stock_count = 5000
stored_stock_count = sample_defaults.get("stock_count")
if stored_stock_count is not None:
    try:
        stored_stock_count = int(stored_stock_count)
    except (TypeError, ValueError):
        stored_stock_count = None
if stored_stock_count is not None and stored_stock_count < 1000:
    stored_stock_count = 1000
default_stock_count = clamp_int(stored_stock_count, 20, max_stock_count, 1000)
default_stock_count = int(round(default_stock_count / 50) * 50)
default_stock_count = clamp_int(default_stock_count, 20, max_stock_count, 1000)
default_auto_fetch_missing = bool(sample_defaults.get("auto_fetch_missing", False))

with st.sidebar:
    st.subheader("样本")
    end_date = st.date_input("结束日期", value=default_end_date, key="sample_end_date")
    start_date = st.date_input("开始日期", value=default_start_date, key="sample_start_date")
    stock_count = st.slider(
        "股票数量",
        min_value=20,
        max_value=max_stock_count,
        value=default_stock_count,
        step=50,
        key="sample_stock_count",
    )
    auto_fetch_missing = st.toggle(
        "缺失数据自动补齐",
        value=default_auto_fetch_missing,
        key="sample_auto_fetch_missing",
    )

    st.subheader("策略")
    ma_short = st.slider("短均线", min_value=5, max_value=60, value=20, step=1)
    ma_long = st.slider("长均线", min_value=30, max_value=180, value=60, step=5)
    momentum_window = st.slider("动量窗口", min_value=5, max_value=120, value=20, step=5)
    top_n = st.slider("候选/持仓数量", min_value=3, max_value=100, value=20, step=1)
    min_avg_amount_yi = st.slider("20日均额下限", min_value=0.2, max_value=10.0, value=1.0, step=0.1)
    min_listed_days = st.slider("上市天数下限", min_value=30, max_value=500, value=120, step=10)
    cost_bps = st.slider("单边成本 bps", min_value=0.0, max_value=50.0, value=8.0, step=1.0)

    st.subheader("风控")
    max_per_group = st.slider("单组最多入选", min_value=1, max_value=10, value=3, step=1)
    scan_grid_size = st.select_slider("参数扫描范围", options=["关闭", "小", "中"], value="小")

    st.subheader("实时")
    realtime_auto_refresh = st.toggle("自动刷新", value=True)
    refresh_seconds = st.slider("刷新秒数", min_value=10, max_value=120, value=30, step=10)
    min_realtime_amount_yi = st.slider("盘中成交额下限", min_value=0.1, max_value=20.0, value=0.5, step=0.1)
    min_intraday_pct = st.slider("盘中涨幅下限", min_value=-5.0, max_value=10.0, value=0.0, step=0.5)
    max_intraday_pct = st.slider("盘中涨幅上限", min_value=0.0, max_value=20.0, value=9.8, step=0.5)
    refresh_realtime = st.button("刷新实时行情", width="stretch")

    st.subheader("预测")
    prediction_min_samples = st.slider("最少历史样本", min_value=10, max_value=120, value=40, step=10)

    refresh_list = st.button("刷新股票列表", width="stretch")
    update_prices = st.button("更新样本日线", type="primary", width="stretch")

current_sample_defaults = {
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat(),
    "stock_count": int(stock_count),
    "auto_fetch_missing": bool(auto_fetch_missing),
}
if sample_defaults != current_sample_defaults:
    ui_defaults["sample"] = current_sample_defaults
    save_ui_defaults(ui_defaults)

if ma_short >= ma_long:
    st.warning("短均线需要小于长均线。")
    st.stop()

try:
    stock_list = cached_stock_list(refresh=refresh_list, min_rows=min(stock_count, 1000))
except Exception as exc:  # noqa: BLE001 - show data-source failures in app.
    st.error(f"股票列表加载失败：{exc}")
    st.stop()

universe = liquid_universe(stock_list, stock_count)
selected_codes = universe["code"].tolist()
min_avg_amount = min_avg_amount_yi * 100000000
min_realtime_amount = min_realtime_amount_yi * 100000000

if update_prices:
    progress = st.progress(0)
    status = st.empty()

    def progress_cb(idx: int, total: int, code: str, state: str) -> None:
        progress.progress(idx / max(total, 1))
        status.write(f"{idx}/{total} {code} {state}")

    with st.spinner("更新行情中"):
        _, update_errors = update_daily_cache(selected_codes, start_date, end_date, progress=progress_cb)
    if update_errors:
        st.warning(f"{len(update_errors)} 只股票更新失败，已跳过。")
        st.dataframe(table_view(pd.DataFrame(update_errors, columns=["code", "error"])), width="stretch", hide_index=True)
    else:
        st.success("样本日线已更新。")

bars_by_code, missing_codes = load_cached_bars_for_codes(selected_codes, start_date, end_date)
auto_fetch_limit = 200
if missing_codes and auto_fetch_missing and len(missing_codes) <= auto_fetch_limit:
    with st.spinner(f"补齐 {len(missing_codes)} 只股票的日线"):
        fetched, fetch_errors = update_daily_cache(missing_codes, start_date, end_date)
    bars_by_code.update(fetched)
    if fetch_errors:
        st.warning(f"{len(fetch_errors)} 只股票无法获取。")
elif missing_codes and auto_fetch_missing:
    st.info(
        f"有 {len(missing_codes)} 只股票缺少日线缓存，数量较大，已跳过自动补齐以避免页面卡住。"
        "实时行情仍会覆盖全部样本；回测、日线选股和个股预测需要点击“更新样本日线”补齐。"
    )
elif missing_codes:
    st.info(f"有 {len(missing_codes)} 只股票缺少日线缓存。实时行情仍会覆盖全部样本；回测、日线选股和个股预测需要点击“更新样本日线”补齐。")

panel = build_panel(
    bars_by_code,
    stock_list,
    ma_short=ma_short,
    ma_long=ma_long,
    momentum_window=momentum_window,
)
factor_snapshot = build_factor_snapshot(panel, stock_list)

equity, holdings, metrics = run_backtest(
    panel,
    top_n=top_n,
    min_avg_amount=min_avg_amount,
    min_listed_days=min_listed_days,
    cost_bps=cost_bps,
)

if refresh_realtime:
    cached_realtime_quotes.clear()

market_tab, realtime_tab, operation_tab, prediction_tab, data_tab, backtest_tab, pick_tab, risk_tab, scan_tab, detail_tab = st.tabs(
    ["市场", "实时", "操作", "预测", "数据", "回测", "选股", "风控", "扫描", "个股"]
)
realtime_run_every = refresh_seconds if realtime_auto_refresh else None

with market_tab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本/股票池", f"{len(selected_codes)}/{len(stock_list)}")
    col2.metric("缓存可用", f"{len(bars_by_code)}")
    col3.metric("回测交易日", f"{len(equity)}")
    col4.metric("最新列表", str(stock_list["updated_at"].iloc[0]) if "updated_at" in stock_list else "-")

    @st.fragment(run_every=realtime_run_every)
    def market_index_panel() -> None:
        try:
            index_quotes = cached_realtime_index_quotes()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"实时指数加载失败：{exc}")
            return

        if index_quotes.empty:
            st.warning("没有获取到实时指数行情。")
            return

        latest_quote = index_quotes["quote_datetime"].dropna()
        quote_time = latest_quote.max().strftime("%Y-%m-%d %H:%M:%S") if not latest_quote.empty else "-"
        st.caption(f"指数实时行情时间：{quote_time}")
        st.dataframe(
            table_view(
                index_quotes,
                [
                    "index_name",
                    "latest",
                    "price_change",
                    "pct_change",
                    "open",
                    "high",
                    "low",
                    "amount_yi",
                    "quote_time",
                ],
            ),
            width="stretch",
            hide_index=True,
        )
        index_history = cached_index_history(start_date, end_date)
        market_predictions = market_index_predictions(
            index_history,
            index_quotes,
            ma_short=ma_short,
            ma_long=ma_long,
            momentum_window=momentum_window,
            min_samples=max(40, prediction_min_samples),
        )
        st.subheader("行情预测")
        if market_predictions.empty:
            st.warning("指数历史样本不足，暂时无法生成行情预测。")
        else:
            st.dataframe(
                table_view(
                    market_predictions,
                    [
                        "index_name",
                        "direction",
                        "probability_up",
                        "confidence",
                        "expected_next_return",
                        "sample_count",
                        "sample_source",
                        "latest",
                        "pct_change",
                        "amount_ratio",
                        "reason",
                        "quote_time",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption("行情预测是基于指数历史相似状态的概率信号，不是确定性判断。")

        st.subheader("板块行情判断")
        board_sample_size = min(len(selected_codes), 600)
        board_codes = selected_codes[:board_sample_size]
        try:
            board_quotes = cached_realtime_quotes(tuple(board_codes))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"板块实时行情加载失败：{exc}")
            return

        board_summary, board_view = board_market_judgment(board_quotes, stock_list)
        if not board_summary or board_view.empty:
            st.warning("当前样本无法生成板块行情判断。")
        else:
            board_cols = st.columns(5)
            board_cols[0].metric("行情判断", str(board_summary["market_view"]))
            board_cols[1].metric("上涨占比", pct(board_summary["overall_up_ratio"]))
            board_cols[2].metric("加权涨跌", f"{board_summary['overall_weighted_pct_change']:.2f}%")
            board_cols[3].metric("强势板块", f"{board_summary['strong_board_count']}/{board_summary['board_count']}")
            board_cols[4].metric("领涨板块", f"{board_summary['leading_board']}")
            st.dataframe(
                table_view(
                    board_view.head(20),
                    [
                        "rank",
                        "board_name",
                        "market_status",
                        "board_score",
                        "quote_count",
                        "up_ratio",
                        "weighted_pct_change",
                        "amount_yi",
                        "amount_share",
                        "strong_count",
                        "limit_up_count",
                        "limit_down_count",
                        "leader_code",
                        "leader_name",
                        "leader_pct_change",
                        "laggard_name",
                        "laggard_pct_change",
                        "quote_time",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"板块判断基于当前样本中成交额靠前 {board_sample_size} 只股票，"
                "按成交额加权涨跌幅、上涨家数占比和成交额占比综合排序；"
                "没有 Tushare 行业字段时，先按主板/创业板/科创板等市场分组。"
            )

    market_index_panel()

with realtime_tab:
    @st.fragment(run_every=realtime_run_every)
    def realtime_panel() -> None:
        realtime_quotes, realtime_error, realtime_picks, _, _ = realtime_state(
            selected_codes,
            panel,
            min_avg_amount,
            min_listed_days,
            min_realtime_amount,
            min_intraday_pct,
            max_intraday_pct,
            top_n,
            prediction_min_samples,
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("实时样本", f"{len(realtime_quotes)}")
        metric_cols[1].metric("候选股", f"{len(realtime_picks)}")
        metric_cols[2].metric("刷新间隔", f"{refresh_seconds}s" if realtime_auto_refresh else "手动")
        latest_quote_time = "-"
        if not realtime_quotes.empty and "quote_datetime" in realtime_quotes:
            latest_quote = realtime_quotes["quote_datetime"].dropna()
            if not latest_quote.empty:
                latest_quote_time = latest_quote.max().strftime("%Y-%m-%d %H:%M:%S")
        metric_cols[3].metric("行情时间", latest_quote_time)

        if realtime_error:
            st.warning(f"实时行情加载失败：{realtime_error}")
        elif realtime_picks.empty:
            st.warning("当前参数下没有盘中候选股票。")
        else:
            st.dataframe(
                table_view(
                    realtime_picks,
                    [
                        "rank",
                        "code",
                        "name",
                        "latest",
                        "pct_change_rt",
                        "amount_rt",
                        "amount_ratio",
                        "momentum",
                        "distance_to_ma_short",
                        "score",
                        "quote_time",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )

        if not realtime_quotes.empty:
            st.dataframe(
                table_view(
                    realtime_quotes,
                    ["code", "name", "latest", "pct_change", "amount", "high", "low", "quote_time"],
                ),
                width="stretch",
                hide_index=True,
            )

    realtime_panel()

with operation_tab:
    plan_col1, plan_col2, plan_col3, plan_col4 = st.columns(4)
    total_assets = plan_col1.number_input("账户市值", min_value=0.0, value=100000.0, step=10000.0, format="%.0f")
    cash_available = plan_col2.number_input("可用现金", min_value=0.0, value=100000.0, step=10000.0, format="%.0f")
    reserve_cash_pct = plan_col3.slider("保留现金", min_value=0, max_value=60, value=20, step=5) / 100
    max_position_pct = plan_col4.slider("单票上限", min_value=5, max_value=40, value=15, step=1) / 100

    plan_col5, plan_col6 = st.columns([1, 3])
    max_buy_candidates = plan_col5.slider("最多买入", min_value=1, max_value=20, value=min(5, top_n), step=1)
    holdings_text = plan_col6.text_area(
        "现有持仓",
        placeholder="600000,1000,8.50,2026-06-22\n000001,500,10.20,2026-06-19",
        height=90,
    )
    st.caption("持仓格式：代码,股数,成本价,买入日期。买入日期为今天的持仓会按 A 股 T+1 规则锁定为不可卖。")
    holdings_df = parse_holdings_text(holdings_text)

    @st.fragment(run_every=realtime_run_every)
    def operation_panel() -> None:
        operation_scope = max(top_n, max_buy_candidates * 4, 60)
        _, realtime_error, _, _, operations = realtime_state(
            selected_codes,
            panel,
            min_avg_amount,
            min_listed_days,
            min_realtime_amount,
            min_intraday_pct,
            max_intraday_pct,
            operation_scope,
            prediction_min_samples,
        )
        metric_cols = st.columns(4)
        if operations.empty:
            buy_count = sell_count = hold_count = 0
            best_ev = None
        else:
            buy_count = int((operations["action"] == "买入候选").sum())
            sell_count = int((operations["action"] == "卖出/回避").sum())
            hold_count = int((operations["action"] == "持有观察").sum())
            best_ev = operations["expected_value"].max()
        metric_cols[0].metric("买入候选", f"{buy_count}")
        metric_cols[1].metric("卖出/回避", f"{sell_count}")
        metric_cols[2].metric("持有观察", f"{hold_count}")
        metric_cols[3].metric("最高期望收益", pct(best_ev))

        if realtime_error:
            st.warning(f"实时行情加载失败：{realtime_error}")
        elif operations.empty:
            st.warning("当前样本缺少日线缓存或预测数据，暂时无法生成操作建议。")
        else:
            trade_plan = optimize_trade_plan(
                operations,
                holdings_df,
                total_assets=total_assets,
                cash_available=cash_available,
                reserve_cash_pct=reserve_cash_pct,
                max_position_pct=max_position_pct,
                max_buy_candidates=max_buy_candidates,
                trade_date=end_date,
            )
            actionable = trade_plan[trade_plan["trade_shares"] != 0].copy() if not trade_plan.empty else pd.DataFrame()
            planned_buy_value = actionable[actionable["trade_value"] > 0]["trade_value"].sum() if not actionable.empty else 0.0
            planned_sell_value = -actionable[actionable["trade_value"] < 0]["trade_value"].sum() if not actionable.empty else 0.0
            plan_expected_profit = trade_plan["expected_profit_amount"].sum(skipna=True) if not trade_plan.empty else 0.0
            plan_risk_amount = trade_plan["max_risk_amount"].sum(skipna=True) if not trade_plan.empty else 0.0

            st.subheader("最优买卖计划")
            plan_metrics = st.columns(4)
            plan_metrics[0].metric("计划买入", money(planned_buy_value))
            plan_metrics[1].metric("计划卖出", money(planned_sell_value))
            plan_metrics[2].metric("组合期望", money(plan_expected_profit))
            plan_metrics[3].metric("组合风险", money(plan_risk_amount))

            if trade_plan.empty:
                st.warning("当前约束下没有生成可执行买卖计划。")
            else:
                st.dataframe(
                    table_view(
                        trade_plan.head(max(top_n, max_buy_candidates)),
                        [
                            "rank",
                            "code",
                            "name",
                            "recommendation",
                            "trade_shares",
                            "trade_value",
                            "current_shares",
                            "sellable_shares",
                            "trading_status",
                            "t1_locked",
                            "target_position_pct",
                            "expected_profit_amount",
                            "max_risk_amount",
                            "probability_up",
                            "risk_pct",
                            "latest",
                            "limit_up",
                            "limit_down",
                            "stop_loss",
                            "take_profit",
                            "reason",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.subheader("操作信号")
            st.dataframe(
                table_view(
                    operations.head(top_n),
                    [
                        "rank",
                        "code",
                        "name",
                        "action",
                        "action_score",
                        "expected_value",
                        "reward_risk",
                        "probability_up",
                        "confidence",
                        "latest",
                        "trading_status",
                        "limit_up",
                        "limit_down",
                        "stop_loss",
                        "take_profit",
                        "suggested_position",
                        "pct_change_rt",
                        "amount_ratio_rt",
                        "reason",
                        "quote_time",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption("操作建议按策略信号强度排序，仅用于交易辅助；仓位、止损、止盈是策略参考，不是确定收益。")

    operation_panel()

with prediction_tab:
    @st.fragment(run_every=realtime_run_every)
    def prediction_panel() -> None:
        _, _, _, predictions, _ = realtime_state(
            selected_codes,
            panel,
            min_avg_amount,
            min_listed_days,
            min_realtime_amount,
            min_intraday_pct,
            max_intraday_pct,
            top_n,
            prediction_min_samples,
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("预测股票", f"{len(predictions)}")
        if predictions.empty:
            up_count = 0
            down_count = 0
            avg_probability = None
        else:
            up_count = int((predictions["direction"] == "偏涨").sum())
            down_count = int((predictions["direction"] == "偏跌").sum())
            avg_probability = predictions["probability_up"].mean()
        metric_cols[1].metric("偏涨", f"{up_count}")
        metric_cols[2].metric("偏跌", f"{down_count}")
        metric_cols[3].metric("平均上涨概率", pct(avg_probability))

        if predictions.empty:
            st.warning("当前样本不足，暂时无法生成明日预测。")
        else:
            st.dataframe(
                table_view(
                    predictions,
                    [
                        "rank",
                        "code",
                        "name",
                        "direction",
                        "probability_up",
                        "confidence",
                        "expected_next_return",
                        "sample_count",
                        "sample_source",
                        "latest",
                        "pct_change_rt",
                        "amount_ratio_rt",
                        "reason",
                        "quote_time",
                    ],
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption("预测为历史统计信号，不是确定性结论；上涨概率接近 50% 时应视为中性。")

    prediction_panel()

with data_tab:
    st.dataframe(
        table_view(
            universe,
            [
                "code",
                "name",
                "latest",
                "pct_change",
                "amount",
                "turnover",
                "pe",
                "pb",
                "list_date",
            ],
        ),
        width="stretch",
        hide_index=True,
    )

with backtest_tab:
    if equity.empty:
        st.warning("暂无可回测数据。")
    else:
        metric_cols = st.columns(6)
        metric_cols[0].metric("累计收益", pct(metrics.get("total_return")))
        metric_cols[1].metric("年化收益", pct(metrics.get("annual_return")))
        metric_cols[2].metric("最大回撤", pct(metrics.get("max_drawdown")))
        metric_cols[3].metric("夏普", "-" if pd.isna(metrics.get("sharpe")) else f"{metrics.get('sharpe'):.2f}")
        metric_cols[4].metric("胜率", pct(metrics.get("win_rate")))
        metric_cols[5].metric("平均换手", pct(metrics.get("avg_turnover")))

        try:
            hs300 = fetch_index_bars("沪深300", start_date, end_date)
        except Exception:
            hs300 = pd.DataFrame()
        benchmark = normalized_benchmark(hs300, equity["date"])
        st.plotly_chart(make_equity_chart(equity, benchmark), use_container_width=True)

        latest_holdings = holdings[holdings["date"] == holdings["date"].max()] if not holdings.empty else pd.DataFrame()
        st.dataframe(
            table_view(latest_holdings, ["rank", "code", "name", "weight", "close", "momentum", "avg_amount_20"])
            if not latest_holdings.empty
            else latest_holdings,
            width="stretch",
            hide_index=True,
        )

with pick_tab:
    intraday_pick_tab, daily_pick_tab = st.tabs(["盘中实时选股", "日线选股"])

    with intraday_pick_tab:
        @st.fragment(run_every=realtime_run_every)
        def intraday_pick_panel() -> None:
            realtime_quotes, realtime_error, realtime_picks, _, _ = realtime_state(
                selected_codes,
                panel,
                min_avg_amount,
                min_listed_days,
                min_realtime_amount,
                min_intraday_pct,
                max_intraday_pct,
                top_n,
                prediction_min_samples,
            )
            metric_cols = st.columns(4)
            metric_cols[0].metric("实时样本", f"{len(realtime_quotes)}")
            metric_cols[1].metric("入选股票", f"{len(realtime_picks)}")
            metric_cols[2].metric("刷新方式", f"{refresh_seconds}s" if realtime_auto_refresh else "手动")
            latest_quote_time = "-"
            if not realtime_quotes.empty and "quote_datetime" in realtime_quotes:
                latest_quote = realtime_quotes["quote_datetime"].dropna()
                if not latest_quote.empty:
                    latest_quote_time = latest_quote.max().strftime("%Y-%m-%d %H:%M:%S")
            metric_cols[3].metric("行情时间", latest_quote_time)

            if realtime_error:
                st.warning(f"实时行情加载失败：{realtime_error}")
            elif realtime_picks.empty:
                st.warning("当前参数下没有盘中实时候选股票。可以放宽盘中成交额、涨幅或均线条件。")
            else:
                neutral_picks = add_group_neutral_rank(realtime_picks, factor_snapshot, max_per_group, top_n)
                show_picks = neutral_picks if not neutral_picks.empty else realtime_picks
                st.dataframe(
                    table_view(
                        show_picks,
                        [
                            "neutral_rank" if "neutral_rank" in show_picks else "rank",
                            "code",
                            "name",
                            "risk_group",
                            "latest",
                            "pct_change_rt",
                            "amount_rt",
                            "amount_ratio",
                            "momentum",
                            "factor_score",
                            "distance_to_ma_short",
                            "score",
                            "quote_time",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption("盘中实时选股会随实时行情缓存刷新；右侧“刷新实时行情”会立即清空缓存并重新拉取。")

        intraday_pick_panel()

    with daily_pick_tab:
        trade_date, picks = latest_candidates(panel, min_avg_amount, min_listed_days, top_n)
        if trade_date is None or picks.empty:
            st.warning("当前参数下没有日线候选股票。")
        else:
            st.caption(f"日线交易日：{trade_date.date()}。日线选股基于本地日线缓存，只有点击“更新样本日线”后才会变。")
            neutral_picks = add_group_neutral_rank(picks, factor_snapshot, max_per_group, top_n)
            show_picks = neutral_picks if not neutral_picks.empty else picks
            display = table_view(
                show_picks,
                [
                    "neutral_rank" if "neutral_rank" in show_picks else "rank",
                    "code",
                    "name",
                    "risk_group",
                    "close",
                    "momentum",
                    "factor_score",
                    "avg_amount_20",
                    "ma_short",
                    "ma_long",
                    "signal_strength",
                ],
            )
            st.dataframe(display, width="stretch", hide_index=True)

with risk_tab:
    if factor_snapshot.empty:
        st.warning("暂无风控数据。请先更新样本日线。")
    else:
        st.subheader("股票池因子")
        factor_cols = st.columns(4)
        factor_cols[0].metric("覆盖股票", f"{len(factor_snapshot)}")
        factor_cols[1].metric("风险分组", f"{factor_snapshot['risk_group'].nunique() if 'risk_group' in factor_snapshot else 0}")
        factor_cols[2].metric("平均20日波动", pct(factor_snapshot["volatility_20"].mean() if "volatility_20" in factor_snapshot else None))
        factor_cols[3].metric("平均60日回撤", pct(factor_snapshot["drawdown_60"].mean() if "drawdown_60" in factor_snapshot else None))

        factor_view = factor_snapshot.sort_values("factor_score", ascending=False).head(top_n).copy()
        st.dataframe(
            table_view(
                factor_view,
                [
                    "code",
                    "name",
                    "risk_group",
                    "close",
                    "momentum",
                    "trend_strength",
                    "avg_amount_20",
                    "volatility_20",
                    "drawdown_60",
                    "pb",
                    "factor_score",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

        latest_positions = holdings[holdings["date"] == holdings["date"].max()] if not holdings.empty else pd.DataFrame()
        if latest_positions.empty:
            st.info("暂无回测最新持仓，无法计算组合暴露。更新样本日线并跑出回测后会显示。")
        else:
            st.subheader("组合分组暴露")
            group_view = group_exposure(latest_positions, factor_snapshot)
            st.dataframe(
                table_view(
                    group_view,
                    ["risk_group", "position_weight", "position_count", "universe_weight", "universe_count", "active_weight"],
                ),
                width="stretch",
                hide_index=True,
            )

            st.subheader("组合因子暴露")
            exposure_view = factor_exposure(latest_positions, factor_snapshot)
            st.dataframe(
                table_view(exposure_view, ["factor", "portfolio_value", "universe_value", "active_value", "interpretation"]),
                width="stretch",
                hide_index=True,
            )

        st.caption("当前如果没有行业字段，会先按主板/创业板/科创板等市场分组；后续接入 Tushare 后可替换为申万行业。")

with scan_tab:
    if scan_grid_size == "关闭":
        st.info("参数扫描已关闭。可在左侧“风控”里改为小或中。")
    elif not bars_by_code:
        st.warning("暂无日线缓存，无法做参数扫描。")
    else:
        if scan_grid_size == "小":
            short_values = sorted({max(5, ma_short - 5), ma_short, min(60, ma_short + 5)})
            long_values = sorted({max(30, ma_long - 20), ma_long, min(180, ma_long + 20)})
            momentum_values = sorted({max(5, momentum_window - 10), momentum_window, min(120, momentum_window + 10)})
            top_values = sorted({max(3, top_n - 5), top_n})
        else:
            short_values = sorted({10, 15, 20, 30, ma_short})
            long_values = sorted({50, 60, 90, 120, ma_long})
            momentum_values = sorted({10, 20, 40, 60, momentum_window})
            top_values = sorted({10, 20, 30, top_n})

        st.caption(
            f"扫描组合数约 {len(short_values) * len(long_values) * len(momentum_values) * len(top_values)}，"
            "仅用于检查参数稳定性，不用于过度拟合。"
        )
        if st.button("运行参数扫描", type="primary", width="stretch"):
            with st.spinner("正在扫描参数"):
                scan_result = run_parameter_scan(
                    bars_by_code,
                    stock_list,
                    ma_short_values=short_values,
                    ma_long_values=long_values,
                    momentum_windows=momentum_values,
                    top_n_values=top_values,
                    min_avg_amount=min_avg_amount,
                    min_listed_days=min_listed_days,
                    cost_bps=cost_bps,
                )
            if scan_result.empty:
                st.warning("没有生成可用扫描结果。")
            else:
                best = scan_result.iloc[0]
                metric_cols = st.columns(4)
                metric_cols[0].metric("最佳夏普", "-" if pd.isna(best["sharpe"]) else f"{best['sharpe']:.2f}")
                metric_cols[1].metric("最佳年化", pct(best["annual_return"]))
                metric_cols[2].metric("最大回撤", pct(best["max_drawdown"]))
                metric_cols[3].metric("组合数", f"{len(scan_result)}")
                st.dataframe(
                    table_view(
                        scan_result.head(30),
                        [
                            "ma_short_param",
                            "ma_long_param",
                            "momentum_window_param",
                            "top_n_param",
                            "trading_days",
                            "total_return",
                            "annual_return",
                            "annual_volatility",
                            "sharpe",
                            "max_drawdown",
                            "win_rate",
                            "avg_turnover",
                        ],
                    ),
                    width="stretch",
                    hide_index=True,
                )

with detail_tab:
    if panel.empty:
        st.warning("暂无个股数据。")
    else:
        options = universe["code"] + " " + universe["name"]
        selected = st.selectbox("股票", options.tolist())
        code = selected.split(" ")[0]
        stock_name = selected.split(" ", 1)[1] if " " in selected else code
        bars = panel[panel["code"] == code].copy()
        if bars.empty:
            st.warning("该股票没有缓存日线。")
        else:
            st.plotly_chart(make_price_chart(bars, f"{code} {stock_name}"), use_container_width=True)
            st.dataframe(table_view(bars.tail(30).sort_values("date", ascending=False)), width="stretch", hide_index=True)

st.markdown(
    '<div class="risk-note">本软件仅用于量化研究和回测验证，不构成投资建议，不连接券商交易。</div>',
    unsafe_allow_html=True,
)
