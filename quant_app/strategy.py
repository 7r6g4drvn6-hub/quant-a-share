from __future__ import annotations

import pandas as pd


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _allocate_with_caps(scores: pd.Series, budget: float, caps: pd.Series) -> pd.Series:
    allocations = pd.Series(0.0, index=scores.index)
    remaining = max(float(budget), 0.0)
    active = scores[scores > 0].copy()
    caps = caps.reindex(scores.index).fillna(0.0).clip(lower=0.0)

    for _ in range(len(active) + 1):
        if remaining <= 0 or active.empty:
            break
        score_sum = float(active.sum())
        if score_sum <= 0:
            break
        proposed = active / score_sum * remaining
        room = (caps - allocations).reindex(active.index).clip(lower=0.0)
        capped = proposed >= room
        if not capped.any():
            allocations.loc[active.index] += proposed
            break

        capped_index = capped[capped].index
        allocations.loc[capped_index] += room.loc[capped_index]
        remaining -= float(room.loc[capped_index].sum())
        active = active.drop(index=capped_index)

    return allocations


def add_indicators(
    bars: pd.DataFrame,
    ma_short: int = 20,
    ma_long: int = 60,
    momentum_window: int = 20,
) -> pd.DataFrame:
    df = bars.sort_values("date").copy()
    df["ma_short"] = df["close"].rolling(ma_short).mean()
    df["ma_long"] = df["close"].rolling(ma_long).mean()
    df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1
    df["avg_amount_20"] = df["amount"].rolling(20).mean()
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["trading_days"] = range(1, len(df) + 1)
    return df


def build_panel(
    bars_by_code: dict[str, pd.DataFrame],
    stock_list: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    momentum_window: int,
) -> pd.DataFrame:
    frames = []
    for code, bars in bars_by_code.items():
        if bars.empty:
            continue
        enriched = add_indicators(bars, ma_short, ma_long, momentum_window)
        enriched["code"] = code
        frames.append(enriched)
    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    info = stock_list[["code", "name", "list_date", "is_st"]].copy()
    info["code"] = info["code"].astype(str).str.zfill(6)
    panel = panel.merge(info, on="code", how="left")
    panel["is_st"] = panel["is_st"].fillna(False)
    return panel


def candidates_for_date(
    panel: pd.DataFrame,
    trade_date,
    min_avg_amount: float,
    min_listed_days: int,
    top_n: int,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    trade_date = pd.to_datetime(trade_date)
    day = panel[panel["date"] == trade_date].copy()
    if day.empty:
        return day

    if "list_date" in day:
        listed_days = (trade_date - day["list_date"]).dt.days
    else:
        listed_days = day["trading_days"]

    mask = (
        (~day["is_st"].fillna(False))
        & (day["trading_days"] >= min_listed_days)
        & (listed_days.fillna(min_listed_days) >= min_listed_days)
        & (day["avg_amount_20"] >= min_avg_amount)
        & (day["close"] > day["ma_short"])
        & (day["ma_short"] > day["ma_long"])
        & day["momentum"].notna()
    )
    result = day[mask].sort_values("momentum", ascending=False).head(top_n).copy()
    if result.empty:
        return result
    result["rank"] = range(1, len(result) + 1)
    result["signal_strength"] = (
        result["momentum"].rank(pct=True) * 0.7
        + (result["close"] / result["ma_short"] - 1).rank(pct=True) * 0.3
    )
    return result


def latest_candidates(
    panel: pd.DataFrame,
    min_avg_amount: float,
    min_listed_days: int,
    top_n: int,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    if panel.empty:
        return None, pd.DataFrame()
    trade_date = panel["date"].max()
    return trade_date, candidates_for_date(panel, trade_date, min_avg_amount, min_listed_days, top_n)


def latest_indicator_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    idx = panel.sort_values("date").groupby("code", as_index=False).tail(1).index
    return panel.loc[idx].copy()


def intraday_candidates(
    panel: pd.DataFrame,
    realtime_quotes: pd.DataFrame,
    min_avg_amount: float,
    min_listed_days: int,
    min_realtime_amount: float,
    min_intraday_pct: float,
    max_intraday_pct: float,
    top_n: int,
) -> pd.DataFrame:
    if panel.empty or realtime_quotes.empty:
        return pd.DataFrame()

    latest = latest_indicator_snapshot(panel)
    if latest.empty:
        return pd.DataFrame()

    quote_cols = [
        "code",
        "latest",
        "open",
        "prev_close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_change",
        "quote_date",
        "quote_time",
        "quote_datetime",
    ]
    quotes = realtime_quotes[[col for col in quote_cols if col in realtime_quotes]].copy()
    quotes["code"] = quotes["code"].astype(str).str.zfill(6)
    merged = latest.merge(quotes, on="code", how="inner", suffixes=("_daily", "_rt"))
    if merged.empty:
        return merged

    if "list_date" in merged:
        listed_days = (pd.Timestamp.today().normalize() - merged["list_date"]).dt.days
    else:
        listed_days = merged["trading_days"]

    merged["amount_ratio"] = merged["amount_rt"] / merged["avg_amount_20"]
    merged["distance_to_ma_short"] = merged["latest"] / merged["ma_short"] - 1
    merged["intraday_position"] = (merged["latest"] - merged["low_rt"]) / (merged["high_rt"] - merged["low_rt"])
    merged["intraday_position"] = merged["intraday_position"].replace([float("inf"), -float("inf")], pd.NA)

    mask = (
        (~merged["is_st"].fillna(False))
        & (merged["trading_days"] >= min_listed_days)
        & (listed_days.fillna(min_listed_days) >= min_listed_days)
        & (merged["avg_amount_20"] >= min_avg_amount)
        & (merged["amount_rt"] >= min_realtime_amount)
        & (merged["latest"] > merged["ma_short"])
        & (merged["ma_short"] > merged["ma_long"])
        & (merged["pct_change_rt"] >= min_intraday_pct)
        & (merged["pct_change_rt"] <= max_intraday_pct)
        & merged["momentum"].notna()
    )
    result = merged[mask].copy()
    if result.empty:
        return result

    result["score"] = (
        result["momentum"].rank(pct=True) * 0.35
        + result["pct_change_rt"].rank(pct=True) * 0.25
        + result["amount_ratio"].rank(pct=True) * 0.25
        + result["distance_to_ma_short"].rank(pct=True) * 0.15
    )
    result = result.sort_values(["score", "amount_rt"], ascending=False).head(top_n).copy()
    result["rank"] = range(1, len(result) + 1)
    return result


def _prepare_prediction_frame(panel: pd.DataFrame) -> pd.DataFrame:
    history = panel.sort_values(["code", "date"]).copy()
    history["next_return"] = history.groupby("code")["close"].shift(-1) / history["close"] - 1
    history["trend_flag"] = (history["close"] > history["ma_short"]) & (history["ma_short"] > history["ma_long"])
    history["distance_to_ma_short"] = history["close"] / history["ma_short"] - 1
    history["amount_ratio_daily"] = history["amount"] / history["avg_amount_20"]
    history["momentum_state"] = pd.cut(
        history["momentum"],
        bins=[-float("inf"), -0.05, 0.0, 0.05, 0.15, float("inf")],
        labels=False,
    )
    history["distance_state"] = pd.cut(
        history["distance_to_ma_short"],
        bins=[-float("inf"), -0.05, 0.0, 0.03, 0.08, float("inf")],
        labels=False,
    )
    history["amount_state"] = pd.cut(
        history["amount_ratio_daily"],
        bins=[-float("inf"), 0.7, 1.0, 1.5, 2.5, float("inf")],
        labels=False,
    )
    return history


def next_day_predictions(
    panel: pd.DataFrame,
    realtime_quotes: pd.DataFrame | None,
    top_n: int,
    min_samples: int = 40,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    history = _prepare_prediction_frame(panel)
    latest = latest_indicator_snapshot(history)
    if latest.empty:
        return pd.DataFrame()

    if realtime_quotes is not None and not realtime_quotes.empty:
        quote_cols = [
            "code",
            "latest",
            "amount",
            "pct_change",
            "quote_time",
            "quote_datetime",
        ]
        quotes = realtime_quotes[[col for col in quote_cols if col in realtime_quotes]].copy()
        quotes["code"] = quotes["code"].astype(str).str.zfill(6)
        latest = latest.merge(quotes, on="code", how="left", suffixes=("_daily", "_rt"))
    else:
        latest["latest"] = pd.NA
        latest["amount_rt"] = pd.NA
        latest["pct_change_rt"] = pd.NA
        latest["quote_time"] = pd.NA

    rows = []
    usable_history = history.dropna(
        subset=["next_return", "momentum_state", "distance_state", "amount_state", "ma_short", "ma_long"]
    )
    for _, current in latest.iterrows():
        code = current["code"]
        stock_history = usable_history[usable_history["code"] == code]
        if stock_history.empty:
            continue

        similar = stock_history[
            (stock_history["trend_flag"] == current["trend_flag"])
            & (stock_history["momentum_state"] == current["momentum_state"])
            & (stock_history["amount_state"] == current["amount_state"])
        ]
        sample_source = "同股相似状态"
        if len(similar) < min_samples:
            similar = stock_history[stock_history["trend_flag"] == current["trend_flag"]]
            sample_source = "同股同趋势"
        if len(similar) < max(10, min_samples // 2):
            similar = stock_history
            sample_source = "同股全部历史"
        if len(similar) < 10:
            continue

        base_probability = float((similar["next_return"] > 0).mean())
        expected_next_return = float(similar["next_return"].mean())
        probability = base_probability
        reasons = []

        latest_price = current.get("latest", pd.NA)
        if pd.isna(latest_price):
            latest_price = current["close"]

        if latest_price > current["ma_short"] and current["ma_short"] > current["ma_long"]:
            probability += 0.03
            reasons.append("多头趋势")
        else:
            probability -= 0.05
            reasons.append("趋势偏弱")

        if pd.notna(current.get("momentum")) and current["momentum"] > 0:
            probability += 0.02
            reasons.append("日线动量为正")
        else:
            probability -= 0.02
            reasons.append("日线动量不足")

        realtime_pct = current.get("pct_change_rt", pd.NA)
        if pd.notna(realtime_pct):
            probability += _clip(float(realtime_pct), -8.0, 8.0) * 0.006
            reasons.append("盘中上涨" if realtime_pct > 0 else "盘中走弱")

        realtime_amount = current.get("amount_rt", pd.NA)
        realtime_amount_ratio = pd.NA
        if pd.notna(realtime_amount) and pd.notna(current.get("avg_amount_20")) and current["avg_amount_20"] > 0:
            realtime_amount_ratio = float(realtime_amount) / float(current["avg_amount_20"])
            volume_adjust = min(max(realtime_amount_ratio - 1.0, 0.0) * 0.03, 0.06)
            if pd.notna(realtime_pct) and realtime_pct >= 0:
                probability += volume_adjust
                if realtime_amount_ratio > 1.0:
                    reasons.append("放量上攻")
            elif pd.notna(realtime_pct) and realtime_pct < 0:
                probability -= volume_adjust
                if realtime_amount_ratio > 1.0:
                    reasons.append("放量下跌")

        probability = _clip(probability, 0.05, 0.95)
        confidence = _clip(abs(probability - 0.5) * 1.5 + min(len(similar), 250) / 250 * 0.25, 0.05, 0.95)
        if probability >= 0.56:
            direction = "偏涨"
        elif probability <= 0.44:
            direction = "偏跌"
        else:
            direction = "中性"

        rows.append(
            {
                "code": code,
                "name": current.get("name", ""),
                "direction": direction,
                "probability_up": probability,
                "confidence": confidence,
                "expected_next_return": expected_next_return,
                "base_probability": base_probability,
                "sample_count": len(similar),
                "sample_source": sample_source,
                "latest": latest_price,
                "pct_change_rt": realtime_pct,
                "amount_ratio_rt": realtime_amount_ratio,
                "momentum": current.get("momentum", pd.NA),
                "distance_to_ma_short": latest_price / current["ma_short"] - 1 if current["ma_short"] else pd.NA,
                "quote_time": current.get("quote_time", pd.NA),
                "reason": "、".join(reasons[:4]),
            }
        )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["signal_strength"] = (result["probability_up"] - 0.5).abs() * result["confidence"]
    result = result.sort_values(["signal_strength", "probability_up"], ascending=False).head(top_n).copy()
    result["rank"] = range(1, len(result) + 1)
    return result


def _index_prediction_history(
    bars: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    momentum_window: int,
) -> pd.DataFrame:
    history = add_indicators(bars, ma_short, ma_long, momentum_window)
    history = history.sort_values("date").copy()
    history["next_return"] = history["close"].shift(-1) / history["close"] - 1
    history["trend_flag"] = (history["close"] > history["ma_short"]) & (history["ma_short"] > history["ma_long"])
    history["distance_to_ma_short"] = history["close"] / history["ma_short"] - 1
    history["amount_ratio_daily"] = history["amount"] / history["avg_amount_20"]
    history["momentum_state"] = pd.cut(
        history["momentum"],
        bins=[-float("inf"), -0.03, 0.0, 0.03, 0.08, float("inf")],
        labels=False,
    )
    history["distance_state"] = pd.cut(
        history["distance_to_ma_short"],
        bins=[-float("inf"), -0.02, 0.0, 0.015, 0.04, float("inf")],
        labels=False,
    )
    history["amount_state"] = pd.cut(
        history["amount_ratio_daily"],
        bins=[-float("inf"), 0.75, 1.0, 1.35, 2.0, float("inf")],
        labels=False,
    )
    return history


def market_index_predictions(
    index_bars: dict[str, pd.DataFrame],
    realtime_index_quotes: pd.DataFrame | None,
    ma_short: int,
    ma_long: int,
    momentum_window: int,
    min_samples: int = 80,
) -> pd.DataFrame:
    if not index_bars:
        return pd.DataFrame()

    quote_map: dict[str, pd.Series] = {}
    if realtime_index_quotes is not None and not realtime_index_quotes.empty:
        quote_map = {
            row["index_name"]: row
            for _, row in realtime_index_quotes.iterrows()
            if "index_name" in row and pd.notna(row["index_name"])
        }

    rows = []
    for index_name, bars in index_bars.items():
        if bars.empty:
            continue
        history = _index_prediction_history(bars, ma_short, ma_long, momentum_window)
        usable = history.dropna(subset=["next_return", "momentum_state", "distance_state", "amount_state"])
        if usable.empty:
            continue
        current = history.iloc[-1].copy()
        realtime = quote_map.get(index_name)

        latest_price = current["close"]
        realtime_pct = pd.NA
        realtime_amount_ratio = pd.NA
        quote_time = pd.NA
        if realtime is not None:
            latest_price = realtime.get("latest", latest_price)
            realtime_pct = realtime.get("pct_change", pd.NA)
            quote_time = realtime.get("quote_time", pd.NA)
            realtime_amount = realtime.get("amount", pd.NA)
            if pd.notna(realtime_amount) and pd.notna(current.get("avg_amount_20")) and current["avg_amount_20"] > 0:
                realtime_amount_ratio = float(realtime_amount) / float(current["avg_amount_20"])

        current_trend = latest_price > current["ma_short"] and current["ma_short"] > current["ma_long"]
        current_distance = latest_price / current["ma_short"] - 1 if current["ma_short"] else pd.NA

        similar = usable[
            (usable["trend_flag"] == current_trend)
            & (usable["momentum_state"] == current["momentum_state"])
            & (usable["amount_state"] == current["amount_state"])
        ]
        sample_source = "相似趋势/动量/量能"
        if len(similar) < min_samples:
            similar = usable[usable["trend_flag"] == current_trend]
            sample_source = "同趋势历史"
        if len(similar) < max(20, min_samples // 2):
            similar = usable
            sample_source = "全部历史"
        if len(similar) < 20:
            continue

        base_probability = float((similar["next_return"] > 0).mean())
        expected_next_return = float(similar["next_return"].mean())
        probability = base_probability
        reasons = []

        if current_trend:
            probability += 0.025
            reasons.append("指数多头趋势")
        else:
            probability -= 0.035
            reasons.append("指数趋势偏弱")

        if pd.notna(current.get("momentum")) and current["momentum"] > 0:
            probability += 0.018
            reasons.append("阶段动量为正")
        else:
            probability -= 0.018
            reasons.append("阶段动量不足")

        if pd.notna(realtime_pct):
            probability += _clip(float(realtime_pct), -4.0, 4.0) * 0.012
            reasons.append("盘中指数上涨" if realtime_pct > 0 else "盘中指数走弱")

        if pd.notna(realtime_amount_ratio):
            volume_adjust = min(max(float(realtime_amount_ratio) - 1.0, 0.0) * 0.02, 0.04)
            if pd.notna(realtime_pct) and realtime_pct >= 0:
                probability += volume_adjust
                if realtime_amount_ratio > 1.0:
                    reasons.append("指数放量上攻")
            elif pd.notna(realtime_pct) and realtime_pct < 0:
                probability -= volume_adjust
                if realtime_amount_ratio > 1.0:
                    reasons.append("指数放量下跌")

        probability = _clip(probability, 0.05, 0.95)
        confidence = _clip(abs(probability - 0.5) * 1.5 + min(len(similar), 500) / 500 * 0.3, 0.05, 0.95)
        if probability >= 0.55:
            direction = "偏涨"
        elif probability <= 0.45:
            direction = "偏跌"
        else:
            direction = "中性"

        rows.append(
            {
                "index_name": index_name,
                "direction": direction,
                "probability_up": probability,
                "confidence": confidence,
                "expected_next_return": expected_next_return,
                "base_probability": base_probability,
                "sample_count": len(similar),
                "sample_source": sample_source,
                "latest": latest_price,
                "pct_change": realtime_pct,
                "momentum": current.get("momentum", pd.NA),
                "distance_to_ma_short": current_distance,
                "amount_ratio": realtime_amount_ratio,
                "quote_time": quote_time,
                "reason": "、".join(reasons[:4]),
            }
        )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    order = {"上证指数": 0, "深证成指": 1, "创业板指": 2, "沪深300": 3}
    result["sort_order"] = result["index_name"].map(order).fillna(99)
    return result.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)


def operation_recommendations(
    panel: pd.DataFrame,
    realtime_quotes: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    top_n: int,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    latest = latest_indicator_snapshot(panel)
    if latest.empty:
        return pd.DataFrame()

    quotes = pd.DataFrame()
    if realtime_quotes is not None and not realtime_quotes.empty:
        quote_cols = ["code", "latest", "amount", "pct_change", "high", "low", "quote_time"]
        quotes = realtime_quotes[[col for col in quote_cols if col in realtime_quotes]].copy()
        quotes["code"] = quotes["code"].astype(str).str.zfill(6)
        quotes = quotes.rename(
            columns={
                "latest": "latest_rt",
                "amount": "amount_rt",
                "pct_change": "pct_change_rt",
                "high": "high_rt",
                "low": "low_rt",
            }
        )

    merged = latest.merge(quotes, on="code", how="left") if not quotes.empty else latest.copy()

    if predictions is not None and not predictions.empty:
        pred_cols = [
            "code",
            "direction",
            "probability_up",
            "confidence",
            "expected_next_return",
            "sample_count",
        ]
        pred = predictions[[col for col in pred_cols if col in predictions]].copy()
        pred["code"] = pred["code"].astype(str).str.zfill(6)
        merged = merged.merge(pred, on="code", how="left", suffixes=("", "_pred"))

    rows = []
    for _, row in merged.iterrows():
        latest_price = row.get("latest_rt", pd.NA)
        if pd.isna(latest_price):
            latest_price = row.get("close", pd.NA)
        if pd.isna(latest_price) or latest_price <= 0:
            continue

        probability = row.get("probability_up", 0.5)
        probability = 0.5 if pd.isna(probability) else float(probability)
        confidence = row.get("confidence", 0.1)
        confidence = 0.1 if pd.isna(confidence) else float(confidence)
        expected_next_return = row.get("expected_next_return", 0.0)
        expected_next_return = 0.0 if pd.isna(expected_next_return) else float(expected_next_return)
        realtime_pct = row.get("pct_change_rt", pd.NA)
        amount_rt = row.get("amount_rt", pd.NA)
        avg_amount_20 = row.get("avg_amount_20", pd.NA)
        amount_ratio = pd.NA
        if pd.notna(amount_rt) and pd.notna(avg_amount_20) and avg_amount_20 > 0:
            amount_ratio = float(amount_rt) / float(avg_amount_20)

        ma_short = row.get("ma_short", pd.NA)
        ma_long = row.get("ma_long", pd.NA)
        momentum = row.get("momentum", pd.NA)
        trend_good = pd.notna(ma_short) and pd.notna(ma_long) and latest_price > ma_short and ma_short > ma_long
        trend_bad = pd.notna(ma_short) and pd.notna(ma_long) and (latest_price < ma_short or ma_short < ma_long)
        distance_to_ma_short = latest_price / ma_short - 1 if pd.notna(ma_short) and ma_short else pd.NA

        stop_loss = latest_price * 0.94
        if pd.notna(ma_short):
            stop_loss = max(stop_loss, float(ma_short) * 0.98)
        if stop_loss >= latest_price:
            stop_loss = latest_price * 0.97
        risk_pct = max((latest_price - stop_loss) / latest_price, 0.01)

        reward_risk = _clip(1.25 + confidence * 1.3 + max(probability - 0.5, 0) * 2.0, 1.0, 3.0)
        take_profit = latest_price * (1 + risk_pct * reward_risk)
        gain_pct = (take_profit - latest_price) / latest_price
        expected_value = probability * gain_pct - (1 - probability) * risk_pct

        buy_score = 0.0
        sell_score = 0.0
        reasons = []
        if trend_good:
            buy_score += 0.25
            reasons.append("趋势向上")
        if trend_bad:
            sell_score += 0.25
            reasons.append("趋势破坏")
        if pd.notna(momentum) and momentum > 0:
            buy_score += min(float(momentum), 0.3) * 0.6
            reasons.append("动量为正")
        elif pd.notna(momentum):
            sell_score += 0.10
            reasons.append("动量不足")
        if probability >= 0.58:
            buy_score += (probability - 0.5) * 1.2
            reasons.append("上涨概率较高")
        elif probability <= 0.44:
            sell_score += (0.5 - probability) * 1.2
            reasons.append("上涨概率偏低")
        if pd.notna(realtime_pct):
            if realtime_pct > 0:
                buy_score += min(float(realtime_pct), 8.0) * 0.018
                reasons.append("盘中走强")
            else:
                sell_score += min(abs(float(realtime_pct)), 8.0) * 0.02
                reasons.append("盘中走弱")
        if pd.notna(amount_ratio):
            if amount_ratio > 1.0 and (pd.isna(realtime_pct) or realtime_pct >= 0):
                buy_score += min(amount_ratio - 1.0, 2.0) * 0.08
                reasons.append("放量配合")
            elif amount_ratio > 1.0 and pd.notna(realtime_pct) and realtime_pct < 0:
                sell_score += min(amount_ratio - 1.0, 2.0) * 0.10
                reasons.append("放量下跌")
        if pd.notna(distance_to_ma_short) and distance_to_ma_short > 0.15:
            buy_score -= 0.12
            reasons.append("短线偏离过大")
        if expected_value > 0:
            buy_score += min(expected_value * 10, 0.12)
        else:
            sell_score += min(abs(expected_value) * 10, 0.12)

        if sell_score >= 0.50:
            action = "卖出/回避"
            action_score = sell_score
            suggested_position = "0%-5%"
        elif buy_score >= 0.62 and expected_value > 0:
            action = "买入候选"
            action_score = buy_score
            suggested_position = "10%-25%" if confidence >= 0.25 else "5%-15%"
        elif buy_score >= 0.48 and expected_value >= -0.005:
            action = "持有观察"
            action_score = buy_score
            suggested_position = "已有仓位持有；无仓等回踩"
        else:
            action = "观望"
            action_score = max(buy_score, sell_score)
            suggested_position = "0%-10%"

        rows.append(
            {
                "code": row["code"],
                "name": row.get("name", ""),
                "action": action,
                "action_score": action_score,
                "expected_value": expected_value,
                "reward_risk": reward_risk,
                "risk_pct": risk_pct,
                "gain_pct": gain_pct,
                "probability_up": probability,
                "confidence": confidence,
                "latest": latest_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "suggested_position": suggested_position,
                "pct_change_rt": realtime_pct,
                "amount_ratio_rt": amount_ratio,
                "momentum": momentum,
                "distance_to_ma_short": distance_to_ma_short,
                "quote_time": row.get("quote_time", pd.NA),
                "reason": "、".join(dict.fromkeys(reasons[:5])),
            }
        )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    priority = {"买入候选": 0, "卖出/回避": 1, "持有观察": 2, "观望": 3}
    result["priority"] = result["action"].map(priority).fillna(9)
    result = result.sort_values(
        ["priority", "action_score", "expected_value"],
        ascending=[True, False, False],
    ).head(top_n).copy()
    result["rank"] = range(1, len(result) + 1)
    return result.drop(columns=["priority"])


def optimize_trade_plan(
    operations: pd.DataFrame,
    holdings: pd.DataFrame | None,
    total_assets: float,
    cash_available: float,
    reserve_cash_pct: float,
    max_position_pct: float,
    max_buy_candidates: int,
    lot_size: int = 100,
) -> pd.DataFrame:
    columns = [
        "rank",
        "code",
        "name",
        "recommendation",
        "action",
        "trade_shares",
        "trade_value",
        "current_shares",
        "current_value",
        "target_value",
        "target_position_pct",
        "expected_profit_amount",
        "max_risk_amount",
        "expected_value",
        "risk_pct",
        "probability_up",
        "confidence",
        "latest",
        "stop_loss",
        "take_profit",
        "reason",
    ]
    if operations.empty:
        return pd.DataFrame(columns=columns)

    ops = operations.copy()
    ops["code"] = ops["code"].astype(str).str.zfill(6)
    numeric_columns = [
        "latest",
        "action_score",
        "expected_value",
        "reward_risk",
        "risk_pct",
        "gain_pct",
        "probability_up",
        "confidence",
        "stop_loss",
        "take_profit",
    ]
    for column in numeric_columns:
        if column not in ops:
            ops[column] = pd.NA
        ops[column] = pd.to_numeric(ops[column], errors="coerce")

    holding_frame = pd.DataFrame(columns=["code", "shares", "cost_price"])
    if holdings is not None and not holdings.empty:
        holding_frame = holdings.copy()
        if "cost_price" not in holding_frame:
            holding_frame["cost_price"] = pd.NA
        holding_frame["code"] = holding_frame["code"].astype(str).str.zfill(6)
        holding_frame["shares"] = pd.to_numeric(holding_frame["shares"], errors="coerce").fillna(0.0)
        holding_frame["cost_price"] = pd.to_numeric(holding_frame["cost_price"], errors="coerce")
        holding_frame = holding_frame[holding_frame["shares"] > 0].copy()
        holding_frame["position_cost"] = holding_frame["shares"] * holding_frame["cost_price"].fillna(0.0)
        holding_frame = (
            holding_frame.groupby("code", as_index=False)
            .agg(shares=("shares", "sum"), position_cost=("position_cost", "sum"))
            .reset_index(drop=True)
        )
        holding_frame["cost_price"] = holding_frame["position_cost"] / holding_frame["shares"]

    ops = ops.merge(holding_frame[["code", "shares", "cost_price"]], on="code", how="left")
    ops["current_shares"] = pd.to_numeric(ops["shares"], errors="coerce").fillna(0.0)
    ops["current_value"] = ops["current_shares"] * ops["latest"].fillna(0.0)

    total_assets = max(float(total_assets or 0.0), 0.0)
    cash_available = max(float(cash_available or 0.0), 0.0)
    reserve_cash_pct = _clip(float(reserve_cash_pct or 0.0), 0.0, 0.9)
    max_position_pct = _clip(float(max_position_pct or 0.0), 0.01, 1.0)
    lot_size = max(int(lot_size), 1)
    max_buy_candidates = max(int(max_buy_candidates), 1)

    known_position_value = float(ops["current_value"].sum())
    if total_assets <= 0:
        total_assets = cash_available + known_position_value
    if total_assets <= 0:
        return pd.DataFrame(columns=columns)

    ops["risk_pct"] = ops["risk_pct"].fillna((ops["latest"] - ops["stop_loss"]) / ops["latest"])
    ops["risk_pct"] = ops["risk_pct"].replace([float("inf"), -float("inf")], pd.NA).fillna(0.05)
    ops["risk_pct"] = ops["risk_pct"].clip(lower=0.005, upper=0.2)
    ops["expected_value"] = ops["expected_value"].fillna(0.0)
    ops["probability_up"] = ops["probability_up"].fillna(0.5)
    ops["confidence"] = ops["confidence"].fillna(0.1)
    ops["action_score"] = ops["action_score"].fillna(0.0)

    sell_mask = (
        (ops["current_shares"] > 0)
        & (
            ops["action"].eq("卖出/回避")
            | (ops["expected_value"] < -0.005)
            | (ops["probability_up"] < 0.44)
        )
    )
    forced_sell_value = float(ops.loc[sell_mask, "current_value"].sum())
    exposure_cap = total_assets * (1 - reserve_cash_pct)
    available_for_buy = min(
        cash_available + forced_sell_value,
        max(0.0, exposure_cap - known_position_value + forced_sell_value),
    )

    ops["optimizer_score"] = (
        ops["expected_value"].clip(lower=0.0) * 5.0
        + (ops["probability_up"] - 0.5).clip(lower=0.0) * ops["confidence"] * 2.0
        + ops["action_score"].clip(lower=0.0) * 0.18
    )
    buy_mask = (
        ops["action"].eq("买入候选")
        & (ops["expected_value"] > 0)
        & (ops["probability_up"] >= 0.55)
        & ops["latest"].notna()
        & (ops["latest"] > 0)
    )
    buy_universe = ops[buy_mask].sort_values("optimizer_score", ascending=False).head(max_buy_candidates)
    buy_increments = pd.Series(0.0, index=ops.index)
    if not buy_universe.empty and available_for_buy > 0:
        max_value_per_stock = total_assets * max_position_pct
        caps = (max_value_per_stock - buy_universe["current_value"]).clip(lower=0.0)
        allocations = _allocate_with_caps(buy_universe["optimizer_score"], available_for_buy, caps)
        buy_increments.loc[allocations.index] = allocations

    ops["target_value"] = ops["current_value"]
    ops.loc[sell_mask, "target_value"] = 0.0
    ops.loc[ops["current_shares"] <= 0, "target_value"] = 0.0
    ops["target_value"] += buy_increments
    ops["target_value"] = ops["target_value"].clip(lower=0.0, upper=total_assets * max_position_pct)

    rows = []
    for _, row in ops.iterrows():
        latest_price = row.get("latest", pd.NA)
        if pd.isna(latest_price) or latest_price <= 0:
            continue
        current_shares = float(row["current_shares"])
        current_value = float(row["current_value"])
        target_value = float(row["target_value"])
        raw_trade_shares = (target_value - current_value) / float(latest_price)

        if raw_trade_shares > 0:
            trade_shares = int(raw_trade_shares // lot_size * lot_size)
        elif raw_trade_shares < 0:
            if target_value <= 0:
                trade_shares = -int(current_shares)
            else:
                sell_lots = int(abs(raw_trade_shares) // lot_size * lot_size)
                trade_shares = -min(sell_lots, int(current_shares))
        else:
            trade_shares = 0

        trade_value = trade_shares * float(latest_price)
        target_value_after_lot = max(current_value + trade_value, 0.0)

        if trade_shares > 0:
            recommendation = "买入"
        elif trade_shares < 0:
            recommendation = "卖出"
        elif current_shares > 0 and row["action"] == "买入候选":
            recommendation = "持有/可加仓"
        elif current_shares > 0:
            recommendation = "持有观察"
        elif row["action"] == "卖出/回避":
            recommendation = "回避"
        else:
            recommendation = "观望"

        rows.append(
            {
                "code": row["code"],
                "name": row.get("name", ""),
                "recommendation": recommendation,
                "action": row["action"],
                "trade_shares": trade_shares,
                "trade_value": trade_value,
                "current_shares": current_shares,
                "current_value": current_value,
                "target_value": target_value_after_lot,
                "target_position_pct": target_value_after_lot / total_assets,
                "expected_profit_amount": target_value_after_lot * float(row["expected_value"]),
                "max_risk_amount": target_value_after_lot * float(row["risk_pct"]),
                "expected_value": float(row["expected_value"]),
                "risk_pct": float(row["risk_pct"]),
                "probability_up": float(row["probability_up"]),
                "confidence": float(row["confidence"]),
                "latest": float(latest_price),
                "stop_loss": row.get("stop_loss", pd.NA),
                "take_profit": row.get("take_profit", pd.NA),
                "reason": row.get("reason", ""),
            }
        )

    known_codes = set(ops["code"])
    for _, holding in holding_frame[~holding_frame["code"].isin(known_codes)].iterrows():
        rows.append(
            {
                "code": holding["code"],
                "name": "",
                "recommendation": "无法定价",
                "action": "不在当前扫描",
                "trade_shares": 0,
                "trade_value": 0.0,
                "current_shares": float(holding["shares"]),
                "current_value": pd.NA,
                "target_value": pd.NA,
                "target_position_pct": pd.NA,
                "expected_profit_amount": pd.NA,
                "max_risk_amount": pd.NA,
                "expected_value": pd.NA,
                "risk_pct": pd.NA,
                "probability_up": pd.NA,
                "confidence": pd.NA,
                "latest": pd.NA,
                "stop_loss": pd.NA,
                "take_profit": pd.NA,
                "reason": "该持仓不在当前股票池或操作结果中",
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    priority = {"买入": 0, "卖出": 1, "持有/可加仓": 2, "持有观察": 3, "观望": 4, "回避": 5, "无法定价": 6}
    result["priority"] = result["recommendation"].map(priority).fillna(9)
    result = result.sort_values(
        ["priority", "expected_profit_amount", "probability_up"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result[columns]
