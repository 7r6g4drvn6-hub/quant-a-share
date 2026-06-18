from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DAILY_DIR = DATA_DIR / "daily"
INDEX_DIR = DATA_DIR / "index"
STOCK_LIST_PATH = DATA_DIR / "stock_list.csv"

EASTMONEY_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SINA_LIST_URL = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
SINA_KLINE_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20kline=/CN_MarketData.getKLineData"
SINA_REALTIME_URL = "https://hq.sinajs.cn/list={symbols}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
}

STOCK_FIELDS = {
    "f12": "code",
    "f14": "name",
    "f2": "latest",
    "f3": "pct_change",
    "f5": "volume",
    "f6": "amount",
    "f8": "turnover",
    "f9": "pe",
    "f10": "volume_ratio",
    "f18": "prev_close",
    "f20": "total_market_cap",
    "f21": "float_market_cap",
    "f23": "pb",
    "f24": "pct_change_60d",
    "f25": "pct_change_ytd",
    "f26": "list_date",
}

KLINE_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_change",
    "price_change",
    "turnover",
]

INDEX_MAP = {
    "上证指数": "1.000001",
    "深证成指": "0.399001",
    "创业板指": "0.399006",
    "沪深300": "1.000300",
}

SINA_INDEX_SYMBOLS = {
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
    "沪深300": "sh000300",
}


class DataFetchError(RuntimeError):
    """Raised when a remote market-data request fails."""


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DAILY_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


def _request_json(url: str, params: dict) -> dict:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if payload.get("rc") not in (0, None):
                raise DataFetchError(f"Eastmoney returned rc={payload.get('rc')}: {payload}")
            return payload
        except Exception as exc:  # noqa: BLE001 - retry public data endpoints.
            last_error = exc
            time.sleep(0.4)
    raise DataFetchError(str(last_error))


def _date_to_yyyymmdd(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value.replace("-", "")
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%Y%m%d")


def normalize_code(code: str | int) -> str:
    text = str(code).strip()
    if "." in text:
        text = text.split(".")[-1]
    return text.zfill(6)


def market_for_code(code: str | int) -> int:
    code = normalize_code(code)
    return 1 if code.startswith(("5", "6", "9")) else 0


def secid_for_code(code: str | int) -> str:
    code = normalize_code(code)
    return f"{market_for_code(code)}.{code}"


def sina_symbol_for_code(code: str | int) -> str:
    code = normalize_code(code)
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{code}"


def _finalize_stock_list(df: pd.DataFrame) -> pd.DataFrame:
    df["code"] = df["code"].map(normalize_code)
    df["market"] = df["code"].map(market_for_code)
    df["secid"] = df["code"].map(secid_for_code)

    numeric_columns = [
        "latest",
        "pct_change",
        "volume",
        "amount",
        "turnover",
        "pe",
        "volume_ratio",
        "prev_close",
        "total_market_cap",
        "float_market_cap",
        "pb",
        "pct_change_60d",
        "pct_change_ytd",
    ]
    for column in numeric_columns:
        if column not in df:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "list_date" not in df:
        df["list_date"] = pd.NaT
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df["is_st"] = df["name"].str.contains("ST|退", case=False, regex=True, na=False)
    df["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df.sort_values("amount", ascending=False, na_position="last").reset_index(drop=True)


def _fetch_stock_list_eastmoney() -> pd.DataFrame:
    ensure_data_dirs()
    params = {
        "pn": 1,
        "pz": 6000,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f6",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": ",".join(STOCK_FIELDS.keys()),
    }
    payload = _request_json(EASTMONEY_CLIST_URL, params)
    rows = payload.get("data", {}).get("diff", [])
    if not rows:
        raise DataFetchError("No stock list rows returned")

    df = pd.DataFrame(rows).rename(columns=STOCK_FIELDS)
    df["list_date"] = pd.to_datetime(
        df["list_date"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    df = _finalize_stock_list(df)
    df.to_csv(STOCK_LIST_PATH, index=False)
    return df


def _fetch_stock_list_sina(max_pages: int = 10, page_size: int = 100) -> pd.DataFrame:
    ensure_data_dirs()
    rows = []
    for page in range(1, max_pages + 1):
        params = {
            "page": page,
            "num": page_size,
            "sort": "amount",
            "asc": 0,
            "node": "hs_a",
            "symbol": "",
            "_s_r_a": "init",
        }
        response = requests.get(SINA_LIST_URL, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        page_rows = response.json()
        if not page_rows:
            break
        rows.extend(page_rows)
        if len(page_rows) < page_size:
            break
        time.sleep(0.15)

    if not rows:
        raise DataFetchError("No stock list rows returned from Sina")

    raw = pd.DataFrame(rows)
    df = pd.DataFrame(
        {
            "code": raw["code"],
            "name": raw["name"],
            "latest": raw.get("trade"),
            "pct_change": raw.get("changepercent"),
            "volume": raw.get("volume"),
            "amount": raw.get("amount"),
            "turnover": raw.get("turnoverratio"),
            "pe": raw.get("per"),
            "volume_ratio": pd.NA,
            "prev_close": raw.get("settlement"),
            "total_market_cap": pd.to_numeric(raw.get("mktcap"), errors="coerce") * 10000,
            "float_market_cap": pd.to_numeric(raw.get("nmc"), errors="coerce") * 10000,
            "pb": raw.get("pb"),
            "pct_change_60d": pd.NA,
            "pct_change_ytd": pd.NA,
            "list_date": pd.NaT,
        }
    )
    df = _finalize_stock_list(df)
    df.to_csv(STOCK_LIST_PATH, index=False)
    return df


def fetch_stock_list() -> pd.DataFrame:
    try:
        return _fetch_stock_list_eastmoney()
    except Exception:
        return _fetch_stock_list_sina()


def load_stock_list(refresh: bool = False) -> pd.DataFrame:
    ensure_data_dirs()
    if refresh or not STOCK_LIST_PATH.exists():
        return fetch_stock_list()
    return pd.read_csv(STOCK_LIST_PATH, dtype={"code": str}, parse_dates=["list_date"])


def fetch_realtime_quotes(codes: Iterable[str], batch_size: int = 80) -> pd.DataFrame:
    code_list = [normalize_code(code) for code in codes]
    rows = []
    for start in range(0, len(code_list), batch_size):
        batch = code_list[start : start + batch_size]
        symbols = ",".join(sina_symbol_for_code(code) for code in batch)
        response = requests.get(
            SINA_REALTIME_URL.format(symbols=symbols),
            headers={**HEADERS, "Referer": "https://finance.sina.com.cn"},
            timeout=10,
        )
        response.raise_for_status()
        response.encoding = "gbk"
        for line in response.text.splitlines():
            match = re.match(r'var hq_str_(?P<symbol>[^=]+)="(?P<body>.*)";', line.strip())
            if not match:
                continue
            body = match.group("body")
            if not body:
                continue
            fields = body.split(",")
            if len(fields) < 32:
                continue
            symbol = match.group("symbol")
            code = normalize_code(symbol[2:])
            prev_close = pd.to_numeric(fields[2], errors="coerce")
            latest = pd.to_numeric(fields[3], errors="coerce")
            pct_change = (latest / prev_close - 1) * 100 if prev_close and prev_close > 0 else pd.NA
            rows.append(
                {
                    "code": code,
                    "symbol": symbol,
                    "name": fields[0],
                    "open": pd.to_numeric(fields[1], errors="coerce"),
                    "prev_close": prev_close,
                    "latest": latest,
                    "high": pd.to_numeric(fields[4], errors="coerce"),
                    "low": pd.to_numeric(fields[5], errors="coerce"),
                    "bid": pd.to_numeric(fields[6], errors="coerce"),
                    "ask": pd.to_numeric(fields[7], errors="coerce"),
                    "volume": pd.to_numeric(fields[8], errors="coerce"),
                    "amount": pd.to_numeric(fields[9], errors="coerce"),
                    "pct_change": pct_change,
                    "quote_date": fields[30],
                    "quote_time": fields[31],
                }
            )
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(
            columns=[
                "code",
                "symbol",
                "name",
                "open",
                "prev_close",
                "latest",
                "high",
                "low",
                "bid",
                "ask",
                "volume",
                "amount",
                "pct_change",
                "quote_date",
                "quote_time",
            ]
        )
    df = pd.DataFrame(rows)
    df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
    df["quote_datetime"] = pd.to_datetime(
        df["quote_date"].astype(str) + " " + df["quote_time"].astype(str),
        errors="coerce",
    )
    return df.sort_values("amount", ascending=False, na_position="last").reset_index(drop=True)


def fetch_realtime_index_quotes() -> pd.DataFrame:
    symbols = ",".join(SINA_INDEX_SYMBOLS.values())
    response = requests.get(
        SINA_REALTIME_URL.format(symbols=symbols),
        headers={**HEADERS, "Referer": "https://finance.sina.com.cn"},
        timeout=10,
    )
    response.raise_for_status()
    response.encoding = "gbk"

    symbol_to_name = {symbol: name for name, symbol in SINA_INDEX_SYMBOLS.items()}
    rows = []
    for line in response.text.splitlines():
        match = re.match(r'var hq_str_(?P<symbol>[^=]+)="(?P<body>.*)";', line.strip())
        if not match:
            continue
        symbol = match.group("symbol")
        body = match.group("body")
        fields = body.split(",")
        if len(fields) < 32 or not body:
            continue
        prev_close = pd.to_numeric(fields[2], errors="coerce")
        latest = pd.to_numeric(fields[3], errors="coerce")
        price_change = latest - prev_close if pd.notna(prev_close) and pd.notna(latest) else pd.NA
        pct_change = price_change / prev_close * 100 if pd.notna(price_change) and prev_close else pd.NA
        rows.append(
            {
                "index_name": symbol_to_name.get(symbol, fields[0]),
                "symbol": symbol,
                "open": pd.to_numeric(fields[1], errors="coerce"),
                "prev_close": prev_close,
                "latest": latest,
                "high": pd.to_numeric(fields[4], errors="coerce"),
                "low": pd.to_numeric(fields[5], errors="coerce"),
                "volume": pd.to_numeric(fields[8], errors="coerce"),
                "amount": pd.to_numeric(fields[9], errors="coerce"),
                "price_change": price_change,
                "pct_change": pct_change,
                "quote_date": fields[30],
                "quote_time": fields[31],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "index_name",
                "symbol",
                "latest",
                "price_change",
                "pct_change",
                "amount",
                "quote_date",
                "quote_time",
            ]
        )
    df["quote_datetime"] = pd.to_datetime(
        df["quote_date"].astype(str) + " " + df["quote_time"].astype(str),
        errors="coerce",
    )
    df["amount_yi"] = df["amount"] / 100000000
    order = list(SINA_INDEX_SYMBOLS)
    df["sort_order"] = df["index_name"].map({name: idx for idx, name in enumerate(order)})
    return df.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)


def _fetch_daily_bars_eastmoney(
    code: str,
    start: date | datetime | str,
    end: date | datetime | str,
    adjust: str,
) -> pd.DataFrame:
    fqt = {"none": 0, "qfq": 1, "hfq": 2}.get(adjust, 1)
    params = {
        "secid": secid_for_code(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "klt": 101,
        "fqt": fqt,
        "beg": _date_to_yyyymmdd(start),
        "end": _date_to_yyyymmdd(end),
        "lmt": 1000000,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    payload = _request_json(EASTMONEY_KLINE_URL, params)
    rows = payload.get("data", {}).get("klines", [])
    if not rows:
        raise DataFetchError(f"No kline rows returned for {code}")

    records = [row.split(",") for row in rows]
    df = pd.DataFrame(records, columns=KLINE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    for column in KLINE_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["code"] = code
    return df.sort_values("date").reset_index(drop=True)


def _fetch_daily_bars_sina(
    code_or_symbol: str,
    start: date | datetime | str,
    end: date | datetime | str,
) -> pd.DataFrame:
    symbol = code_or_symbol if code_or_symbol.startswith(("sh", "sz")) else sina_symbol_for_code(code_or_symbol)
    code = normalize_code(code_or_symbol[2:] if code_or_symbol.startswith(("sh", "sz")) else code_or_symbol)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    calendar_days = max((end_dt - start_dt).days, 1)
    datalen = min(1500, max(300, int(calendar_days * 1.1)))
    params = {
        "symbol": symbol,
        "scale": 240,
        "ma": "no",
        "datalen": datalen,
    }
    response = requests.get(SINA_KLINE_URL, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    text = response.text
    match = re.search(r"(\[.*\])", text, flags=re.S)
    if not match:
        raise DataFetchError(f"No Sina kline payload returned for {code}")
    rows = json.loads(match.group(1))
    if not rows:
        raise DataFetchError(f"No Sina kline rows returned for {code}")

    df = pd.DataFrame(rows).rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for column in ["open", "close", "high", "low", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["amount"] = df["close"] * df["volume"]
    df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
    df["pct_change"] = df["close"].pct_change() * 100
    df["price_change"] = df["close"].diff()
    df["turnover"] = pd.NA
    df["code"] = code
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    return df[KLINE_COLUMNS + ["code"]].sort_values("date").reset_index(drop=True)


def fetch_daily_bars(
    code: str | int,
    start: date | datetime | str,
    end: date | datetime | str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    code = normalize_code(code)
    try:
        return _fetch_daily_bars_eastmoney(code, start, end, adjust)
    except Exception:
        return _fetch_daily_bars_sina(code, start, end)


def daily_cache_path(code: str | int) -> Path:
    return DAILY_DIR / f"{normalize_code(code)}.csv"


def index_cache_path(name: str) -> Path:
    return INDEX_DIR / f"{name}.csv"


def save_daily_bars(code: str | int, bars: pd.DataFrame) -> None:
    ensure_data_dirs()
    bars.to_csv(daily_cache_path(code), index=False)


def load_daily_bars(code: str | int) -> pd.DataFrame:
    path = daily_cache_path(code)
    return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])


def load_or_fetch_daily_bars(
    code: str | int,
    start: date | datetime | str,
    end: date | datetime | str,
    refresh: bool = False,
) -> pd.DataFrame:
    path = daily_cache_path(code)
    if refresh or not path.exists():
        bars = fetch_daily_bars(code, start, end)
        save_daily_bars(code, bars)
        return bars
    bars = load_daily_bars(code)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return bars[(bars["date"] >= start_dt) & (bars["date"] <= end_dt)].copy()


def update_daily_cache(
    codes: Iterable[str],
    start: date | datetime | str,
    end: date | datetime | str,
    progress: Callable[[int, int, str, str], None] | None = None,
    sleep_seconds: float = 0.05,
    max_workers: int = 6,
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    code_list = [normalize_code(code) for code in codes]
    bars_by_code: dict[str, pd.DataFrame] = {}
    errors: list[tuple[str, str]] = []
    total = len(code_list)

    def fetch_and_save(code: str) -> tuple[str, pd.DataFrame]:
        bars = fetch_daily_bars(code, start, end)
        save_daily_bars(code, bars)
        return code, bars

    if total == 0:
        return bars_by_code, errors

    workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_and_save, code): code for code in code_list}
        for idx, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                code, bars = future.result()
                bars_by_code[code] = bars
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - surface per-code data failures in UI.
                errors.append((code, str(exc)))
                status = "failed"
            if progress:
                progress(idx, total, code, status)
            time.sleep(sleep_seconds)
    return bars_by_code, errors


def update_daily_cache_sequential(
    codes: Iterable[str],
    start: date | datetime | str,
    end: date | datetime | str,
    progress: Callable[[int, int, str, str], None] | None = None,
    sleep_seconds: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    code_list = [normalize_code(code) for code in codes]
    bars_by_code: dict[str, pd.DataFrame] = {}
    errors: list[tuple[str, str]] = []
    total = len(code_list)
    for idx, code in enumerate(code_list, start=1):
        try:
            bars = fetch_daily_bars(code, start, end)
            save_daily_bars(code, bars)
            bars_by_code[code] = bars
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - surface per-code data failures in UI.
            errors.append((code, str(exc)))
            status = "failed"
        if progress:
            progress(idx, total, code, status)
        time.sleep(sleep_seconds)
    return bars_by_code, errors


def load_cached_bars_for_codes(
    codes: Iterable[str],
    start: date | datetime | str,
    end: date | datetime | str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    bars_by_code: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for code in [normalize_code(code) for code in codes]:
        path = daily_cache_path(code)
        if not path.exists():
            missing.append(code)
            continue
        bars = load_daily_bars(code)
        bars = bars[(bars["date"] >= start_dt) & (bars["date"] <= end_dt)].copy()
        if bars.empty:
            missing.append(code)
        else:
            bars_by_code[code] = bars
    return bars_by_code, missing


def fetch_index_bars(
    name: str,
    start: date | datetime | str,
    end: date | datetime | str,
    refresh: bool = False,
) -> pd.DataFrame:
    ensure_data_dirs()
    if name not in INDEX_MAP:
        raise ValueError(f"Unknown index: {name}")
    path = index_cache_path(name)
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        return df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()

    try:
        params = {
            "secid": INDEX_MAP[name],
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "klt": 101,
            "fqt": 1,
            "beg": _date_to_yyyymmdd(start),
            "end": _date_to_yyyymmdd(end),
            "lmt": 1000000,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
        payload = _request_json(EASTMONEY_KLINE_URL, params)
        rows = payload.get("data", {}).get("klines", [])
        if not rows:
            raise DataFetchError(f"No kline rows returned for index {name}")
        df = pd.DataFrame([row.split(",") for row in rows], columns=KLINE_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        for column in KLINE_COLUMNS[1:]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    except Exception:
        df = _fetch_daily_bars_sina(SINA_INDEX_SYMBOLS[name], start, end).drop(columns=["code"])
    df["name"] = name
    df.to_csv(path, index=False)
    return df
