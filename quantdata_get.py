# quantdata_get.py
# Function:
# Retrieve daily price data for a single stock within a specified time range,
# compute technical indicators, and construct a window sequence with shape
# (N, T, F) using the past T trading days while preserving the corresponding dates.
#
# Note:
# This module internally uses AkShare's stock_zh_a_hist interface. The returned
# data include volume and, when available, amount. If amount is entirely missing,
# it is approximated by close * volume. The external interface remains unchanged.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd



# Configuration

@dataclass
class QuantConfig:
    """
    Basic configuration for quantitative data processing.
    """
    # CSV file used to store price data for all stocks.
    prices_path: str = "/home/guyh/hys/prices_multi.csv"

    # Whether to fetch data online from AkShare if local data are unavailable.
    use_akshare: bool = True

# AkShare data retrieval

def _fetch_and_append_symbol_data(
    config: QuantConfig,
    symbol: str,
    start_date: str,
    end_date: str
) -> None:
    """
    Retrieve complete daily K-line data using AkShare's stock_zh_a_hist
    interface and append the data to prices_multi.csv.

    Parameters
    ----------
    symbol:
        Stock code in the user-defined format, e.g., "A002538".
    start_date, end_date:
        Date strings in the format "YYYY-MM-DD".
    """
    print(f"[AkShare] Fetching daily price data for {symbol} using stock_zh_a_hist ...")

    try:
        import akshare as ak
    except ImportError:
        print(
            "[AkShare] akshare is not installed. "
            "Please install it first with: pip install akshare"
        )
        return

    # Convert symbol format, e.g., A002538 -> 002538.
    raw_sym = symbol.replace("A", "")

    # Convert date format, e.g., 2023-01-01 -> 20230101.
    sd = start_date.replace("-", "")
    ed = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=raw_sym,
            period="daily",
            start_date=sd,
            end_date=ed,
            adjust="qfq"  # Forward-adjusted prices. Modify if needed.
        )
    except Exception as e:
        print(f"[AkShare] Failed to fetch data for {symbol}. Reason: {e}")
        return

    if df is None or df.empty:
        print(
            f"[AkShare] No data were retrieved for {symbol} "
            f"within {start_date} to {end_date}."
        )
        return

    # Rename columns according to the current AkShare output format.
    # Field names may vary slightly across AkShare versions.
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df.rename(columns=rename_map, inplace=True)

    # Add the symbol column and standardize the date and numeric fields.
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        df["amount"] = np.nan

    # Retain only the core columns.
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]]

    path = config.prices_path

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_old["date"] = pd.to_datetime(df_old["date"], format="mixed", errors="coerce")
        df_all = pd.concat([df_old, df], ignore_index=True)
    else:
        df_all = df

    # Save the updated price table.
    df_all.to_csv(path, index=False, encoding="utf-8-sig")
    print(
        f"[AkShare] Data for {symbol} have been written to {path}. "
        f"New rows added: {len(df)}."
    )


def _load_price_data(
    config: QuantConfig,
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Load daily price data for a given symbol from prices_multi.csv within
    the specified time range. If the local file is unavailable and
    use_akshare=True, data are fetched once from AkShare.
    """
    path = config.prices_path

    if not os.path.exists(path):
        print(f"[QuantData] Local price file does not exist: {path}.")
        if config.use_akshare:
            # Attempt to fetch data from AkShare and write them to disk.
            _fetch_and_append_symbol_data(config, symbol, start_date, end_date)
        else:
            raise FileNotFoundError(
                f"{path} does not exist and use_akshare=False."
            )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[QuantData] {path} was still not found after attempting AkShare retrieval."
        )

    df_all = pd.read_csv(path)

    # Support both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS" date formats.
    df_all["date"] = pd.to_datetime(df_all["date"], format="mixed", errors="coerce")

    # Filter by symbol and date range.
    mask_symbol = df_all["symbol"] == symbol
    mask_date = (df_all["date"] >= pd.to_datetime(start_date)) & (
        df_all["date"] <= pd.to_datetime(end_date)
    )
    df = df_all.loc[mask_symbol & mask_date].copy()

    if df.empty and config.use_akshare:
        # Attempt another data retrieval if no local records are found.
        _fetch_and_append_symbol_data(config, symbol, start_date, end_date)

        df_all = pd.read_csv(path)
        df_all["date"] = pd.to_datetime(df_all["date"], format="mixed", errors="coerce")

        mask_symbol = df_all["symbol"] == symbol
        mask_date = (df_all["date"] >= pd.to_datetime(start_date)) & (
            df_all["date"] <= pd.to_datetime(end_date)
        )
        df = df_all.loc[mask_symbol & mask_date].copy()

    if df.empty:
        raise ValueError(
            f"No data found for symbol={symbol} within the range "
            f"{start_date} to {end_date}."
        )

    # Ensure that numeric columns are converted to numeric types.
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If amount is entirely missing, approximate it using close * volume.
    if "amount" in df.columns:
        if df["amount"].isna().all():
            print(
                f"[QuantData] The amount column for {symbol} is entirely NaN. "
                f"Approximating amount using close * volume."
            )
            df["amount"] = df["close"] * df["volume"]

    df = df.sort_values("date").reset_index(drop=True)

    print(f"[QuantData] Loaded {len(df)} raw daily records for {symbol}.")
    print(df.head())

    return df

# Technical indicator calculation

def _calc_ma(series: pd.Series, window: int) -> pd.Series:
    """Calculate the simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the relative strength index."""
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=close.index)
    loss = pd.Series(loss, index=close.index)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, 1e-10)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD indicators: DIF, DEA, and HIST."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea

    return pd.DataFrame({
        "macd": dif,
        "macd_signal": dea,
        "macd_hist": hist
    })


def _add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the input DataFrame, which should contain
    open, high, low, close, volume, and amount columns.

    Added indicators include MA(5), MA(10), MA(20), RSI(14), and MACD.
    """
    df = df.copy()

    # Moving averages.
    df["ma_5"] = _calc_ma(df["close"], 5)
    df["ma_10"] = _calc_ma(df["close"], 10)
    df["ma_20"] = _calc_ma(df["close"], 20)

    # RSI.
    df["rsi_14"] = _calc_rsi(df["close"], period=14)

    # MACD.
    macd_df = _calc_macd(df["close"])
    df = pd.concat([df, macd_df], axis=1)

    print(
        f"[QuantData] Technical indicators added. Total rows: {len(df)}. "
        f"Some early rows may contain NaN values due to insufficient window length."
    )

    return df

# Construct time windows (N, T, F)

def _build_time_windows(
    df: pd.DataFrame,
    T: int,
    symbol: str,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Construct a window sequence with shape (N, T, F) from daily price
    and technical indicator data. Each date corresponds to the last day
    of the window.

    Returns:
        {
            "symbol": symbol,
            "X": np.ndarray,        # (N, T, F)
            "dates": np.ndarray,    # (N,)
            "features": List[str]   # Feature names
        }
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Automatically select feature columns:
    # numeric columns excluding symbol and date.
    if feature_cols is None:
        exclude = {"symbol", "date"}
        raw_feats = [
            c for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]
        feature_cols = [c for c in raw_feats if df[c].notna().any()]

    # Fill missing values. Early NaN values in technical indicators are
    # handled by forward and backward filling.
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # Check whether NaN values still remain.
    nan_counts = df[feature_cols].isna().sum()
    print(f"[QuantData] Number of NaN values after filling for {symbol}:")
    print(nan_counts)

    if len(df) < T:
        raise ValueError(
            f"{symbol}: Number of rows ({len(df)}) is smaller than T={T}. "
            f"Time windows cannot be constructed."
        )

    X_list = []
    dates_list = []

    for i in range(T - 1, len(df)):
        window = df.iloc[i - T + 1:i + 1]  # The most recent T trading days.

        # Safety check: skip the window if any NaN values remain.
        if window[feature_cols].isna().any().any():
            continue

        X_list.append(window[feature_cols].values)
        dates_list.append(window["date"].iloc[-1])

    if not X_list:
        raise ValueError(
            f"{symbol}: Data exist within the specified range, but all "
            f"T={T} day windows contain NaN values. No valid window can be "
            f"constructed."
        )

    X = np.stack(X_list, axis=0)  # (N, T, F)
    dates_arr = np.array(dates_list)

    print(
        f"[QuantData] Successfully constructed time windows for {symbol}: "
        f"X.shape = {X.shape} (N, T, F)."
    )

    return {
        "symbol": symbol,
        "X": X,
        "dates": dates_arr,
        "features": feature_cols,
    }

# Public interface: retrieve quantitative windows for one stock

def get_quant_windows_for_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    T: int,
    config: QuantConfig
) -> Dict[str, Any]:
    """
    Public interface.

    This function:
        1. Loads or downloads daily price data for the given symbol
           within [start_date, end_date].
        2. Computes technical indicators.
        3. Constructs time windows of length T.
        4. Returns X with shape (N, T, F) and the corresponding dates.

    Parameters
    ----------
    symbol:
        Stock code, e.g., "A002538".
    start_date, end_date:
        Date strings in the format "YYYY-MM-DD".
    T:
        Window length, e.g., 20.
    config:
        QuantConfig object.

    Returns
    -------
    {
        "symbol": ...,
        "X": np.ndarray,      # (N, T, F)
        "dates": np.ndarray,  # (N,)
        "features": List[str]
    }
    """
    # 1. Load or retrieve raw daily price data.
    df_price = _load_price_data(config, symbol, start_date, end_date)

    # 2. Add technical indicators.
    df_feat = _add_tech_indicators(df_price)

    # 3. Construct time windows.
    result = _build_time_windows(
        df_feat,
        T=T,
        symbol=symbol,
        feature_cols=None
    )
    return result

# Test entry point

if __name__ == "__main__":
    # Modify symbol, date range, or T here if needed.
    cfg = QuantConfig(
        prices_path="/home/guyh/hys/prices_multi.csv",
        use_akshare=True
    )

    sym = "A002538"
    start = "2023-01-01"
    end = "2024-03-01"
    T = 20

    res = get_quant_windows_for_symbol(sym, start, end, T, cfg)

    X = res["X"]
    dates = res["dates"]
    feats = res["features"]

    print(f"symbol: {res['symbol']}")
    print("X shape:", X.shape)
    print("First five window dates:", dates[:5])
    print("Number of features:", len(feats))
    print("Feature names:", feats)
