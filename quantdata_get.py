# quantdata_get.py
# 功能：获取单只股票在指定时间区间内的日线数据，计算技术指标，
#       并把过去 T 天的数据拼成一个 (N, T, F) 的窗口序列，同时保留对应日期。
# 说明：内部用 AkShare 的 stock_zh_a_hist（含成交量 volume，成交额 amount 若全 NaN 则用 close*volume 近似），
#       对外接口不变。

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


# =====================
# 配置
# =====================

@dataclass
class QuantConfig:
    """
    量化数据的基本配置
    """
    # 存放所有股票价格数据的 CSV
    prices_path: str = "/home/guyh/hys/prices_multi.csv"
    # 若本地没有数据，是否用 AkShare 在线获取
    use_akshare: bool = True


# =====================
# AkShare 数据获取部分
# =====================

def _fetch_and_append_symbol_data(config: QuantConfig,
                                  symbol: str,
                                  start_date: str,
                                  end_date: str) -> None:
    """
    使用 AkShare 的 stock_zh_a_hist 获取完整 K 线数据（含成交额 amount），
    并追加写入 prices_multi.csv。

    参数
    ----
    symbol: 例如 "A002538" 这种你自己的代码格式
    start_date, end_date: "YYYY-MM-DD" 字符串
    """
    print(f"[AkShare] 使用 stock_zh_a_hist 拉取 {symbol} 的日线数据...")

    try:
        import akshare as ak
    except ImportError:
        print("[AkShare] 未安装 akshare，请先在环境中安装：pip install akshare")
        return

    # symbol 可能为 A002538 -> 转成 002538
    raw_sym = symbol.replace("A", "")

    # 时间格式转换：2023-01-01 -> 20230101
    sd = start_date.replace("-", "")
    ed = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=raw_sym,
            period="daily",
            start_date=sd,
            end_date=ed,
            adjust="qfq"  # 前复权，可按需求改
        )
    except Exception as e:
        print(f"[AkShare] 获取 {symbol} 时出错：{e}")
        return

    if df is None or df.empty:
        print(f"[AkShare] 没有拉到 {symbol} 在 {start_date}~{end_date} 的数据。")
        return

    # 列名重命名（根据当前 akshare 版本，字段名可能略有变化，可打印 df.columns 看）
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

    # 增加 symbol 列，格式对齐
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        df["amount"] = np.nan

    # 只保留核心列
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]]

    path = config.prices_path

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_old["date"] = pd.to_datetime(df_old["date"], format="mixed", errors="coerce")
        df_all = pd.concat([df_old, df], ignore_index=True)
    else:
        df_all = df

    # 保存
    df_all.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[AkShare] 已将 {symbol} 的数据写入 {path}，新增 {len(df)} 行。")


def _load_price_data(config: QuantConfig,
                     symbol: str,
                     start_date: str,
                     end_date: str) -> pd.DataFrame:
    """
    从 prices_multi.csv 中读取某个 symbol 在指定时间区间的日线数据。
    若没有且允许 use_akshare，则调用 AkShare 获取一次。
    """
    path = config.prices_path

    if not os.path.exists(path):
        print(f"[QuantData] {path} 不存在。")
        if config.use_akshare:
            # 尝试从 AkShare 拉一次数据并写入
            _fetch_and_append_symbol_data(config, symbol, start_date, end_date)
        else:
            raise FileNotFoundError(f"{path} 不存在且 use_akshare=False。")

    if not os.path.exists(path):
        raise FileNotFoundError(f"[QuantData] 在尝试 AkShare 后仍未找到 {path}。")

    df_all = pd.read_csv(path)
    # 兼容 "YYYY-MM-DD" 和 "YYYY-MM-DD HH:MM:SS"
    df_all["date"] = pd.to_datetime(df_all["date"], format="mixed", errors="coerce")

    # 过滤 symbol & 区间
    mask_symbol = (df_all["symbol"] == symbol)
    mask_date = (df_all["date"] >= pd.to_datetime(start_date)) & (
        df_all["date"] <= pd.to_datetime(end_date)
    )
    df = df_all.loc[mask_symbol & mask_date].copy()

    if df.empty and config.use_akshare:
        # 再尝试拉一遍
        _fetch_and_append_symbol_data(config, symbol, start_date, end_date)
        df_all = pd.read_csv(path)
        df_all["date"] = pd.to_datetime(df_all["date"], format="mixed", errors="coerce")
        mask_symbol = (df_all["symbol"] == symbol)
        mask_date = (df_all["date"] >= pd.to_datetime(start_date)) & (
            df_all["date"] <= pd.to_datetime(end_date)
        )
        df = df_all.loc[mask_symbol & mask_date].copy()

    if df.empty:
        raise ValueError(
            f"在区间 {start_date} ~ {end_date} 中，找不到 symbol={symbol} 的数据"
        )

    # 确保数值列为数值
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 如果 amount 全 NaN：用 close * volume 近似填充
    if "amount" in df.columns:
        if df["amount"].isna().all():
            print(f"[QuantData] {symbol} 的 amount 列全为 NaN，使用 close*volume 近似构造成交额。")
            df["amount"] = df["close"] * df["volume"]

    df = df.sort_values("date").reset_index(drop=True)
    print(f"[QuantData] 读取到 {symbol} 原始日线数据行数: {len(df)}")
    print(df.head())

    return df


# =====================
# 技术指标计算
# =====================

def _calc_ma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均线 MA"""
    return series.rolling(window=window, min_periods=1).mean()


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数 RSI"""
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


def _calc_macd(close: pd.Series,
               fast: int = 12,
               slow: int = 26,
               signal: int = 9) -> pd.DataFrame:
    """MACD 指标：DIF, DEA, HIST"""
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
    对输入 df（包含 open/high/low/close/volume/amount）添加技术指标：
    MA(5,10,20)、RSI(14)、MACD。
    """
    df = df.copy()

    # 移动平均线
    df["ma_5"] = _calc_ma(df["close"], 5)
    df["ma_10"] = _calc_ma(df["close"], 10)
    df["ma_20"] = _calc_ma(df["close"], 20)

    # RSI
    df["rsi_14"] = _calc_rsi(df["close"], period=14)

    # MACD
    macd_df = _calc_macd(df["close"])
    df = pd.concat([df, macd_df], axis=1)

    print(f"[QuantData] 加指标后总行数: {len(df)}，其中可能有部分行某些指标为 NaN（前期不足窗口）")
    return df


# =====================
# 构造时间窗口 (N, T, F)
# =====================

def _build_time_windows(df: pd.DataFrame,
                        T: int,
                        symbol: str,
                        feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    把日线+指标数据构造成 (N, T, F) 的窗口序列，并返回对应日期（窗口最后一天）。

    返回：
        {
            "symbol": symbol,
            "X": np.ndarray,    # (N, T, F)
            "dates": np.ndarray # (N,)
            "features": List[str]  # 特征名列表
        }
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 自动选特征列：数值列且不是 symbol/date
    if feature_cols is None:
        exclude = {"symbol", "date"}
        raw_feats = [
            c for c in df.columns
            if (c not in exclude and np.issubdtype(df[c].dtype, np.number))
        ]
        feature_cols = [c for c in raw_feats if df[c].notna().any()]

    # 缺失值填充（技术指标前期 NaN 用前向/后向填充）
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # 再次检查是否仍有 NaN
    nan_counts = df[feature_cols].isna().sum()
    print(f"[QuantData] {symbol} 特征列 NaN 数量（填充后）:")
    print(nan_counts)

    if len(df) < T:
        raise ValueError(f"{symbol}: 数据行数 {len(df)} < T={T}，无法构造窗口")

    X_list = []
    dates_list = []

    for i in range(T - 1, len(df)):
        window = df.iloc[i - T + 1:i + 1]  # 最近 T 天

        # 理论上已经填充完 NaN，这里加一道保险
        if window[feature_cols].isna().any().any():
            continue

        X_list.append(window[feature_cols].values)
        dates_list.append(window["date"].iloc[-1])

    if not X_list:
        raise ValueError(
            f"{symbol}: 在区间内虽然有数据，但所有 (T={T}) 日窗口都含 NaN，无法构造有效窗口。"
        )

    X = np.stack(X_list, axis=0)  # (N, T, F)
    dates_arr = np.array(dates_list)

    print(
        f"[QuantData] 为 {symbol} 构造窗口成功: X.shape = {X.shape}（N, T, F）"
    )

    return {
        "symbol": symbol,
        "X": X,
        "dates": dates_arr,
        "features": feature_cols,
    }


# =====================
# 对外主函数：获取某股票的量价窗口
# =====================

def get_quant_windows_for_symbol(symbol: str,
                                 start_date: str,
                                 end_date: str,
                                 T: int,
                                 config: QuantConfig) -> Dict[str, Any]:
    """
    对外主接口：
        1. 读取（或下载）某个 symbol 在区间 [start_date, end_date] 的日线数据；
        2. 计算技术指标；
        3. 构造长度为 T 的时间窗口；
        4. 返回 (N,T,F) 以及每个窗口对应的日期。

    参数
    ----
    symbol: 例如 "A002538"
    start_date, end_date: "YYYY-MM-DD"
    T: 窗口长度（例如 20）
    config: QuantConfig

    返回
    ----
    {
        "symbol": ...,
        "X": np.ndarray,      # (N, T, F)
        "dates": np.ndarray,  # (N,)
        "features": List[str]
    }
    """
    # 1. 读取或获取原始日线
    df_price = _load_price_data(config, symbol, start_date, end_date)

    # 2. 添加技术指标
    df_feat = _add_tech_indicators(df_price)

    # 3. 构造时间窗口
    result = _build_time_windows(df_feat, T=T, symbol=symbol, feature_cols=None)
    return result


# =====================
# 测试入口（直接运行本文件时）
# =====================

if __name__ == "__main__":
    # 你可以在这里改 symbol / 时间区间 / T
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
    print("前 5 个窗口日期（每个窗口最后一天）:", dates[:5])
    print("特征个数:", len(feats))
    print("特征名:", feats)
