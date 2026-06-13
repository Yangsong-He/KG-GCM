# /home/guyh/hys/data_process.py
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm


# ===============================================================
# 1. 通用 .pt 读取
# ===============================================================

def _load_pt_file(path: str) -> Dict:
    """安全加载 .pt 文件，返回 dict。"""
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path} 不是 dict，而是 {type(data)}")
    return data


# ===============================================================
# 2. 聚合 daily_vecs 目录中的文本向量（按日期）
# ===============================================================

def _aggregate_daily_embeddings(
    base_dir: str,
    is_qa: bool,
    symbol: Optional[str] = None,
    embed_dim: int = 768,
) -> Dict[str, torch.Tensor]:
    """
    聚合 daily_vecs 目录下的向量，按日期做平均。
    - 对 ChinaDaily：所有文本按 date 聚合
    - 对 QA：只保留 symbol 匹配的问答，再按 date 聚合

    返回:
        date_str -> embedding (torch.FloatTensor[embed_dim])
    """
    if not os.path.isdir(base_dir):
        print(f"[TextAgg] 目录不存在: {base_dir}，返回空字典")
        return {}

    files = sorted(
        f for f in os.listdir(base_dir)
        if f.endswith(".pt")
    )

    print(
        f"[TextAgg] 从 {base_dir} 聚合文本，文件数 = {len(files)}, "
        f"is_qa={is_qa}, symbol={symbol}"
    )

    # 累积：date_str -> (sum_vec, count)
    sums: Dict[str, torch.Tensor] = {}
    cnts: Dict[str, int] = {}

    for fname in tqdm(files, desc="  [TextAgg] loading .pt", leave=False):
        fpath = os.path.join(base_dir, fname)
        try:
            d = _load_pt_file(fpath)
        except Exception as e:
            print(f"[TextAgg][WARN] 读取 {fpath} 失败: {e}")
            continue

        # 兼容 'date' 或 'dates'
        dates = d.get("date") or d.get("dates")
        if dates is None:
            print(f"[TextAgg][WARN] {fpath} 中找不到 'date' 或 'dates'，跳过")
            continue

        embs = d.get("embeddings")
        if embs is None:
            print(f"[TextAgg][WARN] {fpath} 中找不到 'embeddings'，跳过")
            continue

        if not torch.is_tensor(embs):
            embs = torch.as_tensor(embs, dtype=torch.float32)

        if embs.ndim != 2:
            print(f"[TextAgg][WARN] {fpath} embeddings 维度不是 2，而是 {embs.shape}，跳过")
            continue

        # 对 QA，需要按 symbol 过滤
        if is_qa:
            syms = d.get("symbols") or d.get("symbol")
            if syms is None:
                print(f"[TextAgg][WARN] QA 文件 {fpath} 没有 'symbols' 信息，跳过")
                continue
            if len(syms) != embs.shape[0]:
                print(
                    f"[TextAgg][WARN] QA 文件 {fpath} symbols 长度 {len(syms)} "
                    f"与 embeddings 行数 {embs.shape[0]} 不一致，跳过"
                )
                continue

        if len(dates) != embs.shape[0]:
            print(
                f"[TextAgg][WARN] {fpath} dates 长度 {len(dates)} != "
                f"embeddings 行数 {embs.shape[0]}，截断到最短"
            )
        n = min(len(dates), embs.shape[0])

        for i in range(n):
            # QA：按股票代码过滤
            if is_qa and symbol is not None:
                if syms[i] != symbol:
                    continue

            ds = str(dates[i])[:10]  # 仅保留 'YYYY-MM-DD'
            vec = embs[i]

            if ds not in sums:
                sums[ds] = vec.clone().detach().float()
                cnts[ds] = 1
            else:
                sums[ds] += vec
                cnts[ds] += 1

    # 计算每日平均向量
    daily: Dict[str, torch.Tensor] = {}
    for ds, svec in sums.items():
        c = cnts[ds]
        daily[ds] = svec / max(c, 1)

    print(
        f"[TextAgg] 聚合完成: 共 {len(daily)} 天有文本，"
        f"示例日期: {list(daily.keys())[:5]}"
    )
    return daily


# ===============================================================
# 3. 从 quant_res 抽取 X 和 日期（修复 or 导致的 ValueError）
# ===============================================================

def _extract_X_and_dates(
    quant_res,
    symbol: str
) -> Tuple[torch.Tensor, List[str]]:
    """
    尝试从 quant_res 中抽取:
        - X: torch.FloatTensor (N, T, F)
        - window_dates: 长度 N 的日期列表 (str: 'YYYY-MM-DD')

    兼容几种可能格式：
    1) dict，含 key: 'X' + 'window_dates' / 'dates' / 'window_end_dates'
    2) list/tuple: (X, dates, ...) 的形式
    """

    # ---------- 1) dict 情况 ----------
    if isinstance(quant_res, dict):
        # 避免使用 `or` 链接 numpy/tensor，逐个 key 检查
        X = None
        for k in ("X", "x", "windows"):
            if k in quant_res:
                X = quant_res[k]
                break

        dates = None
        for k in ("window_dates", "dates", "window_end_dates"):
            if k in quant_res:
                dates = quant_res[k]
                break

        if X is not None and dates is not None:
            if not torch.is_tensor(X):
                X = torch.as_tensor(X, dtype=torch.float32)
            date_list = [str(d)[:10] for d in dates]
            print(
                f"[DataProcess] 从 quant_res(dict) 抽取: "
                f"X.shape={tuple(X.shape)}, len(dates)={len(date_list)}"
            )
            return X, date_list
        else:
            print(
                f"[DataProcess][WARN] quant_res(dict) 中没有同时找到 X 和 dates，"
                f"keys={list(quant_res.keys())}"
            )

    # ---------- 2) list/tuple 情况 ----------
    if isinstance(quant_res, (list, tuple)) and len(quant_res) >= 2:
        X, dates = quant_res[0], quant_res[1]
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)
        date_list = [str(d)[:10] for d in dates]
        print(
            f"[DataProcess] 从 quant_res(list/tuple) 抽取: "
            f"X.shape={tuple(X.shape)}, len(dates)={len(date_list)}"
        )
        return X, date_list

    # ---------- 结构无法识别 ----------
    if isinstance(quant_res, dict):
        extra = list(quant_res.keys())
    elif isinstance(quant_res, (list, tuple)):
        extra = f"len={len(quant_res)}"
    else:
        extra = f"type={type(quant_res)}"

    raise KeyError(
        f"quant_res 的结构无法识别，symbol={symbol}，额外信息={extra}"
    )


# ===============================================================
# 4. 把 (N, T, F) 的量化窗口映射到 (N, embed_dim)
# ===============================================================

def _build_quant_vecs_from_windows(
    X: torch.Tensor,
    embed_dim: int = 768
) -> torch.Tensor:
    """
    把 (N, T, F) 的价格窗口拉平成 (N, embed_dim)，
    如果 T*F > embed_dim，则截断；
    如果 T*F < embed_dim，则右侧补 0。
    """
    if X.ndim != 3:
        raise ValueError(
            f"_build_quant_vecs_from_windows 期望 X 是 3 维 (N, T, F)，"
            f"实际是 {X.shape}"
        )

    N, T, F = X.shape
    flat = X.reshape(N, T * F)  # (N, D)
    D = flat.shape[1]

    if D == embed_dim:
        return flat

    if D > embed_dim:
        # 直接截前 embed_dim 维
        return flat[:, :embed_dim]

    # D < embed_dim: 右侧补 0
    pad = torch.zeros(N, embed_dim - D, dtype=flat.dtype)
    return torch.cat([flat, pad], dim=1)


# ===============================================================
# 5. 主函数：给一只股票，输出每天的 text_vec / quant_vec
# ===============================================================

def prepare_symbol_daily_data(
    symbol: str,
    quant_res,
    w_news: float = 0.5,
    w_qa: float = 0.5,
    embed_dim: int = 768,
    chinadaily_dir: str = "./daily_vecs/chinadaily",
    qa_dir: str = "./daily_vecs/qa",
):
    """
    综合入口函数：
    给定一只股票的量化窗口结果 quant_res，生成最终用于模型的：
        - dates: 每天的日期列表 (长度 N)
        - text_vecs: 新闻 + QA 加权后的文本向量 (N, embed_dim)
        - quant_vecs: 由价格窗口构造的量化向量 (N, embed_dim)

    返回:
        {
            "symbol": symbol,
            "dates": List[str],
            "text_vecs": torch.FloatTensor (N, embed_dim),
            "quant_vecs": torch.FloatTensor (N, embed_dim),
        }
    """
    # 1. 从 quant_res 抽取 X 和 window_dates
    X, window_dates = _extract_X_and_dates(quant_res, symbol)
    N = len(window_dates)

    # 2. 文本聚合（只做一次）
    news_daily = _aggregate_daily_embeddings(
        chinadaily_dir,
        is_qa=False,
        symbol=None,
        embed_dim=embed_dim,
    )
    qa_daily = _aggregate_daily_embeddings(
        qa_dir,
        is_qa=True,
        symbol=symbol,
        embed_dim=embed_dim,
    )

    # 3. 准备文本向量 & 量化向量
    text_vecs = torch.zeros(N, embed_dim, dtype=torch.float32)
    quant_vecs = _build_quant_vecs_from_windows(X, embed_dim=embed_dim)

    missing_news = 0
    missing_qa = 0

    for i, ds in enumerate(window_dates):
        # 文本部分：news + QA
        v_news = news_daily.get(ds)
        v_qa = qa_daily.get(ds)

        if v_news is None:
            missing_news += 1
            v_news = torch.zeros(embed_dim, dtype=torch.float32)
        if v_qa is None:
            missing_qa += 1
            v_qa = torch.zeros(embed_dim, dtype=torch.float32)

        text_vecs[i] = w_news * v_news + w_qa * v_qa

    print(
        f"[DataProcess] {symbol}: 最终对齐天数 N={N}, "
        f"text_vecs.shape={tuple(text_vecs.shape)}, "
        f"quant_vecs.shape={tuple(quant_vecs.shape)}"
    )
    print(
        f"[DataProcess] {symbol}: 缺少新闻天数={missing_news}, "
        f"缺少QA天数={missing_qa}"
    )

    return {
        "symbol": symbol,
        "dates": window_dates,   # 'YYYY-MM-DD' 字符串列表
        "text_vecs": text_vecs,
        "quant_vecs": quant_vecs,
    }
