import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

# 1. General .pt file loading

def _load_pt_file(path: str) -> Dict:
    """Safely load a .pt file and return a dictionary."""
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"{path} is expected to be a dict, but got {type(data)}")
    return data

# 2. Aggregate text embeddings in daily_vecs by date

def _aggregate_daily_embeddings(
    base_dir: str,
    is_qa: bool,
    symbol: Optional[str] = None,
    embed_dim: int = 768,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate embeddings under the daily_vecs directory by date.
    - For ChinaDaily data, all text entries are aggregated by date.
    - For QA data, only question-answer entries matching the given symbol
      are retained and then aggregated by date.

    Returns:
        date_str -> embedding (torch.FloatTensor[embed_dim])
    """
    if not os.path.isdir(base_dir):
        print(
            f"[TextAgg] Directory does not exist: {base_dir}. "
            f"Returning an empty dictionary."
        )
        return {}

    files = sorted(
        f for f in os.listdir(base_dir)
        if f.endswith(".pt")
    )

    print(
        f"[TextAgg] Aggregating text embeddings from {base_dir}. "
        f"Number of files = {len(files)}, "
        f"is_qa={is_qa}, symbol={symbol}"
    )

    # Accumulators: date_str -> (sum_vec, count)
    sums: Dict[str, torch.Tensor] = {}
    cnts: Dict[str, int] = {}

    for fname in tqdm(files, desc="  [TextAgg] loading .pt", leave=False):
        fpath = os.path.join(base_dir, fname)
        try:
            d = _load_pt_file(fpath)
        except Exception as e:
            print(f"[TextAgg][WARN] Failed to load {fpath}. Reason: {e}")
            continue

        # Support both 'date' and 'dates'.
        dates = d.get("date") or d.get("dates")
        if dates is None:
            print(
                f"[TextAgg][WARN] Neither 'date' nor 'dates' was found "
                f"in {fpath}. Skipping."
            )
            continue

        embs = d.get("embeddings")
        if embs is None:
            print(
                f"[TextAgg][WARN] 'embeddings' was not found in {fpath}. "
                f"Skipping."
            )
            continue

        if not torch.is_tensor(embs):
            embs = torch.as_tensor(embs, dtype=torch.float32)

        if embs.ndim != 2:
            print(
                f"[TextAgg][WARN] Expected a 2D embedding matrix, "
                f"but got shape {embs.shape} in {fpath}. Skipping."
            )
            continue

        # For QA data, filter entries by stock symbol.
        if is_qa:
            syms = d.get("symbols") or d.get("symbol")
            if syms is None:
                print(
                    f"[TextAgg][WARN] No 'symbols' field found in QA file "
                    f"{fpath}. Skipping."
                )
                continue

            if len(syms) != embs.shape[0]:
                print(
                    f"[TextAgg][WARN] Symbol count ({len(syms)}) does not "
                    f"match embedding rows ({embs.shape[0]}) in QA file "
                    f"{fpath}. Skipping."
                )
                continue

        if len(dates) != embs.shape[0]:
            print(
                f"[TextAgg][WARN] Number of dates ({len(dates)}) does not "
                f"match embedding rows ({embs.shape[0]}) in {fpath}. "
                f"Truncating to the shorter length."
            )

        n = min(len(dates), embs.shape[0])

        for i in range(n):
            # For QA data, retain only entries matching the given stock symbol.
            if is_qa and symbol is not None:
                if syms[i] != symbol:
                    continue

            ds = str(dates[i])[:10]  # Retain only the 'YYYY-MM-DD' part.
            vec = embs[i]

            if ds not in sums:
                sums[ds] = vec.clone().detach().float()
                cnts[ds] = 1
            else:
                sums[ds] += vec
                cnts[ds] += 1

    # Compute the daily average embedding.
    daily: Dict[str, torch.Tensor] = {}
    for ds, svec in sums.items():
        c = cnts[ds]
        daily[ds] = svec / max(c, 1)

    print(
        f"[TextAgg] Aggregation completed. "
        f"Text data available for {len(daily)} days. "
        f"Example dates: {list(daily.keys())[:5]}"
    )
    return daily

# 3. Extract X and dates from quant_res

def _extract_X_and_dates(
    quant_res,
    symbol: str
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract the following fields from quant_res:
        - X: torch.FloatTensor (N, T, F)
        - window_dates: a date list of length N (str: 'YYYY-MM-DD')

    Supported formats:
    1) dict containing keys such as 'X' and
       'window_dates' / 'dates' / 'window_end_dates'
    2) list/tuple in the form of (X, dates, ...)
    """

    # ---------- 1) dict format ----------
    if isinstance(quant_res, dict):
        # Check each key explicitly to avoid ValueError caused by `or`
        # operations on NumPy arrays or tensors.
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
                f"[DataProcess] Extracted from quant_res (dict): "
                f"X.shape={tuple(X.shape)}, len(dates)={len(date_list)}"
            )
            return X, date_list

        print(
            f"[DataProcess][WARN] Failed to locate both X and dates "
            f"in quant_res (dict). Keys: {list(quant_res.keys())}"
        )

    # ---------- 2) list/tuple format ----------
    if isinstance(quant_res, (list, tuple)) and len(quant_res) >= 2:
        X, dates = quant_res[0], quant_res[1]
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.float32)
        date_list = [str(d)[:10] for d in dates]
        print(
            f"[DataProcess] Extracted from quant_res (list/tuple): "
            f"X.shape={tuple(X.shape)}, len(dates)={len(date_list)}"
        )
        return X, date_list

    # ---------- Unrecognized structure ----------
    if isinstance(quant_res, dict):
        extra = list(quant_res.keys())
    elif isinstance(quant_res, (list, tuple)):
        extra = f"len={len(quant_res)}"
    else:
        extra = f"type={type(quant_res)}"

    raise KeyError(
        f"Unsupported quant_res structure for symbol={symbol}. "
        f"Additional information: {extra}"
    )


# 4. Map quantitative windows from (N, T, F) to (N, embed_dim)

def _build_quant_vecs_from_windows(
    X: torch.Tensor,
    embed_dim: int = 768
) -> torch.Tensor:
    """
    Flatten price windows from (N, T, F) to (N, embed_dim).
    If T * F > embed_dim, the flattened vector is truncated.
    If T * F < embed_dim, zero padding is added to the right.
    """
    if X.ndim != 3:
        raise ValueError(
            f"_build_quant_vecs_from_windows expects a 3D tensor "
            f"with shape (N, T, F), but got {X.shape}"
        )

    N, T, F = X.shape
    flat = X.reshape(N, T * F)  # (N, D)
    D = flat.shape[1]

    if D == embed_dim:
        return flat

    if D > embed_dim:
        # Retain the first embed_dim dimensions.
        return flat[:, :embed_dim]

    # D < embed_dim: add zero padding to the right.
    pad = torch.zeros(N, embed_dim - D, dtype=flat.dtype)
    return torch.cat([flat, pad], dim=1)

# 5. Main function: generate daily text and quantitative vectors for a given stock symbol

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
    Main entry point.

    Given quantitative window results for a stock symbol, this function
    generates the final model inputs:
        - dates: daily date list of length N
        - text_vecs: weighted news and QA text embeddings (N, embed_dim)
        - quant_vecs: quantitative vectors constructed from price windows
          (N, embed_dim)

    Returns:
        {
            "symbol": symbol,
            "dates": List[str],
            "text_vecs": torch.FloatTensor (N, embed_dim),
            "quant_vecs": torch.FloatTensor (N, embed_dim),
        }
    """
    # 1. Extract X and window dates from quant_res.
    X, window_dates = _extract_X_and_dates(quant_res, symbol)
    N = len(window_dates)

    # 2. Aggregate text embeddings.
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

    # 3. Prepare text vectors and quantitative vectors.
    text_vecs = torch.zeros(N, embed_dim, dtype=torch.float32)
    quant_vecs = _build_quant_vecs_from_windows(X, embed_dim=embed_dim)

    missing_news = 0
    missing_qa = 0

    for i, ds in enumerate(window_dates):
        # Text component: news + QA.
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
        f"[DataProcess] {symbol}: Aligned trading days = {N}, "
        f"text_vecs.shape={tuple(text_vecs.shape)}, "
        f"quant_vecs.shape={tuple(quant_vecs.shape)}"
    )
    print(
        f"[DataProcess] {symbol}: Days without news = {missing_news}, "
        f"days without QA = {missing_qa}"
    )

    return {
        "symbol": symbol,
        "dates": window_dates,  # List of date strings in the format 'YYYY-MM-DD'.
        "text_vecs": text_vecs,
        "quant_vecs": quant_vecs,
    }
