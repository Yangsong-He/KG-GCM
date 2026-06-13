# /home/guyh/hys/model.py
# 论文风格（两模态版）：Decouple -> Transformer Fusion -> Classifier
# 保持你的 pipeline：读取 daily_final，构造 label，按时间切分 train/val/test，训练 n_epochs，保存模型与预测结果

import os
import sys
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DAILY_FINAL_DIR = os.path.join(CURRENT_DIR, "daily_final")
PRICES_CSV = os.path.join(CURRENT_DIR, "prices_multi.csv")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. 读取 daily_final 里的向量
# ---------------------------------------------------------------------
def load_daily_vectors(symbol: str, base_dir: str = DAILY_FINAL_DIR) -> Dict:
    path = os.path.join(base_dir, f"{symbol}_daily.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 daily_final 文件: {path}")

    data = torch.load(path, map_location="cpu")
    if "dates" not in data or "text_vecs" not in data or "quant_vecs" not in data:
        raise KeyError(f"{path} 中缺少必要键 'dates' / 'text_vecs' / 'quant_vecs'")

    # 标准化日期为 pandas.Timestamp
    dates = [pd.to_datetime(d) for d in data["dates"]]
    data["dates"] = dates

    # 确保 tensor 类型
    if not torch.is_tensor(data["text_vecs"]):
        data["text_vecs"] = torch.as_tensor(data["text_vecs"], dtype=torch.float32)
    if not torch.is_tensor(data["quant_vecs"]):
        data["quant_vecs"] = torch.as_tensor(data["quant_vecs"], dtype=torch.float32)

    return data

# ---------------------------------------------------------------------
# 2. 从 prices_multi.csv 里构造涨跌标签（下一交易日收盘价 > 当日收盘价）
# ---------------------------------------------------------------------
def load_price_series(symbol: str, prices_csv: str = PRICES_CSV) -> pd.DataFrame:
    if not os.path.exists(prices_csv):
        raise FileNotFoundError(f"找不到价格文件: {prices_csv}")

    df = pd.read_csv(prices_csv)
    if "symbol" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        raise KeyError("prices_multi.csv 必须包含 'symbol', 'date', 'close' 三列")

    df = df[df["symbol"] == symbol].copy()
    if df.empty:
        raise ValueError(f"prices_multi.csv 中没有 symbol={symbol} 的记录")

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]

def build_labels_from_prices(price_df: pd.DataFrame) -> Dict[pd.Timestamp, int]:
    dates = price_df["date"].tolist()
    closes = price_df["close"].to_numpy()

    label_by_date: Dict[pd.Timestamp, int] = {}
    for i in range(len(dates) - 1):
        d = pd.to_datetime(dates[i]).normalize()
        c_now = closes[i]
        c_next = closes[i + 1]
        label_by_date[d] = 1 if c_next > c_now else 0

    return label_by_date

# ---------------------------------------------------------------------
# 3. 构造两模态输入：X_text, X_quant, y
# ---------------------------------------------------------------------
def build_inputs_y_for_symbol(
    symbol: str,
    use_text: bool = True,
    use_quant: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, List[pd.Timestamp]]:
    daily_data = load_daily_vectors(symbol)
    dates_vec = daily_data["dates"]
    text_vecs = daily_data["text_vecs"]    # (N, 768)
    quant_vecs = daily_data["quant_vecs"]  # (N, 768)

    if not use_text and not use_quant:
        raise ValueError("至少要使用 text 或 quant 中的一种特征")

    # 价格标签
    price_df = load_price_series(symbol)
    label_by_date = build_labels_from_prices(price_df)

    X_text_list = []
    X_quant_list = []
    y_list = []
    used_dates: List[pd.Timestamp] = []

    for idx, d in enumerate(dates_vec):
        d_norm = pd.to_datetime(d).normalize()
        if d_norm in label_by_date:
            if use_text:
                X_text_list.append(text_vecs[idx].unsqueeze(0))
            if use_quant:
                X_quant_list.append(quant_vecs[idx].unsqueeze(0))
            y_list.append(label_by_date[d_norm])
            used_dates.append(d_norm)

    if len(y_list) == 0:
        raise ValueError(f"{symbol}: 没有和价格标签对齐上的样本，请检查日期对齐")

    X_text = torch.cat(X_text_list, dim=0) if use_text else None
    X_quant = torch.cat(X_quant_list, dim=0) if use_quant else None
    y = torch.tensor(y_list, dtype=torch.long)

    # 你只做两模态，所以这里要求两者都存在（更符合论文 fusion）
    if X_text is None or X_quant is None:
        raise ValueError("两模态 fusion 需要同时 use_text=True 且 use_quant=True")

    return X_text, X_quant, y, used_dates

# ---------------------------------------------------------------------
# 4. 论文式：Decouple -> Transformer Fusion -> Classifier（两 token）
# ---------------------------------------------------------------------
class TwoModalFusionTransformer(nn.Module):
    def __init__(
        self,
        dim_text: int = 768,
        dim_quant: int = 768,
        d_fuse: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Decouple: 各模态投影到同一维度
        self.proj_text = nn.Sequential(
            nn.Linear(dim_text, d_fuse),
            nn.LayerNorm(d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_quant = nn.Sequential(
            nn.Linear(dim_quant, d_fuse),
            nn.LayerNorm(d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion: token self-attention
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_fuse,
            nhead=n_heads,
            dim_feedforward=4 * d_fuse,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Classifier: concat 两个 token
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_fuse),
            nn.Linear(2 * d_fuse, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x_text: torch.Tensor, x_quant: torch.Tensor) -> torch.Tensor:
        # x_text, x_quant: (B, 768)
        z_text = self.proj_text(x_text)   # (B, d_fuse)
        z_quant = self.proj_quant(x_quant)

        tok = torch.stack([z_text, z_quant], dim=1)  # (B, 2, d_fuse)
        tok_fused = self.encoder(tok)                # (B, 2, d_fuse)

        feat = tok_fused.reshape(tok_fused.size(0), -1)  # (B, 2*d_fuse)
        logits = self.head(feat)                         # (B, 2)
        return logits

# ---------------------------------------------------------------------
# 5. 数据划分 + 标准化（按模态分别标准化）
# ---------------------------------------------------------------------
def split_train_val_test(
    N: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, N)
    return idx_train, idx_val, idx_test

def standardize_one_by_train(
    X_train: torch.FloatTensor,
    X_val: torch.FloatTensor,
    X_test: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std

# ---------------------------------------------------------------------
# 6. 训练函数（固定 n_epochs，但记录 best_val_acc 的 best_state）
# ---------------------------------------------------------------------
def train_model_for_symbol(
    symbol: str,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = None,
    d_fuse: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> Dict:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n[Model] 构造两模态特征与标签: {symbol} ...")
    X_text, X_quant, y, used_dates = build_inputs_y_for_symbol(
        symbol=symbol,
        use_text=True,
        use_quant=True,
    )
    N = y.shape[0]
    print(f"[Model] {symbol}: X_text={tuple(X_text.shape)}, X_quant={tuple(X_quant.shape)}, y={tuple(y.shape)}, N={N}")

    # split（按时间顺序）
    idx_train, idx_val, idx_test = split_train_val_test(N)
    Xt_tr, Xt_va, Xt_te = X_text[idx_train], X_text[idx_val], X_text[idx_test]
    Xq_tr, Xq_va, Xq_te = X_quant[idx_train], X_quant[idx_val], X_quant[idx_test]
    y_tr, y_va, y_te = y[idx_train], y[idx_val], y[idx_test]

    print(f"[Model] 数据划分: train={y_tr.shape[0]}, val={y_va.shape[0]}, test={y_te.shape[0]}")

    # 标准化（分别对 text/quant）
    Xt_tr, Xt_va, Xt_te = standardize_one_by_train(Xt_tr, Xt_va, Xt_te)
    Xq_tr, Xq_va, Xq_te = standardize_one_by_train(Xq_tr, Xq_va, Xq_te)

    # DataLoader（两输入 + label）
    train_loader = DataLoader(TensorDataset(Xt_tr, Xq_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xt_va, Xq_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xt_te, Xq_te, y_te), batch_size=batch_size, shuffle=False)

    # 模型
    model = TwoModalFusionTransformer(
        dim_text=X_text.shape[1],
        dim_quant=X_quant.shape[1],
        d_fuse=d_fuse,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # ---- train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb_t, xb_q, yb in train_loader:
            xb_t = xb_t.to(device)
            xb_q = xb_q.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb_t, xb_q)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ---- val
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb_t, xb_q, yb in val_loader:
                xb_t = xb_t.to(device)
                xb_q = xb_q.to(device)
                yb = yb.to(device)
                logits = model(xb_t, xb_q)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * yb.size(0)

                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 加载 best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test predict
    model.eval()
    all_logits = []
    all_y_true = []

    with torch.no_grad():
        for xb_t, xb_q, yb in test_loader:
            xb_t = xb_t.to(device)
            xb_q = xb_q.to(device)
            logits = model(xb_t, xb_q)
            all_logits.append(logits.cpu())
            all_y_true.append(yb.cpu())

    logits_test = torch.cat(all_logits, dim=0)  # (N_test, 2)
    y_true = torch.cat(all_y_true, dim=0)       # (N_test,)

    probs = torch.softmax(logits_test, dim=1)
    y_pred = probs.argmax(dim=1)
    y_proba_pos = probs[:, 1]

    test_dates = [used_dates[i] for i in range(idx_test.start, idx_test.stop)]

    result_dict = {
        "dates": test_dates,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba_pos,
        "symbol": symbol,
        "best_val_acc": float(best_val_acc),
    }

    # save model
    model_path = os.path.join(MODELS_DIR, f"{symbol}_clf.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Model] ✅ 模型已保存: {model_path}")

    # save results
    result_path = os.path.join(RESULTS_DIR, f"{symbol}_results.pt")
    torch.save(result_dict, result_path)
    print(f"[Model] ✅ 预测结果已保存: {result_path}")

    return {
        "model_path": model_path,
        "result_path": result_path,
        "best_val_acc": best_val_acc,
        "n_test": len(test_dates),
    }

# ---------------------------------------------------------------------
# 7. 命令行入口：逐个训练
# ---------------------------------------------------------------------
SYMBOLS = ["A601088", "A600938", "A601857", "A600028", "A600019", "A600941", "A601985", "A600900", "A003816"]


def main():
    for sym in SYMBOLS:
        print("\n" + "=" * 70)
        print(f"*** 训练模型: {sym} ***")
        info = train_model_for_symbol(
            symbol=sym,
            batch_size=32,
            n_epochs=100,
            lr=1e-4,
            weight_decay=1e-4,
            d_fuse=256,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
        )
        print(f"[Summary] {sym}: best_val_acc={info['best_val_acc']:.4f}, n_test={info['n_test']}")

if __name__ == "__main__":
    main()
