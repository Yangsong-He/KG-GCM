# text_embedding.py
# -*- coding: utf-8 -*-

"""
使用 FinBERT 将文本转换为向量，并按块保存结果。

提供三个对外接口：
    - build_chinadaily_embeddings(...)
    - build_qa_embeddings(...)
    - build_all_embeddings(...)

仅负责「文本 -> 向量 + 保存」，不做按日聚合、不做预测。
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch import nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ======================= FinBERT 编码器 ======================= #

class FinBertTextEmbedder(nn.Module):
    """
    使用 FinBERT 将文本编码为 [CLS] 向量。

    参数
    ----
    model_name_or_path : str
        本地模型目录，例如：
        - "/home/guyh/hys/finbert-tone"
        - "/home/guyh/hys/finBERT"
    num_gpus : int
        现在仅用于记录/打印，实际强制只用一张 GPU（cuda:0），避免 NCCL 问题。
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_gpus: int = 1,
        max_length: int = 128,
        batch_size: int = 64,
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size

        # ✅ 确认本地模型目录存在
        if not os.path.isdir(self.model_name_or_path):
            raise FileNotFoundError(
                f"本地模型目录不存在: {self.model_name_or_path}\n"
                f"请确认 __init__.py 里传入的是本地路径（例如 /home/guyh/hys/finbert-tone），"
                f"而不是 'ProsusAI/finbert' 这样的仓库名。"
            )

        # ✅ 强制单卡：只用 cuda:0，完全不启用 DataParallel/NCCL
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.num_gpus = 1
            self.device = torch.device("cuda:0")
        else:
            self.num_gpus = 1
            self.device = torch.device("cpu")

        # ===== 离线加载：local_files_only=True，只从本地模型目录读取 =====
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=True,
        )

        base_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=True,
        )
        base_model.to(self.device)

        # ⚠️ 不再使用 DataParallel，避免 NCCL 错误
        self.bert = base_model
        self.bert.eval()

        cfg = self.bert.config
        self.hidden_size = cfg.hidden_size

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        将一组文本编码为 [CLS] 向量。

        返回 shape: (N, hidden_size) 的 CPU Tensor。
        """
        if len(texts) == 0:
            return torch.empty(0, self.hidden_size)

        bs = batch_size or self.batch_size
        ml = max_length or self.max_length

        all_embs = []
        total = len(texts)
        num_batches = (total + bs - 1) // bs

        for i in range(num_batches):
            start = i * bs
            end = min((i + 1) * bs, total)
            batch_texts = texts[start:end]

            if progress:
                print(f"    [encode_texts] batch {i+1}/{num_batches}, size={len(batch_texts)}")

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.bert(**enc)           # last_hidden_state: (B, L, H)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量 (B, H)

            all_embs.append(cls_emb.cpu())

        embeddings = torch.cat(all_embs, dim=0)  # (N, H)
        return embeddings


# ======================= 工具函数 ======================= #

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ======================= ChinaDaily 处理 ======================= #

def build_chinadaily_embeddings(
    model_name_or_path: str,
    data_dir: str = ".",
    output_dir: str = "daily_vecs",
    num_gpus: int = 1,
    batch_size: int = 64,
    max_length: int = 128,
    chunk_size: int = 20000,
    progress: bool = True,
) -> None:
    """
    将 ChinaDaily.csv 中每一条新闻 (title + content) 编码为向量并保存。

    输出：
        <output_dir>/chinadaily/chinadaily_partXXX.pt
      内含：
        {
            "row_ids": List[int],     # 原 CSV 行号（从 0 开始）
            "date": List[str],        # 对应日期（字符串）
            "embeddings": Tensor(n_rows, hidden_size)
        }
    """
    csv_path = os.path.join(data_dir, "ChinaDaily.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 ChinaDaily.csv: {csv_path}")

    save_dir = os.path.join(output_dir, "chinadaily")
    _ensure_dir(save_dir)

    if progress:
        print(f"[ChinaDaily] 读取文件: {csv_path}")

    embedder = FinBertTextEmbedder(
        model_name_or_path=model_name_or_path,
        num_gpus=num_gpus,
        max_length=max_length,
        batch_size=batch_size,
    )

    if progress:
        print(f"[ChinaDaily] 使用设备: {embedder.device}, num_gpus={embedder.num_gpus}")
        print(f"[ChinaDaily] hidden_size = {embedder.hidden_size}")

    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    global_row_offset = 0
    part_idx = 0

    for df in reader:
        n_rows = len(df)
        if n_rows == 0:
            continue

        row_ids = list(range(global_row_offset, global_row_offset + n_rows))
        global_row_offset += n_rows

        # 处理日期
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

        # 文本：title + content
        df["title"] = df["title"].fillna("")
        df["content"] = df["content"].fillna("")
        texts = (df["title"] + " " + df["content"]).tolist()

        if progress:
            print(f"[ChinaDaily] part {part_idx:03d}, 行数={n_rows}")

        embs = embedder.encode_texts(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            progress=progress,
        )  # (n_rows, hidden_size)

        out_path = os.path.join(save_dir, f"chinadaily_part{part_idx:03d}.pt")
        torch.save(
            {
                "row_ids": row_ids,
                "date": df["date"].tolist(),
                "embeddings": embs,
            },
            out_path,
        )

        if progress:
            print(f"[ChinaDaily] 已保存: {out_path}")

        part_idx += 1

    if progress:
        print(f"[ChinaDaily] 完成，总代码行数 = {global_row_offset}")


# ======================= QA 处理 ======================= #

def build_qa_embeddings(
    model_name_or_path: str,
    data_dir: str = ".",
    output_dir: str = "daily_vecs",
    num_gpus: int = 1,
    batch_size: int = 64,
    max_length: int = 128,
    chunk_size: int = 20000,
    progress: bool = True,
) -> None:
    """
    将 QA.csv 中每一条问答 (question + answer) 编码为向量并保存。

    输出：
        <output_dir>/qa/qa_partXXX.pt
      内含：
        {
            "row_ids": List[int],        # 原 CSV 行号（0 开始）
            "date": List[str],           # 以 answerDate 为主的日期（字符串）
            "symbol": List[str],         # 股票代码
            "embeddings": Tensor(n_rows, hidden_size)
        }
    """
    csv_path = os.path.join(data_dir, "QA.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 QA.csv: {csv_path}")

    save_dir = os.path.join(output_dir, "qa")
    _ensure_dir(save_dir)

    if progress:
        print(f"[QA] 读取文件: {csv_path}")

    embedder = FinBertTextEmbedder(
        model_name_or_path=model_name_or_path,
        num_gpus=num_gpus,
        max_length=max_length,
        batch_size=batch_size,
    )

    if progress:
        print(f"[QA] 使用设备: {embedder.device}, num_gpus={embedder.num_gpus}")
        print(f"[QA] hidden_size = {embedder.hidden_size}")

    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    global_row_offset = 0
    part_idx = 0

    for df in reader:
        n_rows = len(df)
        if n_rows == 0:
            continue

        row_ids = list(range(global_row_offset, global_row_offset + n_rows))
        global_row_offset += n_rows

        # 日期处理：以 answerDate 为准，fallback 用 questionDate
        qd = pd.to_datetime(df["questionDate"], errors="coerce")
        ad = pd.to_datetime(df["answerDate"], errors="coerce")
        date = ad.fillna(qd).dt.date.astype(str)

        # 文本：question + answer
        df["question"] = df["question"].fillna("")
        df["answer"] = df["answer"].fillna("")
        texts = (df["question"] + " " + df["answer"]).tolist()

        # symbol
        symbols = df["symbol"].astype(str).tolist()

        if progress:
            print(f"[QA] part {part_idx:03d}, 行数={n_rows}")

        embs = embedder.encode_texts(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            progress=progress,
        )  # (n_rows, hidden_size)

        out_path = os.path.join(save_dir, f"qa_part{part_idx:03d}.pt")
        torch.save(
            {
                "row_ids": row_ids,
                "date": list(date),
                "symbol": symbols,
                "embeddings": embs,
            },
            out_path,
        )

        if progress:
            print(f"[QA] 已保存: {out_path}")

        part_idx += 1

    if progress:
        print(f"[QA] 完成，总代码行数 = {global_row_offset}")


# ======================= 统一入口 ======================= #

def build_all_embeddings(
    model_name_or_path: str,
    data_dir: str = ".",
    output_dir: str = "daily_vecs",
    num_gpus: int = 1,
    batch_size: int = 64,
    max_length: int = 128,
    chunk_size: int = 20000,
    progress: bool = True,
) -> None:
    """
    同时构建 ChinaDaily 与 QA 的向量。
    """
    build_chinadaily_embeddings(
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        output_dir=output_dir,
        num_gpus=num_gpus,
        batch_size=batch_size,
        max_length=max_length,
        chunk_size=chunk_size,
        progress=progress,
    )

    build_qa_embeddings(
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        output_dir=output_dir,
        num_gpus=num_gpus,
        batch_size=batch_size,
        max_length=max_length,
        chunk_size=chunk_size,
        progress=progress,
    )
