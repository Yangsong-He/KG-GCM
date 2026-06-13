"""
Convert text into vector representations using FinBERT and save the results
by chunks.

This module provides three public interfaces:
    - build_chinadaily_embeddings(...)
    - build_qa_embeddings(...)
    - build_all_embeddings(...)

It only handles text-to-vector encoding and saving. It does not perform
daily aggregation or prediction.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch import nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ======================= FinBERT encoder ======================= #

class FinBertTextEmbedder(nn.Module):
    """
    Encode text into [CLS] vectors using FinBERT.

    Parameters
    ----------
    model_name_or_path : str
        Local model directory, for example:
        - "/home/guyh/hys/finbert-tone"
        - "/home/guyh/hys/finBERT"
    num_gpus : int
        This parameter is only used for logging. The implementation
        enforces single-GPU execution on cuda:0 to avoid NCCL issues.
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

        # Verify that the local model directory exists.
        if not os.path.isdir(self.model_name_or_path):
            raise FileNotFoundError(
                f"Local model directory does not exist: {self.model_name_or_path}\n"
                f"Please make sure that the path passed in __init__.py is a local path "
                f"such as /home/guyh/hys/finbert-tone, rather than a repository name "
                f"such as 'ProsusAI/finbert'."
            )

        # Enforce single-GPU execution on cuda:0 and avoid DataParallel/NCCL.
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.num_gpus = 1
            self.device = torch.device("cuda:0")
        else:
            self.num_gpus = 1
            self.device = torch.device("cpu")

        # Offline loading: local_files_only=True loads files only from
        # the local model directory.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=True,
        )

        base_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=True,
        )
        base_model.to(self.device)

        # DataParallel is intentionally disabled to avoid NCCL errors.
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
        Encode a list of texts into [CLS] vectors.

        Returns a CPU tensor with shape (N, hidden_size).
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
                print(f"    [encode_texts] batch {i + 1}/{num_batches}, size={len(batch_texts)}")

            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=ml,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.bert(**enc)  # last_hidden_state: (B, L, H)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] vectors: (B, H)

            all_embs.append(cls_emb.cpu())

        embeddings = torch.cat(all_embs, dim=0)  # (N, H)
        return embeddings


# ======================= Utility functions ======================= #

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ======================= ChinaDaily processing ======================= #

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
    Encode each news item in ChinaDaily.csv, using title + content, and save
    the resulting embeddings.

    Output:
        <output_dir>/chinadaily/chinadaily_partXXX.pt

    The saved file contains:
        {
            "row_ids": List[int],     # Original CSV row indices starting from 0
            "date": List[str],        # Corresponding dates as strings
            "embeddings": Tensor(n_rows, hidden_size)
        }
    """
    csv_path = os.path.join(data_dir, "ChinaDaily.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ChinaDaily.csv not found: {csv_path}")

    save_dir = os.path.join(output_dir, "chinadaily")
    _ensure_dir(save_dir)

    if progress:
        print(f"[ChinaDaily] Loading file: {csv_path}")

    embedder = FinBertTextEmbedder(
        model_name_or_path=model_name_or_path,
        num_gpus=num_gpus,
        max_length=max_length,
        batch_size=batch_size,
    )

    if progress:
        print(f"[ChinaDaily] Device: {embedder.device}, num_gpus={embedder.num_gpus}")
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

        # Date processing.
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

        # Text field: title + content.
        df["title"] = df["title"].fillna("")
        df["content"] = df["content"].fillna("")
        texts = (df["title"] + " " + df["content"]).tolist()

        if progress:
            print(f"[ChinaDaily] part {part_idx:03d}, rows={n_rows}")

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
            print(f"[ChinaDaily] Saved to: {out_path}")

        part_idx += 1

    if progress:
        print(f"[ChinaDaily] Completed. Total rows processed = {global_row_offset}")


# ======================= QA processing ======================= #

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
    Encode each question-answer pair in QA.csv, using question + answer, and
    save the resulting embeddings.

    Output:
        <output_dir>/qa/qa_partXXX.pt

    The saved file contains:
        {
            "row_ids": List[int],        # Original CSV row indices starting from 0
            "date": List[str],           # Date strings based primarily on answerDate
            "symbol": List[str],         # Stock symbols
            "embeddings": Tensor(n_rows, hidden_size)
        }
    """
    csv_path = os.path.join(data_dir, "QA.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"QA.csv not found: {csv_path}")

    save_dir = os.path.join(output_dir, "qa")
    _ensure_dir(save_dir)

    if progress:
        print(f"[QA] Loading file: {csv_path}")

    embedder = FinBertTextEmbedder(
        model_name_or_path=model_name_or_path,
        num_gpus=num_gpus,
        max_length=max_length,
        batch_size=batch_size,
    )

    if progress:
        print(f"[QA] Device: {embedder.device}, num_gpus={embedder.num_gpus}")
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

        # Date processing: use answerDate by default, and fall back to questionDate.
        qd = pd.to_datetime(df["questionDate"], errors="coerce")
        ad = pd.to_datetime(df["answerDate"], errors="coerce")
        date = ad.fillna(qd).dt.date.astype(str)

        # Text field: question + answer.
        df["question"] = df["question"].fillna("")
        df["answer"] = df["answer"].fillna("")
        texts = (df["question"] + " " + df["answer"]).tolist()

        # Stock symbol.
        symbols = df["symbol"].astype(str).tolist()

        if progress:
            print(f"[QA] part {part_idx:03d}, rows={n_rows}")

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
            print(f"[QA] Saved to: {out_path}")

        part_idx += 1

    if progress:
        print(f"[QA] Completed. Total rows processed = {global_row_offset}")


# ======================= Unified entry point ======================= #

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
    Build embeddings for both ChinaDaily and QA data.
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
