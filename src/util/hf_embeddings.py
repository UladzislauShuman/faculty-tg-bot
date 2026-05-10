"""Параметры HuggingFace / sentence-transformers для LangChain HuggingFaceEmbeddings."""
from __future__ import annotations

from typing import Any, Dict


def huggingface_embedding_model_kwargs(
    embedding_cfg: Dict[str, Any],
) -> Dict[str, Any]:
  """device + опционально local_files_only (без сетевых HEAD к HF Hub при полном кеше)."""
  out: Dict[str, Any] = {"device": embedding_cfg.get("device", "cpu")}
  if embedding_cfg.get("local_files_only", False):
    out["local_files_only"] = True
  return out
