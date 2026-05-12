"""
Оркестрация матрицы evaluation_scenarios из config.yaml: deep-merge overrides,
чекпоинт check_points/default_checkpoint.json, пауза через output/pause.flag.

При падении одного сценария (Exception) статус записывается как failed, цикл
продолжает следующие сценарии; текст ошибки в чекпоинте: scenarios.<name>.error.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.di_containers import Container
from src.evaluation.runner import EvaluationPause, TestPipelineRunner

logger = logging.getLogger(__name__)

MatrixScenario = Dict[str, Any]


def _sanitize_key(value: str) -> str:
  s = str(value).replace(os.sep, "-")
  s = re.sub(r"[^0-9A-Za-z._+]+", "-", s)
  return s.strip("-") or "x"


def deep_merge(base: dict, override: Any) -> dict:
  """Рекурсивное слияние dict; если override не dict — возвращает base."""
  result = copy.deepcopy(base)
  if override is None or not isinstance(override, dict):
    return result

  def _merge_into(dst: dict, src: dict) -> None:
    for k, v in src.items():
      if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
        _merge_into(dst[k], v)
      else:
        dst[k] = copy.deepcopy(v)

  _merge_into(result, override)
  return result


def checkpoint_path(config: dict) -> str:
  paths = config.get("paths") or {}
  return paths.get("default_checkpoint_path", "check_points/default_checkpoint.json")


def pause_flag_path(config: dict) -> str:
  paths = config.get("paths") or {}
  return paths.get("pause_flag", "output/pause.flag")


def _ensure_parent(path: str) -> None:
  parent = os.path.dirname(os.path.abspath(path))
  if parent:
    os.makedirs(parent, exist_ok=True)


def load_checkpoint(path: str) -> Optional[dict]:
  if not os.path.isfile(path):
    return None
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except (json.JSONDecodeError, OSError):
    return None


def save_checkpoint(path: str, data: dict) -> None:
  _ensure_parent(path)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


def _apply_test_index_paths(cfg: dict, chunker: str) -> None:
  base_db = cfg["retrievers"]["vector_store"].get("db_path_base", "data/chroma_db")
  cfg["retrievers"]["vector_store"]["db_path"] = f"{base_db}_{chunker}"
  base_bm25 = cfg["retrievers"]["bm25"].get("index_path_base", "data/bm25_index")
  root, ext = os.path.splitext(base_bm25)
  cfg["retrievers"]["bm25"]["index_path"] = f"{root}_{chunker}{ext}"


def manage_db_state_for_test(config_data: dict, force_reindex: bool) -> bool:
  db_path = config_data["retrievers"]["vector_store"]["db_path"]
  bm25_path = config_data["retrievers"]["bm25"]["index_path"]
  exists = os.path.exists(db_path) and os.path.exists(bm25_path)
  if force_reindex:
    if exists:
      import shutil
      logger.info("[MATRIX] Удаление старых индексов: %s", db_path)
      try:
        shutil.rmtree(db_path)
        if os.path.isfile(bm25_path):
          os.remove(bm25_path)
      except OSError as e:
        raise SystemExit(f"❌ Ошибка: Файлы БД заняты. {e}") from e
    return True
  if not exists:
    logger.warning("[MATRIX] Индексы не найдены, будет создание.")
    return True
  logger.info("[MATRIX] Найдены существующие индексы: %s", db_path)
  return False


def _empty_scenarios_state(names: List[str]) -> Dict[str, Any]:
  return {
      name: {
          "status": "pending",
          "partial_results": [],
          "pause_state": None,
          "fixed_output_stem": None,
          "dialog_session_id": None,
      }
      for name in names
  }


async def run_matrix(base_config: dict, args: Namespace) -> None:
  scenarios_cfg: List[MatrixScenario] = base_config.get("evaluation_scenarios") or []
  if not scenarios_cfg:
    raise SystemExit("❌ evaluation_scenarios пуст или отсутствует в config/config.yaml.")

  cp_path = checkpoint_path(base_config)
  pf_path = pause_flag_path(base_config)

  scenario_names = [str(s["name"]) for s in scenarios_cfg]

  if getattr(args, "resume", False):
    checkpoint = load_checkpoint(cp_path)
    if not checkpoint:
      raise SystemExit(f"❌ Нет файла чекпоинта для --resume: {cp_path}")
  else:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    checkpoint = {
        "run_id": run_id,
        "scenarios": _empty_scenarios_state(scenario_names),
    }
    save_checkpoint(cp_path, checkpoint)

  if os.path.isfile(pf_path):
    try:
      os.remove(pf_path)
    except OSError:
      pass

  for scenario in scenarios_cfg:
    name = str(scenario["name"])
    entry = checkpoint["scenarios"].setdefault(
        name,
        {
            "status": "pending",
            "partial_results": [],
            "pause_state": None,
            "fixed_output_stem": None,
            "dialog_session_id": None,
        },
    )

    if entry.get("status") == "done":
      logger.info("Сценарий «%s» уже выполнен — пропуск.", name)
      continue

    chunker = str(scenario.get("chunker", "markdown"))
    index_mode = str(scenario.get("index_mode", "test"))
    force_index_flag = bool(scenario.get("force_index", False))
    overrides = scenario.get("overrides") or {}

    merged = deep_merge(base_config, overrides)

    _apply_test_index_paths(merged, chunker)
    need_index = manage_db_state_for_test(merged, force_index_flag)

    if not entry.get("fixed_output_stem"):
      entry["fixed_output_stem"] = f"{_sanitize_key(name)}__{checkpoint['run_id']}"

    fixed_stem = entry["fixed_output_stem"]
    preloaded: List[dict] = list(entry.get("partial_results") or [])
    ps_dict = entry.get("pause_state")
    resume_trace = bool(preloaded or ps_dict)

    run_args = Namespace(
        chunker=chunker,
        retriever=str(scenario.get("retriever", "hybrid")),
        index_mode=index_mode,
        force_index=force_index_flag,
        need_index=need_index,
        eval_mode=merged.get("evaluation_settings", {}).get("mode"),
        active_type=merged.get("retrievers", {}).get("active_type"),
        memory_type=None,
        memory_off=False,
        start_question_idx=0,
        resume_dialog_sc_idx=0,
        resume_dialog_step_idx=0,
        resume_dialog_session_id=None,
        preloaded_results=preloaded,
        fixed_output_stem=fixed_stem,
        resume_continue_trace=resume_trace,
        pause_flag_path=pf_path,
    )

    if ps_dict:
      run_args.start_question_idx = int(ps_dict.get("block1_next_idx", 0))
      run_args.resume_dialog_sc_idx = int(ps_dict.get("dialog_scenario_idx", 0))
      run_args.resume_dialog_step_idx = int(ps_dict.get("dialog_step_next_idx", 0))
    run_args.resume_dialog_session_id = entry.get("dialog_session_id")

    entry["status"] = "in_progress"
    save_checkpoint(cp_path, checkpoint)

    try:
      container = Container()
      container.config.from_dict(merged)

      em = merged.get("evaluation_metrics") or {}
      use_judge = bool(em.get("enabled", False))
      runner = TestPipelineRunner(
          container,
          merged,
          faithfulness_evaluator=(
              container.faithfulness_evaluator() if use_judge else None
          ),
          relevance_evaluator=(
              container.relevance_evaluator() if use_judge else None
          ),
      )

      logger.info("%s", "=" * 60)
      logger.info("Матрица: сценарий «%s» (chunker=%s)", name, chunker)
      logger.info("%s", "=" * 60)
      results, pause = await runner.run(run_args)

      if pause is not None:
        entry["status"] = "paused"
        entry["partial_results"] = results
        entry["pause_state"] = {
            "phase": pause.phase,
            "block1_next_idx": pause.block1_next_idx,
            "dialog_scenario_idx": pause.dialog_scenario_idx,
            "dialog_step_next_idx": pause.dialog_step_next_idx,
        }
        if pause.dialog_session_id is not None:
          entry["dialog_session_id"] = pause.dialog_session_id
        elif pause.phase == "questions":
          entry["dialog_session_id"] = None
        save_checkpoint(cp_path, checkpoint)
        logger.warning(
            "Матрица остановлена (пауза). Чекпоинт: %s. "
            "Продолжить: python main.py test-matrix --resume",
            cp_path,
        )
        return

      entry["status"] = "done"
      entry.pop("error", None)
      entry["partial_results"] = []
      entry["pause_state"] = None
      entry["dialog_session_id"] = None
      save_checkpoint(cp_path, checkpoint)

    except Exception as exc:
      logger.exception(
          "Матрица: сценарий «%s» упал с ошибкой — сохраняем failed и идём дальше",
          name,
      )
      entry["status"] = "failed"
      entry["error"] = f"{type(exc).__name__}: {exc}"
      entry["partial_results"] = []
      entry["pause_state"] = None
      entry["dialog_session_id"] = None
      save_checkpoint(cp_path, checkpoint)

  failed_names = [
      n for n, e in checkpoint["scenarios"].items()
      if e.get("status") == "failed"
  ]
  if failed_names:
    logger.warning(
        "Матрица завершена со сбоями в сценариях: %s (см. error в %s)",
        ", ".join(failed_names),
        cp_path,
    )
  else:
    logger.info("Матрица сценариев завершена. Чекпоинт: %s", cp_path)
