"""Прогон E2E evaluation: вопросы из qa-set, RAG, hit-rate, LLM-judge, trace/report.

Пишет отчёты в paths.output_dir; может ставить паузу между шагами и обрабатывать сигналы.
"""
import asyncio
import copy
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.schemas import EvalScore
from src.evaluation.metrics import FaithfulnessEvaluator, RelevanceEvaluator
from src.retrievers.hyde_trace_context import hyde_trace_begin, hyde_trace_end
from src.util.yaml_parser import TestSetLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationPause:
  """Точка остановки между вопросами (пауза / graceful shutdown)."""

  phase: str  # "questions" | "dialogs"
  block1_next_idx: int
  dialog_scenario_idx: int
  dialog_step_next_idx: int
  dialog_session_id: Optional[str] = None


class TestPipelineRunner:
  """Оркестратор тестов: DI container, БД-сервисы бота, RAG, метрики, файлы trace/report.
  """

  # Константы
  MAX_CONTEXT_CHARS = 12000
  HIT_RATE_THRESHOLD = 0.4

  def __init__(
    self,
    container: Any,
    config: Dict,
    faithfulness_evaluator: Optional[FaithfulnessEvaluator] = None,
    relevance_evaluator: Optional[RelevanceEvaluator] = None,
  ) -> None:
    self.container = container
    self.config = config
    self.output_dir = config['paths']['output_dir']
    self.user_service = container.bot_user_service()
    self.session_service = container.bot_session_service()
    self.answer_service = container.bot_answer_service()
    self.faithfulness_evaluator: Optional[FaithfulnessEvaluator] = (
        faithfulness_evaluator
    )
    self.relevance_evaluator: Optional[RelevanceEvaluator] = (
        relevance_evaluator
    )

  def _hyde_trace_enabled_for_run(self) -> bool:
    h = self.config.get("hyde")
    return isinstance(h, dict) and bool(h.get("enabled"))

  @staticmethod
  def _format_hyde_trace_block(events: Optional[List[Dict[str, Any]]]) -> str:
    if not events:
      return ""
    parts: List[str] = ["### 🔬 HyDE (dense-поиск)\n\n"]
    for idx, evt in enumerate(events, 1):
      st = evt.get("status", "?")
      parts.append(f"#### Вызов dense `embed_query` #{idx}: `{st}`\n\n")
      parts.append(
          f"- **LLM время (слот гипотез):** `{evt.get('elapsed_hypothesis_llm_s')} s`\n"
          f"- **num_hypotheses (config):** {evt.get('num_hypotheses_config')}\n"
      )
      prev = evt.get("user_query_preview") or ""
      parts.append(f"- **Входящий текст (preview):**\n```\n{prev}\n```\n")
      hp = evt.get("hypotheses") or []
      if hp:
        parts.append("- **Гипотетические фрагменты:**\n")
        for hi, ht in enumerate(hp, 1):
          esc = ht.replace("```", "``\\`") if ht else "(пусто)"
          parts.append(f"  **{hi}.**\n```text\n{esc}\n```\n")
      hits = evt.get("hypothesis_issues") or []
      if hits:
        parts.append(
            "- **Примечания по слотам LLM:** "
            + ", ".join(f"`{str(h)}`" for h in hits)
            + "\n",
        )
      note = evt.get("embed_phase_note")
      if note:
        parts.append(f"- **Комментарий:** _{note}_\n")
      parts.append("\n")
    parts.append("")
    return "".join(parts)

  @staticmethod
  def _sanitize_filename_token(value: str) -> str:
    s = str(value).replace(os.sep, "-")
    s = re.sub(r"[^0-9A-Za-z._+]+", "-", s)
    return s.strip("-") or "x"

  def _output_label_stem(self, args: Any) -> str:
    """
    Суффикс для report_*.md / trace_*.md:
    eval_mode-chunker-retriever_strategy-index_mode-force_flag-retriever_type-memory_type-timestamp
    """
    eval_mode = (
        getattr(args, "eval_mode", None)
        or self.config.get("evaluation_settings", {}).get("mode", "all")
    )
    chunker = getattr(args, "chunker", "markdown")
    fixed = getattr(args, "fixed_output_stem", None)
    if fixed:
      return self._sanitize_filename_token(fixed)
    retriever_strategy = getattr(args, "retriever", "hybrid")
    index_mode = getattr(args, "index_mode", "test")
    force_flag = "force" if getattr(args, "force_index", False) else "noforce"
    retriever_type = self.config.get("retrievers", {}).get("active_type", "na")
    mem = self.config.get("memory") or {}
    if not mem.get("enabled", False):
      memory_type = "off"
    else:
      memory_type = mem.get("type", "na")
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    parts = [
        eval_mode, chunker, retriever_strategy, index_mode, force_flag,
        retriever_type, memory_type,
    ]
    if (self.config.get("hyde") or {}).get("enabled", False):
      parts.append("hyde_on")
    parts.append(ts)
    return "-".join(self._sanitize_filename_token(p) for p in parts)

  def _simple_ru_stem(self, word: str) -> str:
    """Примитивный стемминг."""
    word = word.lower().strip('.,!?;:()')
    for ending in ['ами', 'ями', 'ов', 'ев', 'ей', 'ам', 'ям', 'ах', 'ях', 'ую',
                   'юю', 'а', 'я', 'о', 'е', 'ы', 'и', 'у', 'ю']:
      if word.endswith(ending) and len(word) > len(ending) + 2:
        return word[:-len(ending)]
    return word

  async def _aevaluate_judge_scores(
    self,
    question: str,
    answer: str,
    full_context: str,
  ) -> Tuple[Optional[EvalScore], Optional[EvalScore]]:
    """Последовательные LLM-вызовы судьи (экономия RAM)."""
    em = self.config.get("evaluation_metrics") or {}
    mflags = em.get("metrics") or {}
    faith_score: Optional[EvalScore] = None
    rel_score: Optional[EvalScore] = None
    if self.faithfulness_evaluator and mflags.get("faithfulness", True):
      faith_score = await self.faithfulness_evaluator.aevaluate(
          answer=answer, context=full_context
      )
    if self.relevance_evaluator and mflags.get("answer_relevance", True):
      rel_score = await self.relevance_evaluator.aevaluate(
          question=question, answer=answer
      )
    return faith_score, rel_score

  @staticmethod
  def _aggregate_judge_score(
    faith: Optional[EvalScore],
    rel: Optional[EvalScore],
  ) -> float:
    if faith and rel:
      return (faith.score + rel.score) / 2.0
    if faith:
      return float(faith.score)
    if rel:
      return float(rel.score)
    return 0.0

  def _build_full_context_from_docs(
    self, docs: list, max_chars: int = MAX_CONTEXT_CHARS
  ) -> str:
    current_chars = 0
    parts: List[str] = []
    for d in docs or []:
      block = f"Источник: {d.metadata.get('source')}\n{d.page_content}"
      if current_chars + len(block) < max_chars:
        parts.append(block)
        current_chars += len(block)
      else:
        break
    return "\n\n".join(parts)

  def _calculate_hit_rate(self, reference: str, retrieved_docs) -> int:
    """Эвристическая проверка: нашел ли ретривер ответ."""
    reference_words = [self._simple_ru_stem(w) for w in reference.split() if
                       len(w) > 3]
    if not reference_words: return 0

    for doc in retrieved_docs:
      content = doc.page_content.lower()
      found_count = sum(1 for stem in reference_words if stem in content)
      if found_count / len(reference_words) >= self.HIT_RATE_THRESHOLD:
        return 1
    return 0

  def _append_to_trace(
      self,
      file_handle,
      i,
      qa: Dict,
      docs,
      answer: str,
      score: float,
      latency: float,
      hit: int,
      faithfulness: Optional[EvalScore] = None,
      relevance: Optional[EvalScore] = None,
      hyde_trace: Optional[List[Dict[str, Any]]] = None,
  ) -> None:
    """Пишет лог в файл в реальном времени."""
    file_handle.write(f"## ❓ Вопрос {i}: {qa['question']}\n\n")
    md_hyde = self._format_hyde_trace_block(hyde_trace)
    if md_hyde:
      file_handle.write(md_hyde + "\n")
    file_handle.write(f"### 🔎 Найденные документы (Top-{len(docs)})\n")
    for j, doc in enumerate(docs, 1):
      source = doc.metadata.get('source', 'N/A')
      preview = doc.page_content.replace('\n', ' ')[:300] + "..."
      file_handle.write(f"**{j}. [{source}]**\n> {preview}\n\n")

    file_handle.write(f"### 🤖 Ответ бота\n{answer}\n\n")
    icon = "✅ HIT" if hit else "❌ MISS"
    file_handle.write(f"### 📏 Оценка\n")
    file_handle.write(f"- **Эталон:** {qa['answer']}\n")
    file_handle.write(
        f"- **Метрики:** {icon} | Score: **{score:.4f}** | Time: {latency:.2f}s\n"
    )
    if faithfulness is not None:
      file_handle.write(
          f"- **Faithfulness:** {faithfulness.score:.2f} — {faithfulness.reason}\n"
      )
    if relevance is not None:
      file_handle.write(
          f"- **Relevance:** {relevance.score:.2f} — {relevance.reason}\n"
      )
    file_handle.write("\n---\n\n")
    file_handle.flush()

  def _save_final_report(
      self, results, args, avg_metrics, output_stem: Optional[str] = None
  ) -> None:
    """Сохраняет итоговый отчет."""
    stem = output_stem or self._output_label_stem(args)
    filename = f"report_{stem}.md"
    report_path = os.path.join(self.output_dir, filename)

    with open(report_path, "w", encoding="utf-8") as f:
      f.write(f"# 📊 Отчет о тестировании RAG\n")
      f.write(f"**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(
          f"**Конфигурация:** Chunker=`{args.chunker}`, Mode=`{args.index_mode}`, Retriever=`{args.retriever}`\n\n"
      )
      f.write("## 📈 Сводные метрики\n")
      f.write(f"- **Hit Rate:** {avg_metrics['hit']:.2%}\n")
      f.write(
          f"- **Score (judge, среднее по шагам):** {avg_metrics['sim']:.4f}\n"
      )
      f.write(f"- **Средняя задержка (latency):** {avg_metrics['lat']:.2f}s\n")

      rows_f = [r for r in results if r.get("faithfulness") is not None]
      rows_r = [r for r in results if r.get("relevance") is not None]
      if rows_f or rows_r:
        f.write("\n### LLM-as-a-Judge (по шагам с оценкой)\n")
        if rows_f:
          avg_faith = sum(r["faithfulness"] for r in rows_f) / len(rows_f)
          f.write(f"- **Avg Faithfulness:** {avg_faith:.4f} _(n={len(rows_f)})_\n")
        if rows_r:
          avg_rel = sum(r["relevance"] for r in rows_r) / len(rows_r)
          f.write(f"- **Avg Relevance:** {avg_rel:.4f} _(n={len(rows_r)})_\n")
        f.write(
            "\n*Faithfulness = опора ответа на retrieved-контекст; Relevance = соответствие вопросу.*\n"
        )
      f.write("\n---\n\n")

      f.write("## 📝 Детализация по вопросам\n")
      for i, res in enumerate(results, 1):
        icon = "✅" if res['hit'] else "❌"
        line = (
            f"### {i}. {res['q']}\n"
            f"- **Эталон:** {res['ref']}\n"
            f"- **Ответ:** {res['a']}\n"
            f"- **Метрики:** {icon} Hit={res['hit']} | "
            f"Score={res['score']:.4f}\n"
        )
        f_extra = res.get("faithfulness")
        r_extra = res.get("relevance")
        if f_extra is not None or r_extra is not None:
          bits = []
          if f_extra is not None:
            bits.append(f"F={f_extra:.3f}")
          if r_extra is not None:
            bits.append(f"R={r_extra:.3f}")
          line = line.rstrip() + f" | {' / '.join(bits)}\n"
        f.write(line + "\n")

    logger.info("Отчёт сохранён: %s", report_path)

  async def run(
      self,
      args: Any,
  ) -> Tuple[List[Dict[str, Any]], Optional[EvaluationPause]]:
    paused_path = getattr(args, "pause_flag_path", None) or os.path.join(
        self.output_dir, "pause.flag"
    )

    def _ensure_pause_parent() -> None:
      pd = os.path.dirname(os.path.abspath(paused_path))
      if pd:
        os.makedirs(pd, exist_ok=True)

    def _touch_pause_signal() -> None:
      _ensure_pause_parent()
      Path(paused_path).touch()

    loop = asyncio.get_running_loop()
    try:
      loop.add_signal_handler(signal.SIGINT, _touch_pause_signal)
      loop.add_signal_handler(signal.SIGTERM, _touch_pause_signal)
    except (NotImplementedError, RuntimeError):
      signal.signal(signal.SIGINT, lambda *_unused: _touch_pause_signal())

    def _pause_requested() -> bool:
      return os.path.isfile(paused_path)

    interrupted: Optional[EvaluationPause] = None
    results: List[Dict[str, Any]]
    preload = getattr(args, "preloaded_results", None)
    if preload is None:
      results = []
    else:
      results = copy.deepcopy(preload)

    processor_name = f"{args.chunker}_processor"
    try:
      processor_provider = getattr(self.container, processor_name)
      self.container.data_processor.override(processor_provider)
    except AttributeError:
      sys.exit(f"❌ Ошибка: Процессор '{processor_name}' не найден.")

    if args.need_index:
      from src.pipelines.indexing.pipeline import run_indexing
      logger.info("Запуск индексации (chunker=%s)", args.chunker)
      processor = self.container.data_processor()
      run_indexing(self.config, processor, mode=args.index_mode)

    test_mode = self.config.get("evaluation_settings", {}).get("mode", "all")
    logger.info("Запуск тестирования (режим=%s)", test_mode)

    loader = TestSetLoader(self.config["paths"]["qa_test_set"])
    retrieval_chain = self.container.retrieval_chain()
    generation_chain = self.container.generation_chain()
    rag_chain = self.container.rag_chain()

    TEST_USER_ID = 0
    await self.user_service.get_or_create_user(
        user_id=TEST_USER_ID,
        first_name="TestRunner",
        username="test_bot",
    )

    os.makedirs(self.output_dir, exist_ok=True)
    output_stem = self._output_label_stem(args)
    trace_path = os.path.join(self.output_dir, f"trace_{output_stem}.md")
    logger.info("Trace лог: %s", trace_path)

    total_hit_rate = sum(int(r["hit"]) for r in results)

    trace_mode = (
        "a"
        if getattr(args, "resume_continue_trace", False)
        and os.path.isfile(trace_path)
        else "w"
    )

    try:
      with open(trace_path, trace_mode, encoding="utf-8") as trace_file:
        if trace_mode == "w":
          _hyde = self.config.get("hyde") or {}
          _hyde_ln = ""
          if isinstance(_hyde, dict):
            _hyde_ln = (
                f"HyDE: enabled={_hyde.get('enabled', False)}, "
                f"num_hypotheses={_hyde.get('num_hypotheses')}, "
                f"verbose_console={_hyde.get('verbose_console', False)}\n"
            )
          trace_file.write(
              f"# Trace Log\nLabel: `{output_stem}`\n"
              f"Chunker: {args.chunker} | Test mode: {test_mode} | "
              f"active_type: {self.config.get('retrievers', {}).get('active_type', 'na')} | "
              f"memory: {self.config.get('memory') or {}}\n{_hyde_ln}\n"
          )
        else:
          trace_file.write("\n\n## RESUME\n\n")

        qa_set: List[Dict[str, Any]] = []
        if test_mode in ("all", "questions"):
          qa_set = loader.get_qa_pairs()
        start_q = int(getattr(args, "start_question_idx", 0) or 0)

        if test_mode in ("all", "questions") and qa_set:
          logger.info("Блок 1: одиночные вопросы (%s шт)", len(qa_set))
          for i in range(start_q, len(qa_set)):
            if _pause_requested():
              interrupted = EvaluationPause("questions", i, 0, 0, None)
              break
            qa = qa_set[i]
            hy_snap: List[Dict[str, Any]] = []
            hy_tok = None
            if self._hyde_trace_enabled_for_run():
              hy_tok = hyde_trace_begin()
            try:
              docs = await retrieval_chain.ainvoke(qa["question"])
            finally:
              if hy_tok is not None:
                hy_snap = hyde_trace_end(hy_tok)
            hit = self._calculate_hit_rate(qa["answer"], docs)
            total_hit_rate += hit

            qpv = qa["question"]
            if len(qpv) > 120:
              qpv = qpv[:117] + "..."
            logger.info("[%s/%s] %s", i + 1, len(qa_set), qpv)

            safe_docs: List[Any] = []
            current_chars = 0
            for d in docs:
              doc_len = len(d.page_content)
              if current_chars + doc_len < self.MAX_CONTEXT_CHARS:
                safe_docs.append(d)
                current_chars += doc_len
              else:
                break

            full_context = "\n\n".join(
                [
                    f"Источник: {d.metadata.get('source')}\n{d.page_content}"
                    for d in safe_docs
                ]
            )

            start_time = time.time()
            try:
              response = await generation_chain.ainvoke(
                  {"context": full_context, "question": qa["question"]}
              )
              latency = time.time() - start_time

              faith_score, rel_score = await self._aevaluate_judge_scores(
                  qa["question"], response, full_context
              )
              score = self._aggregate_judge_score(faith_score, rel_score)

              self._append_to_trace(
                  trace_file,
                  i + 1,
                  qa,
                  safe_docs,
                  response,
                  score,
                  latency,
                  hit,
                  faithfulness=faith_score,
                  relevance=rel_score,
                  hyde_trace=hy_snap,
              )

              icon = "✅" if hit else "❌"
              logger.info(
                  "   -> %s Hit | score=%.2f | latency=%.1fs",
                  icon,
                  score,
                  latency,
              )

              results.append({
                  "q": qa["question"],
                  "a": response,
                  "ref": qa["answer"],
                  "score": score,
                  "latency": latency,
                  "hit": hit,
                  "faithfulness": faith_score.score if faith_score else None,
                  "relevance": rel_score.score if rel_score else None,
              })
            except Exception as e:
              logger.exception("Ошибка при ответе на вопрос #%s", i + 1)
              trace_file.write(f"## Error: {e}\n")

            if _pause_requested():
              interrupted = EvaluationPause(
                  "questions",
                  i + 1,
                  0,
                  0,
                  None,
              )
              break

        resume_sc = int(getattr(args, "resume_dialog_sc_idx", 0) or 0)
        resume_step = int(getattr(args, "resume_dialog_step_idx", 0) or 0)
        resume_sid_obj = getattr(args, "resume_dialog_session_id", None)
        resume_sid: Optional[str] = (
            str(resume_sid_obj) if resume_sid_obj is not None else None
        )

        scenarios: List[Dict[str, Any]] = []
        if test_mode in ("all", "scenarios"):
          scenarios = loader.get_test_scenarios()

        if interrupted is None and scenarios:
          logger.info(
              "Блок 2: многошаговые сценарии (%s шт)", len(scenarios)
          )
          n_qa = len(qa_set)
          for sc_idx, sc in enumerate(scenarios):
            if interrupted:
              break
            if sc_idx < resume_sc:
              continue

            steps = sc["steps"]
            session_id: str
            if sc_idx == resume_sc and resume_sid is not None:
              session_id = resume_sid
            else:
              session_id = await self.session_service.start_new_session(
                  user_id=TEST_USER_ID
              )

            header_written = False
            for step_idx, step in enumerate(steps):
              if interrupted:
                break
              if sc_idx == resume_sc and step_idx < resume_step:
                continue

              if _pause_requested():
                interrupted = EvaluationPause(
                    "dialogs",
                    n_qa,
                    sc_idx,
                    step_idx,
                    session_id,
                )
                break

              if not header_written:
                trace_file.write(
                    f"## 🎬 Сценарий: {sc['name']} (Session: {session_id})\n\n"
                )
                logger.info("Сценарий: %s", sc["name"])
                header_written = True

              display_step = step_idx + 1
              sq = step["q"]
              if len(sq) > 120:
                sq = sq[:117] + "..."
              logger.info("  Шаг %s: %s", display_step, sq)
              start_time = time.time()
              hy_snap_sc: List[Dict[str, Any]] = []
              hy_tok_sc = None
              if self._hyde_trace_enabled_for_run():
                hy_tok_sc = hyde_trace_begin()
              try:
                try:
                  response = await rag_chain.ainvoke(
                      {"input": step["q"]},
                      config={"configurable": {"session_id": session_id}},
                  )
                finally:
                  if hy_tok_sc is not None:
                    hy_snap_sc = hyde_trace_end(hy_tok_sc)
                latency = time.time() - start_time
                bot_answer = response["answer"]
                docs = response.get("context", [])

                await self.answer_service.save_answer(
                    session_id,
                    step["q"],
                    bot_answer,
                )

                hit = self._calculate_hit_rate(step["a"], docs)
                total_hit_rate += hit
                full_context = self._build_full_context_from_docs(docs)
                faith_score, rel_score = await self._aevaluate_judge_scores(
                    step["q"],
                    bot_answer,
                    full_context,
                )
                score = self._aggregate_judge_score(faith_score, rel_score)

                step_qa = {"question": step["q"], "answer": step["a"]}
                self._append_to_trace(
                    trace_file,
                    f"[{sc['name']}] Шаг {display_step}",
                    step_qa,
                    docs,
                    bot_answer,
                    score,
                    latency,
                    hit,
                    faithfulness=faith_score,
                    relevance=rel_score,
                    hyde_trace=hy_snap_sc,
                )

                icon = "✅" if hit else "❌"
                logger.info(
                    "     -> %s Hit | score=%.2f | latency=%.1fs",
                    icon,
                    score,
                    latency,
                )

                results.append({
                    "q": f"[{sc['name']}] {step['q']}",
                    "a": bot_answer,
                    "ref": step["a"],
                    "score": score,
                    "latency": latency,
                    "hit": hit,
                    "faithfulness": (
                        faith_score.score if faith_score else None
                    ),
                    "relevance": rel_score.score if rel_score else None,
                })
              except Exception as e:
                logger.exception(
                    "Ошибка в сценарии «%s», шаг %s",
                    sc["name"],
                    display_step,
                )
                trace_file.write(
                    f"## Error on step {display_step}: {e}\n"
                )

              if _pause_requested():
                nxt = step_idx + 1
                if nxt < len(steps):
                  ns, nt, sid_store = sc_idx, nxt, session_id
                else:
                  ns, nt, sid_store = sc_idx + 1, 0, None
                interrupted = EvaluationPause(
                    "dialogs",
                    n_qa,
                    ns,
                    nt,
                    sid_store,
                )
                break

    finally:
      try:
        loop.remove_signal_handler(signal.SIGINT)
        loop.remove_signal_handler(signal.SIGTERM)
      except (NotImplementedError, RuntimeError, ValueError):
        pass

    if results and interrupted is None:
      avg_metrics = {
          "sim": sum(r["score"] for r in results) / len(results),
          "lat": sum(r["latency"] for r in results) / len(results),
          "hit": total_hit_rate / len(results),
      }
      logger.info("=" * 60)
      logger.info(
          "Итоги chunker=%s режим=%s", args.chunker, test_mode
      )
      logger.info("Hit rate: %.2f%%", avg_metrics["hit"] * 100)
      logger.info("Score (aggregate): %.4f", avg_metrics["sim"])
      logger.info("Avg latency: %.2fs", avg_metrics["lat"])
      rows_f = [r for r in results if r.get("faithfulness") is not None]
      rows_r = [r for r in results if r.get("relevance") is not None]
      if rows_f:
        avg_faith = sum(r["faithfulness"] for r in rows_f) / len(rows_f)
        logger.info(
            "Avg faithfulness: %.4f (n=%s)", avg_faith, len(rows_f)
        )
      if rows_r:
        avg_rel = sum(r["relevance"] for r in rows_r) / len(rows_r)
        logger.info(
            "Avg relevance: %.4f (n=%s)", avg_rel, len(rows_r)
        )
      self._save_final_report(results, args, avg_metrics, output_stem)
    elif not results:
      logger.warning(
          "Нет результатов (проверьте qa-test-set и evaluation_settings.mode)."
      )

    return results, interrupted