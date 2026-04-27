import os
import re
import sys
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.schemas import EvalScore
from src.evaluation.metrics import FaithfulnessEvaluator, RelevanceEvaluator
from src.util.yaml_parser import TestSetLoader


class TestPipelineRunner:
  """
  Класс, отвечающий за выполнение E2E тестов RAG-системы.
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
        retriever_type, memory_type, ts,
    ]
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
  ) -> None:
    """Пишет лог в файл в реальном времени."""
    file_handle.write(f"## ❓ Вопрос {i}: {qa['question']}\n\n")
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

    print(f"\n📄 Отчет сохранен: {report_path}")

  async def run(self, args):
    # 1. Настройка Чанкера
    processor_name = f"{args.chunker}_processor"
    try:
      processor_provider = getattr(self.container, processor_name)
      self.container.data_processor.override(processor_provider)
    except AttributeError:
      sys.exit(f"❌ Ошибка: Процессор '{processor_name}' не найден.")

    # 2. Индексация
    if args.need_index:
      from src.pipelines.indexing.pipeline import run_indexing
      print(f"\n🏗️  Запуск индексации ({args.chunker})...")
      processor = self.container.data_processor()
      run_indexing(self.config, processor, mode=args.index_mode)

    # 3. Подготовка
    test_mode = self.config.get('evaluation_settings', {}).get('mode', 'all')
    print(f"\n🧪 ЗАПУСК ТЕСТИРОВАНИЯ (Режим: {test_mode})...")

    loader = TestSetLoader(self.config['paths']['qa_test_set'])

    # Инициализируем цепочки
    retrieval_chain = self.container.retrieval_chain()
    generation_chain = self.container.generation_chain()
    rag_chain = self.container.rag_chain()

    # Подготовка тестового пользователя (ID=0) для соблюдения FK в БД
    TEST_USER_ID = 0
    await self.user_service.get_or_create_user(
        user_id=TEST_USER_ID,
        first_name="TestRunner",
        username="test_bot"
    )

    # Трассировка: имя trace/report с одинаковой меткой на прогон
    os.makedirs(self.output_dir, exist_ok=True)
    output_stem = self._output_label_stem(args)
    trace_path = os.path.join(self.output_dir, f"trace_{output_stem}.md")
    print(f"📝 Лог: {trace_path}")

    results = []
    total_hit_rate = 0

    with open(trace_path, "w", encoding="utf-8") as trace_file:
      trace_file.write(
        f"# Trace Log\nLabel: `{output_stem}`\n"
        f"Chunker: {args.chunker} | Test mode: {test_mode} | "
        f"active_type: {self.config.get('retrievers', {}).get('active_type', 'na')} | "
        f"memory: {self.config.get('memory') or {}}\n\n"
      )

      # =================================================================
      # БЛОК 1: ОДИНОЧНЫЕ ВОПРОСЫ
      # =================================================================
      if test_mode in ['all', 'questions']:
        qa_set = loader.get_qa_pairs()
        if qa_set:
          print(f"\n🔎 [БЛОК 1] Одиночные вопросы ({len(qa_set)} шт)...")
          retrieval_tasks = [retrieval_chain.ainvoke(qa['question']) for qa in
                             qa_set]
          retrieved_docs_batch = await asyncio.gather(*retrieval_tasks)

          for i, (qa, docs) in enumerate(zip(qa_set, retrieved_docs_batch)):
            hit = self._calculate_hit_rate(qa['answer'], docs)
            total_hit_rate += hit

            print(f"[{i + 1}/{len(qa_set)}] {qa['question'][:50]}...")

            # Обрезка контекста
            safe_docs = []
            current_chars = 0
            for d in docs:
              doc_len = len(d.page_content)
              if current_chars + doc_len < self.MAX_CONTEXT_CHARS:
                safe_docs.append(d)
                current_chars += doc_len
              else:
                break

            full_context = "\n\n".join(
                [f"Источник: {d.metadata.get('source')}\n{d.page_content}" for d
                 in safe_docs])

            start_time = time.time()
            try:
              response = await generation_chain.ainvoke(
                  {"context": full_context, "question": qa['question']})
              latency = time.time() - start_time

              faith_score, rel_score = await self._aevaluate_judge_scores(
                  qa['question'], response, full_context
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
              )

              icon = "✅" if hit else "❌"
              print(
                  f"   -> {icon} Hit | Score: {score:.2f} | {latency:.1f}s"
              )

              results.append({
                "q": qa['question'], "a": response, "ref": qa['answer'],
                "score": score, "latency": latency, "hit": hit,
                "faithfulness": faith_score.score if faith_score else None,
                "relevance": rel_score.score if rel_score else None,
              })
            except Exception as e:
              print(f"   -> ❌ Error: {e}")
              trace_file.write(f"## Error: {e}\n")

      # =================================================================
      # БЛОК 2: МНОГОШАГОВЫЕ СЦЕНАРИИ (ДИАЛОГИ)
      # =================================================================
      if test_mode in ['all', 'scenarios']:
        scenarios = loader.get_test_scenarios()
        if scenarios:
          print(f"\n🎬 [БЛОК 2] Многошаговые сценарии ({len(scenarios)} шт)...")
          for sc in scenarios:
            # Создаем реальную сессию в БД для этого сценария
            session_id = await self.session_service.start_new_session(
              user_id=TEST_USER_ID)

            trace_file.write(
              f"## 🎬 Сценарий: {sc['name']} (Session: {session_id})\n\n")
            print(f"\n--- Сценарий: {sc['name']} ---")

            for step_idx, step in enumerate(sc['steps'], 1):
              print(f"  Шаг {step_idx}: {step['q'][:50]}...")
              start_time = time.time()
              try:
                # 1. Вызываем умную RAG-цепочку
                response = await rag_chain.ainvoke(
                    {"input": step['q']},
                    config={"configurable": {"session_id": session_id}}
                )
                latency = time.time() - start_time
                bot_answer = response['answer']
                docs = response.get('context', [])

                # 2. Сохраняем в БД (теперь ID сессии валидный)
                await self.answer_service.save_answer(session_id, step['q'],
                                                      bot_answer)

                # 3. Считаем метрики
                hit = self._calculate_hit_rate(step['a'], docs)
                total_hit_rate += hit
                full_context = self._build_full_context_from_docs(docs)
                faith_score, rel_score = await self._aevaluate_judge_scores(
                    step['q'], bot_answer, full_context
                )
                score = self._aggregate_judge_score(faith_score, rel_score)

                # 4. Логируем
                step_qa = {"question": step['q'], "answer": step['a']}
                self._append_to_trace(
                    trace_file,
                    f"[{sc['name']}] Шаг {step_idx}",
                    step_qa,
                    docs,
                    bot_answer,
                    score,
                    latency,
                    hit,
                    faithfulness=faith_score,
                    relevance=rel_score,
                )

                icon = "✅" if hit else "❌"
                print(
                    f"     -> {icon} Hit | Score: {score:.2f} | {latency:.1f}s"
                )

                results.append({
                  "q": f"[{sc['name']}] {step['q']}", "a": bot_answer,
                  "ref": step['a'],
                  "score": score, "latency": latency, "hit": hit,
                  "faithfulness": faith_score.score if faith_score else None,
                  "relevance": rel_score.score if rel_score else None,
                })
              except Exception as e:
                print(f"     -> ❌ Error: {e}")
                trace_file.write(f"## Error on step {step_idx}: {e}\n")

    # =================================================================
    # ИТОГИ
    # =================================================================
    if results:
      avg_metrics = {
        "sim": sum(r['score'] for r in results) / len(results),
        "lat": sum(r['latency'] for r in results) / len(results),
        "hit": total_hit_rate / len(results)
      }
      print("\n" + "=" * 60)
      print(f"📊 ИТОГИ ({args.chunker} | Режим: {test_mode})")
      print(f"✅ Hit Rate: {avg_metrics['hit']:.2%}")
      print(f"✅ Score (aggregate): {avg_metrics['sim']:.4f}")
      print(f"⏱️  Avg Latency: {avg_metrics['lat']:.2f}s")
      rows_f = [r for r in results if r.get("faithfulness") is not None]
      rows_r = [r for r in results if r.get("relevance") is not None]
      if rows_f:
        print(
            f"✅ Avg Faithfulness: {sum(r['faithfulness'] for r in rows_f) / len(rows_f):.4f} "
            f"(n={len(rows_f)})"
        )
      if rows_r:
        print(
            f"✅ Avg Relevance: {sum(r['relevance'] for r in rows_r) / len(rows_r):.4f} "
            f"(n={len(rows_r)})"
        )
      self._save_final_report(results, args, avg_metrics, output_stem)
    else:
      print(
        "\n⚠️ Нет результатов для отображения (проверьте qa-test-set.yaml и режим тестирования).")