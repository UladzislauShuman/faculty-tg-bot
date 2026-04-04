import os
import time
import uuid
import sys
import asyncio
from datetime import datetime
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator
from src.util.yaml_parser import TestSetLoader


class TestPipelineRunner:
  """
  Класс, отвечающий за выполнение E2E тестов RAG-системы.
  """

  # Константы
  MAX_CONTEXT_CHARS = 12000
  HIT_RATE_THRESHOLD = 0.4

  def __init__(self, container, config: Dict):
    self.container = container
    self.config = config
    self.output_dir = config['paths']['output_dir']
    self.user_service = container.bot_user_service()
    self.session_service = container.bot_session_service()
    self.answer_service = container.bot_answer_service()

  def _simple_ru_stem(self, word: str) -> str:
    """Примитивный стемминг."""
    word = word.lower().strip('.,!?;:()')
    for ending in ['ами', 'ями', 'ов', 'ев', 'ей', 'ам', 'ям', 'ах', 'ях', 'ую',
                   'юю', 'а', 'я', 'о', 'е', 'ы', 'и', 'у', 'ю']:
      if word.endswith(ending) and len(word) > len(ending) + 2:
        return word[:-len(ending)]
    return word

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

  def _append_to_trace(self, file_handle, i, qa, docs, answer, score, latency,
      hit):
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
        f"- **Метрики:** {icon} | Sim: **{score:.4f}** | Time: {latency:.2f}s\n")
    file_handle.write("\n---\n\n")
    file_handle.flush()

  def _save_final_report(self, results, args, avg_metrics):
    """Сохраняет итоговый отчет."""
    filename = f"report_{args.chunker}_{args.index_mode}.md"
    report_path = os.path.join(self.output_dir, filename)

    with open(report_path, "w", encoding="utf-8") as f:
      f.write(f"# 📊 Отчет о тестировании RAG\n")
      f.write(f"**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(
          f"**Конфигурация:** Chunker=`{args.chunker}`, Mode=`{args.index_mode}`, Retriever=`{args.retriever}`\n\n")
      f.write("## 📈 Сводные метрики\n")
      f.write(f"- **Hit Rate:** {avg_metrics['hit']:.2%}\n")
      f.write(f"- **Similarity:** {avg_metrics['sim']:.4f}\n")
      f.write(f"- **Latency:** {avg_metrics['lat']:.2f}s\n\n")

      f.write("## 📝 Детализация\n")
      for i, res in enumerate(results, 1):
        icon = "✅" if res['hit'] else "❌"
        f.write(
            f"### {i}. {res['q']}\n- **Эталон:** {res['ref']}\n- **Ответ:** {res['a']}\n- **Метрики:** {icon} Hit={res['hit']} | Sim={res['score']:.4f}\n\n")

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

    # Оценщик
    eval_emb = HuggingFaceEmbeddings(
        model_name=self.config.get('evaluation_model', {}).get('name',
                                                               'cointegrated/rubert-tiny2'),
        model_kwargs={'device': 'cpu'}
    )
    evaluator = load_evaluator("embedding_distance", embeddings=eval_emb)

    # Трассировка
    os.makedirs(self.output_dir, exist_ok=True)
    trace_path = os.path.join(self.output_dir,
                              f"trace_{args.chunker}_{datetime.now().strftime('%H%M')}.md")
    print(f"📝 Лог: {trace_path}")

    results = []
    total_hit_rate = 0

    with open(trace_path, "w", encoding="utf-8") as trace_file:
      trace_file.write(
        f"# Trace Log\nConfig: {args.chunker}\nTest Mode: {test_mode}\n\n")

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

              dist = evaluator.evaluate_strings(prediction=response,
                                                reference=qa['answer'])['score']
              score = 1.0 - dist

              self._append_to_trace(trace_file, i + 1, qa, safe_docs, response,
                                    score, latency, hit)

              icon = "✅" if hit else "❌"
              print(f"   -> {icon} Hit | Sim: {score:.2f} | {latency:.1f}s")

              results.append({
                "q": qa['question'], "a": response, "ref": qa['answer'],
                "score": score, "latency": latency, "hit": hit
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
                dist = evaluator.evaluate_strings(prediction=bot_answer,
                                                  reference=step['a'])['score']
                score = 1.0 - dist

                # 4. Логируем
                step_qa = {"question": step['q'], "answer": step['a']}
                self._append_to_trace(trace_file,
                                      f"[{sc['name']}] Шаг {step_idx}", step_qa,
                                      docs, bot_answer, score, latency, hit)

                icon = "✅" if hit else "❌"
                print(f"     -> {icon} Hit | Sim: {score:.2f} | {latency:.1f}s")

                results.append({
                  "q": f"[{sc['name']}] {step['q']}", "a": bot_answer,
                  "ref": step['a'],
                  "score": score, "latency": latency, "hit": hit
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
      print(f"✅ Similarity: {avg_metrics['sim']:.4f}")
      print(f"⏱️  Avg Latency: {avg_metrics['lat']:.2f}s")
      self._save_final_report(results, args, avg_metrics)
    else:
      print(
        "\n⚠️ Нет результатов для отображения (проверьте qa-test-set.yaml и режим тестирования).")