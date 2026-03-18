import os
import time
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
    # 1. Настройка Чанкера (Dynamic DI)
    processor_name = f"{args.chunker}_processor"
    try:
      processor_provider = getattr(self.container, processor_name)
      self.container.data_processor.override(processor_provider)
    except AttributeError:
      sys.exit(f"❌ Ошибка: Процессор '{processor_name}' не найден.")

    # 2. Индексация
    if args.need_index:
      # Ленивый импорт, чтобы избежать циклических зависимостей, если они есть
      from src.pipelines.indexing.pipeline import run_indexing
      print(f"\n🏗️  Запуск индексации ({args.chunker})...")
      processor = self.container.data_processor()
      run_indexing(self.config, processor, mode=args.index_mode)

    # 3. Подготовка
    print(f"\n🧪 ЗАПУСК ТЕСТИРОВАНИЯ...")
    loader = TestSetLoader(self.config['paths']['qa_test_set'])
    qa_set = loader.get_qa_pairs()
    if not qa_set: sys.exit("QA set empty")

    retrieval_chain = self.container.retrieval_chain()
    generation_chain = self.container.generation_chain()

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

    # Шаг 1: Batch Retrieval
    print(f"🔎 [1/2] Поиск документов ({len(qa_set)} шт)...")
    retrieval_tasks = [retrieval_chain.ainvoke(qa['question']) for qa in qa_set]
    retrieved_docs_batch = await asyncio.gather(*retrieval_tasks)

    # Шаг 2: Generation
    print(f"🤖 [2/2] Генерация и оценка...")
    results = []
    total_hit_rate = 0

    with open(trace_path, "w", encoding="utf-8") as trace_file:
      trace_file.write(f"# Trace Log\nConfig: {args.chunker}\n\n")

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
            [f"Источник: {d.metadata.get('source')}\n{d.page_content}" for d in
             safe_docs])

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

    # Итоги
    if results:
      avg_metrics = {
        "sim": sum(r['score'] for r in results) / len(results),
        "lat": sum(r['latency'] for r in results) / len(results),
        "hit": total_hit_rate / len(results)
      }
      print("\n" + "=" * 60)
      print(f"📊 ИТОГИ ({args.chunker})")
      print(f"✅ Hit Rate: {avg_metrics['hit']:.2%}")
      print(f"✅ Similarity: {avg_metrics['sim']:.4f}")
      print(f"⏱️  Avg Latency: {avg_metrics['lat']:.2f}s")
      self._save_final_report(results, args, avg_metrics)