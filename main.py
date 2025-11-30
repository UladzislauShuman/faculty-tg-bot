import argparse
import yaml
import sys
import os
import shutil
import asyncio
import time
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
# Исправленный импорт для новых версий LangChain
from langchain.evaluation import load_evaluator

# Твои модули
from src.di_containers import Container
from src.pipelines.indexing.pipeline import run_indexing
from src.util.yaml_parser import TestSetLoader

# --- КОНСТАНТЫ ---
MAX_CONTEXT_CHARS = 12000
HIT_RATE_THRESHOLD = 0.4


# --- УТИЛИТЫ ---

def manage_db_state_for_test(config_data: dict, force_reindex: bool):
  """Управляет путями ТОЛЬКО для ТЕСТОВОГО режима."""
  db_path = config_data['retrievers']['vector_store']['db_path']
  bm25_path = config_data['retrievers']['bm25']['index_path']

  exists = os.path.exists(db_path) and os.path.exists(bm25_path)

  if force_reindex:
    if exists:
      print(f"♻️  [TEST] Удаление старых индексов: {db_path}...")
      try:
        shutil.rmtree(db_path)
        if os.path.exists(bm25_path): os.remove(bm25_path)
      except OSError as e:
        print(f"❌ Ошибка: Файлы БД заняты. {e}")
        sys.exit(1)
    return True

  if not exists:
    print(f"⚠️  [TEST] Индексы не найдены. Требуется создание.")
    return True

  print(f"✅  [TEST] Найдены существующие индексы: {db_path}")
  return False


def simple_ru_stem(word: str) -> str:
  word = word.lower().strip('.,!?;:()')
  for ending in ['ами', 'ями', 'ов', 'ев', 'ей', 'ам', 'ям', 'ах', 'ях', 'ую',
                 'юю', 'а', 'я', 'о', 'е', 'ы', 'и', 'у', 'ю']:
    if word.endswith(ending) and len(word) > len(ending) + 2:
      return word[:-len(ending)]
  return word


def calculate_hit_rate(qa_pair, retrieved_docs):
  reference_words = [simple_ru_stem(w) for w in qa_pair['answer'].split() if
                     len(w) > 3]
  if not reference_words: return 0

  for doc in retrieved_docs:
    content = doc.page_content.lower()
    found_count = 0
    for stem in reference_words:
      if stem in content:
        found_count += 1

    if found_count / len(reference_words) >= HIT_RATE_THRESHOLD:
      return 1
  return 0


def save_test_report(results, args, avg_metrics, config):
  """Сохраняет подробный отчет в Markdown."""
  output_dir = config['paths']['output_dir']
  filename = f"report_{args.chunker}_{args.index_mode}.md"
  report_path = os.path.join(output_dir, filename)
  os.makedirs(output_dir, exist_ok=True)

  with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"# 📊 Отчет о тестировании RAG\n")
    f.write(f"**Дата:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(
      f"**Конфигурация:** Chunker=`{args.chunker}`, Mode=`{args.index_mode}`, Retriever=`{args.retriever}`\n\n")

    f.write("## 📈 Сводные метрики\n")
    f.write(f"- **Hit Rate (Retrieval):** {avg_metrics['hit']:.2%}\n")
    f.write(f"- **Similarity (Generation):** {avg_metrics['sim']:.4f}\n")
    f.write(f"- **Avg Latency:** {avg_metrics['lat']:.2f}s\n\n")

    f.write("## 📝 Детализация по вопросам\n")
    for i, res in enumerate(results, 1):
      icon = "✅" if res['hit'] else "❌"
      f.write(f"### {i}. {res['q']}\n")
      f.write(f"- **Эталон:** {res['ref']}\n")
      f.write(f"- **Ответ Бота:** {res['a']}\n")
      f.write(
        f"- **Метрики:** {icon} Hit={res['hit']} | Sim={res['score']:.4f} | Time={res['latency']:.2f}s\n")
      f.write("\n---\n")

  print(f"\n📄 Подробный отчет сохранен в: {report_path}")


# --- PIPELINE ТЕСТИРОВАНИЯ ---

async def run_full_test_pipeline(args, container, config_data):
  # 1. Настройка Чанкера
  processor_name = f"{args.chunker}_processor"
  try:
    processor_provider = getattr(container, processor_name)
    container.data_processor.override(processor_provider)
  except AttributeError:
    sys.exit(f"❌ Ошибка: Процессор '{processor_name}' не найден в DI.")

  # 2. Индексация
  if args.need_index:
    print(f"\n🏗️  Запуск тестовой индексации ({args.chunker})...")
    processor = container.data_processor()
    run_indexing(config=config_data, processor=processor, mode=args.index_mode)

  # 3. Тестирование
  print(f"\n🧪 ЗАПУСК ТЕСТИРОВАНИЯ (Retriever: {args.retriever})...")
  loader = TestSetLoader(config_data['paths']['qa_test_set'])
  qa_set = loader.get_qa_pairs()
  if not qa_set: sys.exit("QA set empty")

  retrieval_chain = container.retrieval_chain()
  generation_chain = container.generation_chain()

  eval_emb = HuggingFaceEmbeddings(
      model_name=config_data.get('evaluation_model', {}).get('name',
                                                             'cointegrated/rubert-tiny2'),
      model_kwargs={'device': 'cpu'}
  )
  evaluator = load_evaluator("embedding_distance", embeddings=eval_emb)

  # Шаг 1: Batch Retrieval
  print(f"🔎 [1/2] Поиск документов для {len(qa_set)} вопросов...")
  retrieval_tasks = [retrieval_chain.ainvoke(qa['question']) for qa in qa_set]
  retrieved_docs_batch = await asyncio.gather(*retrieval_tasks)

  # Шаг 2: Sequential Generation
  print(f"🤖 [2/2] Генерация и оценка (последовательно)...")
  results = []
  total_hit_rate = 0

  for i, (qa, docs) in enumerate(zip(qa_set, retrieved_docs_batch)):
    hit = calculate_hit_rate(qa, docs)
    total_hit_rate += hit
    print(
      f"[{i + 1}/{len(qa_set)}] {qa['question'][:40]}... (Docs: {len(docs)}, Hit: {hit})")

    safe_docs = []
    current_chars = 0
    for d in docs:
      doc_len = len(d.page_content)
      if current_chars + doc_len < MAX_CONTEXT_CHARS:
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

      dist = \
      evaluator.evaluate_strings(prediction=response, reference=qa['answer'])[
        'score']
      score = 1.0 - dist

      results.append({
        "q": qa['question'],
        "a": response,
        "ref": qa['answer'],  # Сохраняем эталон
        "score": score,
        "latency": latency,
        "hit": hit  # Сохраняем Hit Rate
      })

    except Exception as e:
      print(f"  ❌ Ошибка: {e}")
      results.append(
          {"q": qa['question'], "a": f"ERROR: {e}", "ref": qa['answer'],
           "score": 0.0, "latency": 0.0, "hit": 0})

  # Отчет
  if results:
    avg_metrics = {
      "sim": sum(r['score'] for r in results) / len(results),
      "lat": sum(r['latency'] for r in results) / len(results),
      "hit": total_hit_rate / len(results)
    }

    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ ({args.chunker} | {args.index_mode})")
    print("=" * 60)
    print(f"✅ Hit Rate (Retrieval): {avg_metrics['hit']:.2%}")
    print(f"✅ Similarity (Generation): {avg_metrics['sim']:.4f}")
    print(f"⏱️  Avg Latency: {avg_metrics['lat']:.2f}s")

    # Сохранение в файл
    save_test_report(results, args, avg_metrics, config_data)

  else:
    print("❌ Нет результатов.")


# --- MAIN CLI ---

def main():
  load_dotenv()
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command")

  # TEST
  test_parser = subparsers.add_parser("test")
  test_parser.add_argument("--chunker", type=str, default="markdown",
                           choices=["markdown", "semantic", "unstructured"])
  test_parser.add_argument("--retriever", type=str, default="hybrid")
  test_parser.add_argument("--index-mode", type=str, default="test",
                           choices=["test", "full"])
  test_parser.add_argument("--force-index", action="store_true")

  # INDEX (Production)
  idx_parser = subparsers.add_parser("index")
  idx_parser.add_argument("mode", nargs="?", default="full",
                          choices=["full", "test"])
  idx_parser.add_argument("--chunker", type=str, default="markdown",
                          choices=["markdown", "semantic", "unstructured"])

  subparsers.add_parser("retrieve").add_argument("-q", "--query")
  subparsers.add_parser("answer").add_argument("-q", "--query")

  args = parser.parse_args()

  try:
    with open('config/config.yaml', 'r') as f:
      config = yaml.safe_load(f)
  except:
    sys.exit("Config missing")

  # Настройка путей для ТЕСТА
  if args.command == "test":
    base_db = config['retrievers']['vector_store'].get('db_path_base',
                                                       'data/chroma_db')
    config['retrievers']['vector_store'][
      'db_path'] = f"{base_db}_{args.chunker}"

    base_bm25 = config['retrievers']['bm25'].get('index_path_base',
                                                 'data/bm25_index')
    base, ext = os.path.splitext(base_bm25)
    config['retrievers']['bm25']['index_path'] = f"{base}_{args.chunker}{ext}"

    args.need_index = manage_db_state_for_test(config, args.force_index)

  container = Container()
  container.config.from_dict(config)

  if args.command == "test":
    asyncio.run(run_full_test_pipeline(args, container, config))

  elif args.command == "index":
    print(f"🚀 Запуск ПРОДАКШН индексации (Chunker: {args.chunker})...")

    prod_db_path = config['retrievers']['vector_store']['db_path']
    prod_bm25_path = config['retrievers']['bm25']['index_path']

    print(f"⚠️  ВНИМАНИЕ: Полная очистка текущей базы: {prod_db_path}")
    try:
      shutil.rmtree(prod_db_path)
      if os.path.exists(prod_bm25_path):
        os.remove(prod_bm25_path)
    except OSError as e:
      sys.exit(f"❌ Не удалось очистить базу: {e}\nОстановите бота и повторите.")

    try:
      processor_provider = getattr(container, f"{args.chunker}_processor")
      container.data_processor.override(processor_provider)
    except AttributeError:
      sys.exit(f"❌ Ошибка: Чанкер {args.chunker} не найден.")

    processor = container.data_processor()
    run_indexing(config, processor, args.mode)

  elif args.command == "retrieve":
    retrieval_step = container.retrieval_chain()
    if args.query:
      print(f"Поиск: {args.query}")
      docs = retrieval_step.invoke(args.query)
      for d in docs:
        print(
          f"\n--- {d.metadata.get('title', 'Doc')} ---\n{d.page_content[:200]}...")

  elif args.command == "answer":
    rag_chain = container.rag_chain()
    if args.query:
      print(f"Вопрос: {args.query}")
      print(rag_chain.invoke(args.query))


if __name__ == "__main__":
  main()