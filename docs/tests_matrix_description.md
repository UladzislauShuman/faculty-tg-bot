# Описание сценариев матрицы тестов

Сценарии задаются в [`config/config.yaml`](../config/config.yaml) в секции `evaluation_scenarios` и выполняются командой **`python main.py test-matrix`** (или пункт **3** в [`test.sh`](../test.sh)).

Ниже четыре сценария. Первый сопоставлен с **быстрым запуском пункта 1** в `test.sh`; остальные меняют только то, что указано явно — всё остальное наследуется из `config.yaml`.

---

## Сводная таблица

| № | Имя в YAML (`name`) | Ретривер | Память диалога | Соответствие `test.sh` |
|---|---------------------|----------|----------------|-------------------------|
| 1 | `baseline_like_test_sh` | Chroma + BM25 (`chroma_bm25`) | как в конфиге: `summary_window` | Пункт 1: `all`, `markdown`, `hybrid`, `test`, **force-index**, `chroma_bm25`, `summary_window` |
| 2 | `qdrant` | Qdrant (`active_type: qdrant`) | как в конфиге (обычно `summary_window`) | Только замена backend поиска на Qdrant |
| 3 | `qdrant_summary_window` | Qdrant | явно `memory.type: summary_window`, память вкл. | Qdrant + режим суммарирующего окна |
| 4 | `qdrant_memory_window` | Qdrant | `memory.type: window` без LLM-summary | Тот же Qdrant, «чистое» скользящее окно |

---

## 1. `baseline_like_test_sh` — базовый, как пункт 1 в `test.sh`

В скрипте [`test.sh`](../test.sh) для пункта **1** задано:

```bash
EVAL_MODE=all
CHUNKER=markdown
RETRIEVER=hybrid
INDEX_MODE=test
FORCE_INDEX=--force-index
ACTIVE_TYPE=chroma_bm25
MEMORY_TYPE=summary_window
```

Эквивалент CLI:

```text
docker-compose exec rag-cli python main.py test all \
  --chunker=markdown --retriever=hybrid --index-mode=test --force-index \
  --active-type=chroma_bm25 --memory-type=summary_window
```

В матрице это повторяется так: `chunker: markdown`, `index_mode: test`, **`force_index: true`** (как **`--force-index`**), переопределения ретривера и памяти **нет** (остаются значения базового конфига, в т.ч. `retrievers.active_type: chroma_bm25` и секция `memory`).

Для набора кейсов используется `evaluation_settings.mode` из конфига (по умолчанию обычно `all`), аналогично тому, как при вызове `python main.py test all` задаётся позиционный аргумент `eval_mode=all`.

---

## 2. `qdrant`

Только смена векторного backend на **Qdrant**: в `overrides` выставлено `retrievers.active_type: qdrant`. Параметры коллекции, `search_k` для Qdrant и прочее берутся из текущего `config.yaml` (`retrievers.qdrant.*`).

Перед прогоном нужен доступный сервис Qdrant с хоста/портом из конфигурации (часто `host: qdrant` при запуске внутри Docker Compose).

---

## 3. `qdrant_summary_window`

Тот же **Qdrant**, что в сценарии 2, плюс **явная** фиксация памяти:

- `memory.enabled: true`
- `memory.type: summary_window`

Имеет смысл для многошаговых кейсов из `qa-test-set.yaml`: один backend поиска, проверка **summary window**.

---

## 4. `qdrant_memory_window`

Снова **Qdrant**; память — **`memory.type: window`** (скользящее окно без LLM-суммаризации). Удобно сравнивать сценарии **3** и **4** при одном backend.

---

## Замечания

- Индекс **Chroma + BM25** привязан к `chunker` (например, `data/chroma_db_markdown`). С одинаковым `chunker` и без `force_index: true` индекс может переиспользоваться между шагами матрицы. Базовый сценарий с **`force_index: true`** каждый полный матричный запуск заново переиндексирует локальную пару Chroma+B25 для этого чанкера (поведение ближе к привычному прогону из `test.sh` пункта 1).
- Наполнение коллекции **Qdrant** — по вашей обычной процедуре индексации; если коллекции нет или она старая, результаты сценариев 2–4 нужно трактовать вместе с тем, как и когда выполнялась индексация под Qdrant.
