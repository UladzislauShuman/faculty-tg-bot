# FPMI RAG Bot

Телеграм-бот с RAG для ответов о **Факультете прикладной математики и информатики (ФПМИ) БГУ**: поиск по базе знаний с сайта и генерация ответа LLM.

**Стек (в коде):** Python 3.11+, **LangChain**, **Aiogram 3.x**, **dependency-injector**, **PostgreSQL**, **ChromaDB** или **Qdrant** (гибрид dense + sparse), **BM25** (для ветки `chroma_bm25`), опциональный **Cross-Encoder reranker**, опциональный **HyDE** для dense-запросов, **семантический роутинг** до RAG.

Параметры экспериментов и окружения задаются в **`config/config.yaml`** (и частично через CLI / `.env`).

## Содержание

- [Быстрый старт](#быстрый-старт-macos--linux)
- [Ручной запуск](#ручной-запуск-если-скрипт-не-сработал)
- [Функционал и архитектура](#функционал-и-архитектура)
- [CLI](#cli-команды)
- [Полезные команды](#полезные-команды)
- [Тестирование и матрица сценариев](#тестирование-и-эксперименты)
- [Метрики в evaluation](#метрики-в-evaluation)
- [Логи и отладка](#логи-и-отладка)
- [Полезные ссылки](#полезные-ссылки)
- [Идеи и бэклог](#идеи-и-бэклог)

## Быстрый старт (macOS / Linux)

Нужны **Docker** и **Docker Compose** (`docker-compose` или `docker compose`). Первый запуск может быть долгим (сборка образов, скачивание моделей, индексация).

Можно использовать **нативную Ollama** на хосте (часто быстрее GPU) — см. [.env.example](.env.example) и [setup.sh](setup.sh).

### 1. Окружение

1. Склонируйте репозиторий.
2. Создайте `.env`:
   ```bash
   cp .env.example .env
   ```
3. Заполните минимум **`TGSERVER__TOKEN`**. Для webhook укажите **`TGSERVER__WEBHOOK_URL`** (см. ниже).
4. **LLM:** по умолчанию `LLM_PROVIDER=ollama`. Для Ollama **внутри Docker** в `.env` обычно `OLLAMA_HOST=http://ollama:11434`; для **Ollama на хосте** — `OLLAMA_HOST=http://host.docker.internal:11434` (на Linux может понадобиться другой адрес).
5. **Эмбеддинги для RAG** берутся из **`config/config.yaml` → `embedding_model.name`** (не из переменной `EMBEDDING_MODEL_NAME` в `.env`, если вы явно не дублируете логику в своём окружении). Пример в `.env.example` может отличаться от yaml — ориентируйтесь на конфиг репозитория.
6. Опционально **LangSmith:** `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` и т.д.

### 2. Автоматическая установка

```bash
chmod +x setup.sh
./setup.sh
```

Скрипт проверяет Ollama, поднимает сервисы (в т.ч. **Postgres**, **Qdrant**, **bot**, **rag-cli**; при выборе Docker-режима — и **ollama** с профилем), выполняет миграции и индексацию по сценарию скрипта.

### 3. Webhook (localtunnel / аналог)

Бот слушает **порт 8080** в контейнере (проброшен наружу в [docker-compose.yml](docker-compose.yml)).

1. В отдельном терминале (нужен Node.js):
   ```bash
   npx localtunnel --port 8080 --subdomain famcs-answer-bot
   ```
   Если поддомен занят — уберите `--subdomain` и используйте выданный URL.

2. В `.env` укажите (обязательно суффикс **`/webhook`**):
   ```ini
   TGSERVER__WEBHOOK_URL=https://famcs-answer-bot.loca.lt/webhook
   ```

3. Перезапуск бота:
   ```bash
   docker-compose restart bot
   ```

Остановка стека: `docker-compose down` (флаг `-v` удалит именованные volumes — используйте осознанно).

**Порты по умолчанию:** приложение бота **8080**, Postgres с хоста **5433** → 5432 в контейнере, Qdrant **6333**, Ollama (если профиль) **11434**.

## Ручной запуск (если скрипт не сработал)

Ориентир — логика [setup.sh](setup.sh).

1. **Сервисы** (без Ollama в Docker, как в native-режиме):
   ```bash
   docker-compose up -d --build bot rag-cli postgres qdrant
   ```
   С Ollama в Docker:
   ```bash
   docker-compose --profile ollama-docker up -d --build
   ```

2. **Миграции:**
   ```bash
   docker-compose exec rag-cli alembic upgrade head
   ```

3. **Индексация** (иначе ответы по базе будут пустыми):
   - быстрый режим — URL из **`qa-test-set.yaml`** (секция URL для `test`, см. `TestSetLoader` / пайплайн):
     ```bash
     docker-compose exec rag-cli python main.py index test
     ```
   - полный обход с **`data_source.url`** и `max_depth` в `config.yaml`:
     ```bash
     docker-compose exec rag-cli python main.py index
     ```
     или явно `python main.py index full`.

   Чанкер по умолчанию для `index` задаётся **`indexing.chunker`** в yaml; переопределение:  
   `python main.py index test --chunker markdown` (варианты: `markdown`, `semantic`, `unstructured`, `parent`).

## Функционал и архитектура

1. **Индексация:** краулер или фиксированный список URL → парсинг HTML (несколько процессоров) → чанки → эмбеддинги (**модель из `embedding_model`**, для E5 в коде добавляются префиксы `passage:` / `query:`) → запись в **Chroma + BM25** или **Qdrant hybrid** в зависимости от **`retrievers.active_type`**. При **`parent_document.enabled`** дополнительно поддерживается связка parent/child (см. конфиг).

2. **Retrieval:** гибрид вектор + sparse (BM25 или Qdrant sparse), опционально **reranker** (`retrievers.reranker.enabled` и `top_n` в конфиге). В истории диалога может участвовать **reformulation** (уточнение запроса с учётом истории), а не отдельный «query expansion синонимами».

3. **HyDE (опционально):** перед dense-`embed_query` LLM генерирует гипотетический фрагмент; управляется **`hyde`** в yaml и флагами CLI **`--hyde on|off`** для `test`, `retrieve`, `answer`. Детальный вывод HyDE в консоль: `hyde.verbose_console` или env **`HYDE_CONSOLE=1`**.

4. **Generation:** LLM провайдера из **`LLM_PROVIDER`** (`ollama`, `yandex_gpt`, …) и блока `providers` в yaml.

5. **Telegram:** Aiogram 3, webhook, DI-контейнер, история в **PostgreSQL**, память **`memory`** (`window` / `summary_window`), команда **`/newchat`**.

6. **Семантический роутинг:** до тяжёлого RAG сообщение может классифицироваться (`semantic_routing` в yaml: `smalltalk`, `direct_link`, `rag`).

## CLI команды

Команды выполняются в контейнере **`rag-cli`** (рабочая директория проекта смонтирована в `/app`):

```bash
docker-compose exec rag-cli python main.py <команда> ...
```

### Индексация

| Команда | Описание |
|--------|----------|
| `index` | Режим по умолчанию **`full`** — краулер по `data_source`. |
| `index test` | Только URL из **`qa-test-set.yaml`**. |
| `index … --chunker <name>` | `markdown`, `semantic`, `unstructured`, `parent`. |

Пайплайн индексации очищает текущие пути Chroma/BM25 из конфига и пересоздаёт Qdrant-коллекцию при выборе Qdrant (см. код `main.py` / `run_indexing`).

Дамп чанков для ручной проверки: **`output/`** + имя из **`paths.indexing`** (по умолчанию `indexing.txt`).

### Retrieve (только поиск)

```bash
docker-compose exec rag-cli python main.py retrieve -q "Ваш вопрос"
docker-compose exec rag-cli python main.py retrieve   # пакетно → файл из paths.retriever
```

Опция: **`--hyde on|off`**.

### Answer (полный RAG)

```bash
docker-compose exec rag-cli python main.py answer -q "Вопрос"
docker-compose exec rag-cli python main.py answer              # пакетно → paths.run_bot
docker-compose exec rag-cli python main.py answer questions    # режим из CLI → evaluation_settings.mode
```

Опция: **`--hyde on|off`**.

### test (E2E evaluation)

```bash
docker-compose exec rag-cli python main.py test [all|questions|scenarios] [опции]
```

Основные опции:

| Опция | Значения | Смысл |
|--------|-----------|--------|
| `--chunker` | `markdown`, `semantic`, `unstructured`, `parent` | Чанкер и суффикс путей `chroma_db_<chunker>`, BM25. |
| `--retriever` | например `hybrid` | Стратегия на уровне CLI (см. код контейнера). |
| `--index-mode` | `test`, `full` | Источник URL при доиндексации в тесте. |
| `--force-index` | флаг | Удалить локальные индексы для выбранного chunker и переиндексировать. |
| `--active-type` | `chroma_bm25`, `qdrant` | Переопределить `retrievers.active_type`. |
| `--memory-type` | `summary_window`, `window` | Переопределить `memory.type`. |
| `--memory-off` | флаг | `memory.enabled: false`. |
| `--hyde` | `on`, `off` | Переопределить `hyde.enabled`. |

Артефакты: **`output/trace_*.md`**, **`output/report_*.md`** (имя зависит от chunker, active_type, memory, HyDE и т.д., см. `TestPipelineRunner._output_label_stem`).

### test-matrix

Матрица сценариев из **`evaluation_scenarios`** в `config.yaml`, чекпоинт **`paths.default_checkpoint_path`**, пауза **`paths.pause_flag`**.

```bash
docker-compose exec rag-cli python main.py test-matrix
docker-compose exec rag-cli python main.py test-matrix --resume
```

Интерактивно: [test.sh](test.sh), возобновление: [resume.sh](resume.sh).

## Полезные команды

```bash
docker-compose logs -f bot
docker-compose down
docker-compose exec rag-cli alembic upgrade head
```

Автогенерация миграций (`alembic revision --autogenerate`) используйте только когда меняете модели БД и понимаете diff — не как рутинный шаг при каждом запуске.

## Тестирование и эксперименты

### Сравнение моделей (A/B тестирование)
В скрипте `test.sh` встроена поддержка профилей моделей. Вы можете запускать тесты на старом стеке моделей, на новом, или на том, что прописан в `config.yaml`, не меняя сам конфиг.

Для переключения профиля используйте переменную окружения `RAG_MODEL_PROFILE`:

```bash
# Запустить тесты на старом стеке (llama3.1, e5-large, DiTy reranker)
RAG_MODEL_PROFILE=prev ./test.sh

# Запустить тесты на новом стеке (qwen2.5, bge-m3, bge-reranker-v2)
RAG_MODEL_PROFILE=new ./test.sh

# Запустить тесты строго по настройкам из config.yaml
RAG_MODEL_PROFILE=yaml ./test.sh
```

Также можно переопределять отдельные модели прямо при вызове Docker:
```bash
docker-compose exec -e RAG_OLLAMA_MODEL=mistral rag-cli python main.py test all
```
*Примечание: Если вы меняете модель эмбеддингов (`RAG_EMBEDDING_MODEL`), обязательно включайте переиндексацию (`--force-index`), иначе поиск по старой базе сломается.*

Минимальный стек для прогонов без Telegram-бота: **Postgres**, **Qdrant**, **rag-cli**, плюс доступная **LLM** (контейнер `ollama` с профилем или нативная Ollama на хосте с корректным `OLLAMA_HOST`).

Пример:

```bash
docker-compose up -d --build postgres qdrant rag-cli
# при необходимости: docker-compose --profile ollama-docker up -d ollama
docker-compose exec rag-cli alembic upgrade head
```

Детали матрицы, паузы **SIGINT/SIGTERM** и `pause.flag` — см. раздел **test-matrix** выше и комментарии в `config.yaml`.

## Метрики в evaluation

Метрики ниже относятся к прогону **`python main.py test`** и итоговому отчёту **`output/report_*.md`** / **`trace_*.md`**.

### Hit / Hit Rate

**Что это:** не классическая IR-метрика, а **эвристика на ретривер** в `TestPipelineRunner._calculate_hit_rate`.

На каждом шаге:

1. Берётся **эталонный ответ** из `qa-test-set.yaml`.
2. Из него выделяются слова длиной **> 3** символов с грубым русским стеммингом.
3. Для **каждого** найденного чанка считается доля этих слов, которые **встречаются в тексте чанка** (подстрока, нижний регистр).
4. Если **хотя бы у одного** чанка эта доля **≥ порога** (`HIT_RATE_THRESHOLD`, сейчас **0.4**, то есть 40%), шаг получает **hit = 1**, иначе **0**.

**Hit Rate** в сводке — среднее этих нулей и единиц по всем шагам (одиночные вопросы + шаги сценариев), в отчёте обычно показывается как процент.

**Как интерпретировать:** **больше Hit Rate → лучше** в смысле «в выдаче чаще есть текст с заметным пересечением по словам с эталоном». Учитывайте ограничения: формулировка эталона может отличаться от сайта; стемминг грубый; порог 40% условный. Это сигнал про **совпадение с эталоном**, а не про абсолютную правильность ответа бота.

### Score (aggregate)

В трейсе поле **Score**, в сводке отчёта — **Score (judge, среднее по шагам)** (во внутреннем коде усреднение лежит в поле `sim`).

**Что это:** на каждом шаге считается агрегат из **LLM-as-a-Judge**, если включено **`evaluation_metrics.enabled`** и подключены соответствующие судьи:

- включены **faithfulness** и **answer relevance** → **Score на шаге = среднее двух оценок судьи** (каждая в диапазоне **0…1**);
- включена только одна из метрик → **Score = эта одна оценка**;
- судья выключен или оценки нет → для шага агрегат может быть **0**.

Итоговое число в отчёте — **среднее Score по шагам**.

**Как интерпретировать:** **больше Score → лучше** по мнению **конкретной** модели-судьи и промптов текущего прогона. Между разными моделями судьи или после смены промптов **абсолютные значения сравнивать осторожно** — «шкала» может сдвигаться.

### Faithfulness

**Смысл:** насколько **ответ опирается на переданный судье контекст** (retrieved чанки), без выдуманных относительно этого контекста фактов.

- шкала **0.0…1.0**;
- **выше → лучше** (судья видит опору на источник);
- **ниже → хуже** (риск галлюцинаций или фактов «мимо» контекста).

В отчёте к оценке добавлено поле **`reason`** (краткое объяснение по-русски) — его стоит смотреть при спорной цифре.

**Важно:** в судью попадает **усечённый** контекст (лимит символов в раннере, `MAX_CONTEXT_CHARS`). Faithfulness оценивается **относительно того фрагмента**, который реально ушёл в промпт судье, а не относительно всей базы.

### Relevance

**Смысл:** насколько **смысл ответа** соответствует **вопросу** (по теме, полноте).

- шкала **0.0…1.0**;
- **выше → лучше**;
- **ниже → хуже** (уход в сторону, общие фразы, нет запрошенной конкретики — в границах того, как это зашито в промпт судьи).

### Avg Faithfulness / Avg Relevance

Средние по тем шагам, где соответствующая оценка присутствует (при включённом судье обычно по всем шагам). **Больше — лучше**, с теми же оговорками про субъективность LLM-judge.

### Latency

Время **генерации ответа** на шаг (секунды), в сводке — **среднее по шагам**.

**Как интерпретировать:** для UX **меньше обычно лучше**. Если сравниваете два конфига: при сопоставимом **Score** предпочтительнее **меньшая** задержка; рост **Score** при росте latency часто нормален (судья, rerank, HyDE, тяжелее модель и т.д.).

### Как согласовывать метрики между собой

| Ситуация | Как читать |
|----------|------------|
| Hit Rate ↑, Score ↓ | В выдаче чаще «цепляется» формулировка эталона, но итоговый ответ судья оценивает ниже (слабая релевантность вопросу или слабая опора на контекст). |
| Hit Rate ↓, Score ↑ | Мало пересечения слов с эталоном в чанках, но ответ судье нравится (другая формулировка той же мысли или эвристика hit промахнулась). |
| Faithfulness ↑, Relevance ↓ | Ответ «прижат» к тексту базы, но **не тот**, что нужен по вопросу (не тот кусок выдали или ответ слишком общий). |
| Relevance ↑, Faithfulness ↓ | По смыслу ближе к вопросу, но судья видит **натягивание** или факты не из выданного контекста. |
| Почти всё ↑, кроме latency | Типичный компромисс качество ↔ время/стоимость. |

### evaluation_model в config.yaml

В **`evaluation_model`** задаётся лёгкая модель эмбеддингов для **вспомогательных** целей в проекте (не путать с **`embedding_model`** для индексации). **Сводный Score** в отчёте `test` строится из **LLM-judge** (`evaluation_metrics`), а не из этой модели напрямую. Актуальный id см. в `config.yaml`.

## Логи и отладка

- В **`main.py`** по умолчанию `logging.basicConfig(level=WARNING)`; этапы пайплайна с префиксом **`[TIMING]`** для пакетов `src.pipelines.rag.*` поднимаются в **INFO**, если не отключено env **`RAG_TIMING_LOGS=0`**.
- HyDE в консоль: **`hyde.verbose_console: true`** или **`HYDE_CONSOLE=1`**.
- Многие модули (evaluation, chunkers, HyDE) пишут структурированные сообщения через `logging` — при отладке поднимите уровень для нужного logger или всего `src`.

## Полезные ссылки

- [LangSmith](https://smith.langchain.com) — трассировка LangChain (при настройке API ключа).

## Идеи и бэклог

Ниже — черновые идеи и заметки; часть уже отражена в коде (HyDE, Qdrant, роутинг, parent-child, reranker), часть может быть устаревшей относительно текущего `main`/`config`.

- HyDE / multi-query / декомпозиция запросов  
- Семантическая маршрутизация (в продукте — `semantic_routing`)  
- Parent-child / small-to-big  
- Reranker (Cross-Encoder, в конфиге `retrievers.reranker`)  
- Self-RAG, semantic cache, RRF (частично закрыто гибридом)  
- Нормализация чисел и сокращений в запросах  
- Профилирование и выбор LLM под учебный контур  

### Вопросы для дальнейшей проработки

- Унифицировать обработку чисел («три» vs «3»).  
- Нужен ли пользователю режим «deep thinking» и как его безопасно внедрить.  
- Специализированные / дообученные модели под университетские QA.
- как проработать логику бота

### Про тесты

- В названиях артефактов / прогонов иметь метку времени и короткий комментарий для поиска.

### Технический долг (примеры)

- Централизация конфиг-флагов (частично сделано через `config.yaml`).  
- Общие куски shell-скриптов.  
- Дубликаты документов между бэкендами при экспериментах.  
- Полировка метрик и отчётов (иконки, несколько метрик).  
- Качество на аббревиатурах, таблицах и списках.

## Примерный план (ориентир)

**Цель на семестр (в т.ч. февраль–июнь 2026):**

- Разбор узких мест по latency и стоимости вызовов.  
- Закрепить продовую LLM (облако или стабильная локальная связка).  
- Итерации по чанкингу и eval-матрице.  
- Улучшить использование диалогового контекста памятью и промптами.

**По возможности:** расширить классификацию запросов и сценарии малой кровью через `config.yaml` и матрицу.
