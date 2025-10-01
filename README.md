# faculty-tg-bot
Телеграм Бот, который отвечает на вопросы о моем факультете, основываясь на официальном сайте, телеграмм каналах.
Использует архитектуру RAG

# команды для запуска
все команды выполняются из корневой папки (faculty-tg-bot)
## команды инициализации
- установить все зависимости `pip install -r requirements.txt`

## команды для проверки работоспособности 
### indexing
- `python main.py index`
### retrieve
- `python main.py retrieve` -- по всем вопросам из qa-test-set.yaml и вывод в output/run_retriever-output.txt
- `python main.py retrieve --query "какие научные школы существуют на факультете?"` -- по конкретному вопросу, все выводиться в консоль
### full rag
- `python main.py answer` -- по всем вопросам из qa-test-set.yaml и вывод в output/run_bot-output.txt
- `python main.py answer --query "кто был первым деканом?"` -- по конкретному вопросу, все выводиться в консоль