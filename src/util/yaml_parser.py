import yaml

def load_qa_test_set(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            test_set = yaml.safe_load(f)
        print(f"✅ Успешно загружено {len(test_set)} пар вопрос-ответ из {file_path}")
        return test_set
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {file_path} не найден.")
        return None
    except Exception as e:
        print(f"❌ Ошибка при чтении YAML файла: {e}")
        return None