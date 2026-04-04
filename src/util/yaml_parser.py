import yaml
import logging

logger = logging.getLogger(__name__)


class TestSetLoader:
  def __init__(self, file_path: str):
    self.file_path = file_path
    self._data = self._load_yaml()

  def _load_yaml(self):
    try:
      with open(self.file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or []
    except FileNotFoundError:
      logger.error(f"❌ Файл {self.file_path} не найден.")
      return []
    except Exception as e:
      logger.error(f"❌ Ошибка чтения YAML: {e}")
      return []

  def get_test_urls(self) -> list[str]:
    """Возвращает список URL, у которых enabled: true"""
    return [
      item['url']
      for item in self._data
      if item.get('enabled', True) and 'url' in item
    ]

  def get_qa_pairs(self) -> list[dict]:
    """Возвращает плоский список вопросов для тестирования"""
    qa_list = []
    for item in self._data:
      if not item.get('enabled', True):
        continue

      for q_item in item.get('questions', []):
        # Если у вопроса нет флага enabled или он true
        if q_item.get('enabled', True):
          qa_list.append({
            "question": q_item['q'],
            "answer": q_item['a'],
            "source_url": item.get('url')  # Полезно для дебага
          })

    logger.info(
      f"✅ Загружено {len(qa_list)} тестовых вопросов из {self.file_path}")
    return qa_list

  def get_test_scenarios(self) -> list[dict]:
    """Возвращает список многошаговых сценариев (диалогов)"""
    scenarios = []
    for item in self._data:
      if not item.get('enabled', True):
        continue

      if 'scenarios' in item:
        for scenario in item['scenarios']:
          scenarios.append({
            "url": item.get('url'),
            "name": scenario['name'],
            "steps": scenario['steps']
          })

    logger.info(
      f"✅ Загружено {len(scenarios)} тестовых сценариев из {self.file_path}")
    return scenarios