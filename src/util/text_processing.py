import re
from typing import List
from nltk.stem.snowball import SnowballStemmer

# Инициализируем стеммер один раз при импорте модуля
_stemmer = SnowballStemmer("russian")


def tokenize_for_bm25(text: str) -> List[str]:
  """
  Очищает текст, разбивает на слова и применяет стемминг.
  Используется как preprocess_func для BM25Retriever.
  """
  if not text:
    return []

  # Приводим к нижнему регистру и достаем только слова (кириллица/латиница/цифры)
  words = re.findall(r'\b\w+\b', text.lower())

  # Применяем стемминг и отбрасываем слишком короткие слова (предлоги)
  return [_stemmer.stem(w) for w in words if len(w) > 2]