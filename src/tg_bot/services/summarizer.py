import asyncio
import logging
from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """Ты сжимаешь историю чата в краткое summary для последующей подсказки ассистенту.

Ниже диалог, где каждая строка в формате:
«Пользователь: …» или «Бот: …».

{dialogue}

Правила:
- Выдай **один** сплошной абзац из **2–4 предложений**; не пиши длинее без крайней необходимости.
- Сохрани ключевые факты, имена, сущности и обсуждаемые темы; опусти воду.
- **Язык** summary — тот же, что и у реплик (русский диалог → summary по-русски).
- **Строгий запрет на галлюцинации:** не добавляй факты, цифры, имена и выводы, которых **нет** в приведённом диалоге. Не додумывай за собеседников.
- **Не** используй JSON, списки, markdown-заголовки, кавычки вокруг всего ответа; только готовый текст summary, без пояснений «вот summary:».
"""


class SummarizerService:
  def __init__(self, llm: BaseLanguageModel, timeout: int = 90) -> None:
    self._chain = (
        PromptTemplate.from_template(SUMMARIZATION_PROMPT)
        | llm
        | StrOutputParser()
    )
    self._timeout = timeout

  async def summarize(self, messages: List[BaseMessage]) -> str:
    """
    Принимает список BaseMessage (HumanMessage + AIMessage), возвращает
    краткое текстовое summary.
    При ошибке или таймауте возвращает пустую строку "".
    """
    lines: List[str] = []
    for msg in messages:
      role = "Пользователь" if isinstance(
          msg, HumanMessage
      ) else "Бот"
      text = msg.content
      if not isinstance(text, str):
        text = str(text)
      lines.append(f"{role}: {text}")
    dialogue = "\n".join(lines)
    if not dialogue.strip():
      return ""
    try:
      raw = await asyncio.wait_for(
          self._chain.ainvoke({"dialogue": dialogue}),
          timeout=self._timeout,
      )
    except asyncio.TimeoutError:
      logger.warning("Summarization timeout")
      return ""
    except Exception as e:
      logger.error("Summarization error: %s", e)
      return ""
    if not isinstance(raw, str):
      raw = str(raw)
    return raw.strip()
