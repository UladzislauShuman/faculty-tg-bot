"""Семантический роутинг для бота: smalltalk / direct_link / rag (regex или LLM)."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

RouteLabel = Literal["smalltalk", "direct_link", "rag"]

_DEFAULT_SMALLTALK = (
    "Здравствуйте! Я бот ФПМИ БГУ. Задайте вопрос о факультете — отвечу по "
    "официальным материалам из базы знаний."
)
_DEFAULT_DIRECT_LINK = (
    "Официальный сайт факультета: https://fpmi.bsu.by\n"
    "Актуальное расписание и объявления ищите в разделах сайта "
    "«Расписание» / «Новости»."
)

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты классификатор намерений пользователя. Ответь ровно одним словом "
            "из набора: smalltalk, direct_link, rag. Без пояснений.",
        ),
        (
            "human",
            "Категории:\n"
            "- smalltalk: приветствия, благодарность без фактического вопроса, "
            "прощание, «как дела» без запроса данных о факультете\n"
            "- direct_link: просьба дать ссылку, сайт, URL; где посмотреть "
            "расписание; официальный сайт\n"
            "- rag: любые фактические вопросы о факультете, декане, баллах, "
            "поступлении, правилах; также уточняющие вопросы по уже обсуждаемым "
            "фактам («сколько это лет?», «к какому году?»)\n\n"
            "Сообщение: {query}\n"
            "Класс:",
        ),
    ]
)

# Сначала «ссылка / расписание», затем короткий smalltalk (см. порядок в classify_regex).
_DIRECT_HINTS = re.compile(
    r"(расписан|расписани|ссылк|\burl\b|https?://|www\.|"
    r"где\s+(посмотреть|найти|искать|открыть)|"
    r"официальн\w*\s+сайт|сайт\s+факультет|сайт\s+фпми|"
    r"(скинь|кинь|дай|порекомендуй|подскажи)\s+(ссыл|сайт|url)|"
    r"деканат\w*\s*\??\s*$)",
    re.IGNORECASE | re.UNICODE,
)

_SMALLTALK_STRICT = re.compile(
    r"^\s*("
    r"привет\w*|здравствуй\w*|здрасьте|"
    r"добрый\s+(день|вечер|утро|полдень)|"
    r"доброго\s+(день|вечера|утра)|"
    r"(большое\s+)?спасибо\w*|благодарю|пасиб|"
    r"пока\w*|до\s+свидания|прощай\w*|"
    r"как\s+дела|что\s+нового|как\s+ты\b|как\s+жизнь|"
    r"hi\b|hello\b|hey\b|thanks\b|thank\s+you\b"
    r")\s*[!?.…,]*\s*$",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class RoutingDecision:
  """use_rag=False: ответ без поиска по базе; при use_llm_for_reply — LLM + история."""

  use_rag: bool
  answer: Optional[str]
  non_rag_label: Optional[Literal["smalltalk", "direct_link"]] = None
  use_llm_for_reply: bool = False


@runtime_checkable
class SemanticRoutingPort(Protocol):
    async def route(self, query: str) -> RoutingDecision: ...


class PassthroughSemanticRouting:
  async def route(self, query: str) -> RoutingDecision:
    _ = query
    return RoutingDecision(use_rag=True, answer=None)


class RegexSemanticRouting:
  def __init__(
      self,
      smalltalk_reply: str,
      direct_link_reply: str,
      *,
      use_llm_reply: bool = True,
  ) -> None:
    self._smalltalk_reply = smalltalk_reply
    self._direct_link_reply = direct_link_reply
    self._use_llm_reply = use_llm_reply

  def _label(self, query: str) -> RouteLabel:
    t = (query or "").strip()
    if not t:
      return "rag"
    tl = t.lower()
    if _DIRECT_HINTS.search(tl):
      return "direct_link"
    if _SMALLTALK_STRICT.match(tl):
      return "smalltalk"
    return "rag"

  async def route(self, query: str) -> RoutingDecision:
    label = self._label(query)
    logger.info("[semantic_routing] method=regex label=%s", label)
    if label == "smalltalk":
      return RoutingDecision(
          use_rag=False,
          answer=self._smalltalk_reply,
          non_rag_label="smalltalk",
          use_llm_for_reply=self._use_llm_reply,
      )
    if label == "direct_link":
      return RoutingDecision(
          use_rag=False,
          answer=self._direct_link_reply,
          non_rag_label="direct_link",
          use_llm_for_reply=self._use_llm_reply,
      )
    return RoutingDecision(use_rag=True, answer=None)


def _normalize_llm_route(raw: str) -> RouteLabel:
    token = (raw or "").strip().lower().split()
    if not token:
        return "rag"
    word = re.sub(r"[^a-z_]", "", token[0])
    if word in ("smalltalk", "direct_link", "directlink"):
        return "smalltalk" if word == "smalltalk" else "direct_link"
    if word == "rag":
        return "rag"
    if "small" in word or "talk" in word:
        return "smalltalk"
    if "direct" in word or "link" in word:
        return "direct_link"
    return "rag"


class LlmSemanticRouting:
  def __init__(
      self,
      chain: Runnable,
      smalltalk_reply: str,
      direct_link_reply: str,
      timeout_seconds: float,
      *,
      use_llm_reply: bool = True,
  ) -> None:
    self._chain = chain
    self._smalltalk_reply = smalltalk_reply
    self._direct_link_reply = direct_link_reply
    self._timeout_seconds = timeout_seconds
    self._use_llm_reply = use_llm_reply

  async def route(self, query: str) -> RoutingDecision:
    label: RouteLabel = "rag"
    try:
      raw = await asyncio.wait_for(
          self._chain.ainvoke({"query": query}),
          timeout=self._timeout_seconds,
      )
      label = _normalize_llm_route(str(raw))
    except asyncio.TimeoutError:
      logger.warning(
          "[semantic_routing] method=llm timeout=%ss — fallback rag",
          self._timeout_seconds,
      )
    except Exception as exc:
      logger.warning(
          "[semantic_routing] method=llm error=%s — fallback rag",
          exc,
      )
    logger.info("[semantic_routing] method=llm label=%s", label)
    if label == "smalltalk":
      return RoutingDecision(
          use_rag=False,
          answer=self._smalltalk_reply,
          non_rag_label="smalltalk",
          use_llm_for_reply=self._use_llm_reply,
      )
    if label == "direct_link":
      return RoutingDecision(
          use_rag=False,
          answer=self._direct_link_reply,
          non_rag_label="direct_link",
          use_llm_for_reply=self._use_llm_reply,
      )
    return RoutingDecision(use_rag=True, answer=None)


def create_semantic_routing_service(config: Any) -> SemanticRoutingPort:
    """Собирает сервис по корню config (ожидается dict из config.yaml)."""
    cfg_dict = config if isinstance(config, dict) else {}
    block = cfg_dict.get("semantic_routing")
    if not isinstance(block, dict):
        block = {}
    if not block.get("enabled", False):
        return PassthroughSemanticRouting()

    method = str(block.get("method", "regex")).lower().strip()
    smalltalk = str(block.get("smalltalk_reply") or _DEFAULT_SMALLTALK)
    direct = str(block.get("direct_link_reply") or _DEFAULT_DIRECT_LINK)

    use_llm_reply = bool(block.get("use_llm_reply", True))

    if method == "regex":
        return RegexSemanticRouting(
            smalltalk,
            direct,
            use_llm_reply=use_llm_reply,
        )
    if method == "llm":
        # Ленивый импорт: pipeline тянет history/БД — для regex не нужен.
        from src.pipelines.rag.pipeline import get_llm_from_config

        llm_cfg = block.get("llm")
        if not isinstance(llm_cfg, dict):
            raise ValueError("semantic_routing.method=llm требует секцию semantic_routing.llm")
        llm = get_llm_from_config(llm_cfg)
        chain = _ROUTER_PROMPT | llm | StrOutputParser()
        timeout = float(block.get("timeout_seconds", 20))
        return LlmSemanticRouting(
            chain,
            smalltalk,
            direct,
            timeout,
            use_llm_reply=use_llm_reply,
        )
    raise ValueError(
        f"semantic_routing.method must be 'regex' or 'llm', got: {method!r}"
    )
