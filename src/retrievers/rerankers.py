from typing import Optional

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import BaseDocumentCompressor

import logging

logger = logging.getLogger(__name__)


def create_reranker(config: dict) -> Optional[BaseDocumentCompressor]:
  """Cross-encoder reranker или None, если reranker.enabled ложь в конфиге."""
  reranker_conf = config['retrievers'].get('reranker', {})

  if not reranker_conf.get('enabled', True):
    logger.info("Reranker выключен в конфиге")
    return None

  model = reranker_conf.get('model')
  logger.info("Reranker: загрузка модели %s top_n=%s", model, reranker_conf.get('top_n'))
  cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_conf['model'])

  return CrossEncoderReranker(
      model=cross_encoder,
      top_n=reranker_conf['top_n']
  )