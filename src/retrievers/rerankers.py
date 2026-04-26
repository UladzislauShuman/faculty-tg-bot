from typing import Optional
from langchain_core.documents import BaseDocumentCompressor
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def create_reranker(config: dict) -> Optional[BaseDocumentCompressor]:
  """
  Создает компрессор документов (реранкер).
  Возвращает None, если реранкер отключен в конфиге.
  """
  reranker_conf = config['retrievers'].get('reranker', {})

  if not reranker_conf.get('enabled', True):
    print("ℹ️ Reranker отключен в конфигурации.")
    return None

  print(f"Инициализация реранкера ({reranker_conf.get('model')})...")
  cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_conf['model'])

  return CrossEncoderReranker(
      model=cross_encoder,
      top_n=reranker_conf['top_n']
  )