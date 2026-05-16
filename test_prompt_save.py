import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from src.pipelines.rag.pipeline import _save_prompt_to_file

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}\n\nContext:\n{context}"),
])

inputs = {
    "input": "What is the meaning of life?",
    "chat_history": [],
    "context": [Document(page_content="42 is the answer.")]
}

_save_prompt_to_file(inputs, qa_prompt)
