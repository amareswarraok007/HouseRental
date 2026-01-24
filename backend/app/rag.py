from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class IncidentRAG:
    def __init__(self, docs_path: Path, persist_path: Path) -> None:
        self.docs_path = docs_path
        self.persist_path = persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = self._select_embeddings()
        self.collection_name = "incident_knowledge_base"
        self.schema = {
            "content": "text",
            "metadata": {
                "service": "string",
                "incident": "string",
                "source": "string",
                "timestamp": "string",
            },
        }

    def _select_embeddings(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        return FakeEmbeddings(size=768)

    def _load_documents(self):
        docs = []
        for doc_path in sorted(self.docs_path.glob("*.txt")):
            loader = TextLoader(str(doc_path))
            docs.extend(loader.load())
        return docs

    def get_store(self) -> Chroma:
        docs = self._load_documents()
        return Chroma.from_documents(
            docs,
            embedding=self.embeddings,
            persist_directory=str(self.persist_path),
            collection_name=self.collection_name,
        )

    def query(self, query_text: str, k: int = 3) -> List[str]:
        store = self.get_store()
        results = store.similarity_search(query_text, k=k)
        return [result.page_content for result in results]

    def add_document(self, content: str, metadata: Dict[str, Any]) -> None:
        store = self.get_store()
        store.add_texts([content], metadatas=[metadata])
