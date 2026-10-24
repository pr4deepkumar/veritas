from __future__ import annotations

from veritas.core.clients.pinecone import PineconeStore


class DocumentSelectionAgent:
    """Lists the indexed document catalog so a query can be scoped to one document."""

    def __init__(self, store: PineconeStore) -> None:
        self.store = store

    def fetch_documents(self) -> list[str]:
        return self.store.list_documents()
