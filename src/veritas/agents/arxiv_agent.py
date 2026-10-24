from __future__ import annotations

import logging

from veritas.core.clients.arxiv import ArxivClient
from veritas.core.models import SourceItem

logger = logging.getLogger(__name__)


class ArxivAgent:
    def __init__(self, client: ArxivClient | None = None) -> None:
        self.client = client or ArxivClient()

    def search(self, query: str, max_results: int = 5) -> list[SourceItem]:
        try:
            results = self.client.search(query, max_results)
        except Exception:  # an optional source must never fail the research request
            logger.warning("arXiv search failed", exc_info=True)
            return []
        logger.info("arXiv search returned %d results", len(results))
        return results
