from __future__ import annotations

import logging

from veritas.core.clients.duckduckgo import DuckDuckGoClient
from veritas.core.models import SourceItem

logger = logging.getLogger(__name__)


class WebSearchAgent:
    def __init__(self, client: DuckDuckGoClient | None = None) -> None:
        self.client = client or DuckDuckGoClient()

    def search(self, query: str, max_results: int = 5) -> list[SourceItem]:
        try:
            results = self.client.search(query, max_results)
        except Exception:  # an optional source must never fail the research request
            logger.warning("Web search failed", exc_info=True)
            return []
        logger.info("Web search returned %d results", len(results))
        return results
