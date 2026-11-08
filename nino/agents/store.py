"""Store for tracking and retrieving recent sub-agent delegation runs."""

import logging
import threading
from typing import Any
from nino.agents.models import AgentRunRecord

logger = logging.getLogger("nino.agents.store")


class DelegationRunStore:
    """Thread-safe store of recent sub-agent execution runs for observability."""

    def __init__(self, max_records: int = 200):
        self.max_records = max_records
        self._runs: dict[str, AgentRunRecord] = {}
        self._lock = threading.Lock()

    def record_run(self, record: AgentRunRecord) -> None:
        """Store a completed or pending delegation run record."""
        with self._lock:
            self._runs[record.run_id] = record
            if len(self._runs) > self.max_records:
                # Evict oldest record
                oldest_id = min(self._runs.keys(), key=lambda k: self._runs[k].created_at)
                del self._runs[oldest_id]
        logger.debug("Recorded delegation run %s (%s)", record.run_id, record.status)

    def get_run(self, run_id: str) -> AgentRunRecord | None:
        """Retrieve a run record by run_id."""
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self, limit: int = 50) -> list[AgentRunRecord]:
        """List recent delegation runs, newest first."""
        with self._lock:
            sorted_runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
            return sorted_runs[:limit]

    def clear(self) -> None:
        """Clear all stored runs (primarily for testing)."""
        with self._lock:
            self._runs.clear()


# Global default store instance
_global_run_store: DelegationRunStore | None = None


def get_run_store() -> DelegationRunStore:
    """Get or instantiate global delegation run store singleton."""
    global _global_run_store
    if _global_run_store is None:
        _global_run_store = DelegationRunStore()
    return _global_run_store
