"""A small, thread-safe, in-memory TTL cache (stdlib only).

Used to cache the expensive on-chain collection step keyed by ``(chain, address)``
so repeated analyses skip the network. Features:

* per-entry TTL with lazy expiry on read,
* a lock so it's safe under the batch endpoint's thread pool,
* a max-entries cap with LRU-ish eviction (oldest / least-recently-used first),
* a TTL that can be a value **or a callable** (so it can track an env var live) —
  a TTL of ``<= 0`` disables the cache entirely (``get`` returns ``None``, ``set``
  is a no-op).

No external dependencies (no Redis); this is a per-process cache.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional, Union

TTLSpec = Union[float, Callable[[], float]]


class TTLCache:
    def __init__(
        self,
        ttl: TTLSpec,
        max_entries: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._max = max(1, int(max_entries))
        self._clock = clock
        self._lock = threading.Lock()
        self._store: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()

    def _ttl_seconds(self) -> float:
        try:
            return float(self._ttl() if callable(self._ttl) else self._ttl)
        except (TypeError, ValueError):
            return 0.0

    @property
    def enabled(self) -> bool:
        return self._ttl_seconds() > 0

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the cached value for ``key``, or ``None`` (miss/expired/disabled)."""
        if self._ttl_seconds() <= 0:
            return None
        now = self._clock()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                del self._store[key]  # lazy expiry
                return None
            self._store.move_to_end(key)  # mark as recently used
            return value

    def set(self, key: Hashable, value: Any) -> None:
        """Store ``value`` under ``key`` (no-op when the cache is disabled)."""
        ttl = self._ttl_seconds()
        if ttl <= 0:
            return
        now = self._clock()
        with self._lock:
            self._store[key] = (now + ttl, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # evict the oldest / LRU entry

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
