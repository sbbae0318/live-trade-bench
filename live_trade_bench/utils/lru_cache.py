from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Generic, MutableMapping, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    Simple thread-safe LRU cache using OrderedDict.
    - get: O(1)
    - put: O(1)
    """

    def __init__(self, capacity: int = 3000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity: int = capacity
        self._store: MutableMapping[K, V] = OrderedDict()
        self._lock: Lock = Lock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._store:
                return None
            # move to end (most-recently-used)
            self._store.move_to_end(key, last=True)
            return self._store[key]

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._store:
                # update and move to end
                self._store[key] = value
                self._store.move_to_end(key, last=True)
                return
            # insert new
            self._store[key] = value
            # evict if over capacity
            if len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


