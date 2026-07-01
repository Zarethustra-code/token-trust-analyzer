"""Tests for utils/cache.py — deterministic (injected clock, no sleeps)."""

from __future__ import annotations

import threading

from utils.cache import TTLCache


class _Clock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_get_set_hit_and_miss():
    c = TTLCache(ttl=100, clock=_Clock())
    assert c.get("k") is None                 # miss
    c.set("k", 42)
    assert c.get("k") == 42                    # hit


def test_ttl_expiry():
    clock = _Clock()
    c = TTLCache(ttl=10, clock=clock)
    c.set("k", "v")
    clock.advance(9)
    assert c.get("k") == "v"                    # still fresh
    clock.advance(2)                            # now 11 > ttl 10
    assert c.get("k") is None                   # expired
    assert len(c) == 0                          # lazily evicted on read


def test_disabled_when_ttl_zero():
    c = TTLCache(ttl=0, clock=_Clock())
    assert c.enabled is False
    c.set("k", "v")
    assert c.get("k") is None
    assert len(c) == 0


def test_ttl_callable_tracks_live_value():
    ttl = {"v": 0}
    c = TTLCache(ttl=lambda: ttl["v"], clock=_Clock())
    c.set("k", 1)
    assert c.get("k") is None                   # disabled (ttl 0)
    ttl["v"] = 100
    c.set("k", 1)
    assert c.get("k") == 1                       # now enabled


def test_max_entries_eviction_is_lru():
    c = TTLCache(ttl=100, max_entries=3, clock=_Clock())
    for k in ("a", "b", "c"):
        c.set(k, k)
    assert c.get("a") == "a"                     # touch 'a' -> most recently used
    c.set("d", "d")                              # over cap -> evict LRU ('b')
    assert len(c) == 3
    assert c.get("b") is None                    # 'b' evicted
    assert c.get("a") == "a" and c.get("c") == "c" and c.get("d") == "d"


def test_thread_safety_smoke():
    c = TTLCache(ttl=100, max_entries=1000, clock=_Clock())

    def worker(base):
        for i in range(200):
            c.set((base, i), i)
            c.get((base, i))

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(c) <= 1000                        # never exceeds the cap; no crash
