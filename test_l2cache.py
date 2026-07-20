"""Tests for the bounded layer-2 RLE cache.

These exercise the store on its own -- no volume, no chunkedgraph. The tests
that prove the cache actually changes what a request does live in
`test_sparsevol_live.py`, next to the fixtures that can read voxels.
"""

import numpy as np
import numpy.testing  # noqa: F401  -- see the note in test_sparsevol_live.py
import pytest
from fastapi import HTTPException

from app import config, l2cache

RUNS = np.array([[1, 2, 3, 4], [10, 2, 3, 1], [4, 5, 3, 7]], dtype=np.int64)
EMPTY = np.zeros((0, 4), dtype=np.int64)


@pytest.fixture
def cache(tmp_path):
    return l2cache.L2Cache(
        str(tmp_path / "l2.sqlite"), max_bytes=10**9, max_keys=1000
    )


# --- storage --------------------------------------------------------------


def test_runs_survive_the_round_trip(cache):
    cache.put_many("ds", 0, {123: RUNS})
    np.testing.assert_array_equal(cache.get_many("ds", 0, [123])[123], RUNS)


def test_labels_beyond_int64_are_preserved(cache):
    """SQLite has no unsigned type, so the bit pattern has to be kept.

    A layer id high enough to set bit 63 is what would break a naive cast, and
    a graphene label is exactly that: the layer lives in the top eight bits.
    """
    l2_id = (200 << 56) | 7
    assert l2_id >= 2**63

    cache.put_many("ds", 0, {l2_id: RUNS})
    assert l2_id in cache.get_many("ds", 0, [l2_id])
    assert cache.have("ds", 0, [l2_id]) == {l2_id}


def test_empty_fragments_are_remembered(cache):
    """An L2 node with no voxels here is an answer, not a miss.

    If it were not stored, every future request would pay the dense read again
    to rediscover that there is nothing there.
    """
    cache.put_many("ds", 0, {5: EMPTY})
    found = cache.get_many("ds", 0, [5])
    assert 5 in found and len(found[5]) == 0
    assert cache.have("ds", 0, [5]) == {5}


def test_dataset_and_scale_are_part_of_the_key(cache):
    """The same node at another scale is different voxels entirely."""
    cache.put_many("a", 0, {1: RUNS})
    assert cache.get_many("a", 1, [1]) == {}
    assert cache.get_many("b", 0, [1]) == {}


def test_absent_keys_are_simply_absent(cache):
    cache.put_many("ds", 0, {1: RUNS})
    assert set(cache.get_many("ds", 0, [1, 2, 3])) == {1}


def test_lookups_span_more_keys_than_sqlite_takes_parameters(cache):
    """A neuron can have more L2 nodes than SQLite allows bound parameters."""
    many = {i: RUNS for i in range(1, 1200)}
    assert cache.put_many("ds", 0, many) == len(many)
    assert len(cache.get_many("ds", 0, list(many))) == len(many)
    assert len(cache.have("ds", 0, list(many))) == len(many)


# --- accounting -----------------------------------------------------------


def test_writing_a_key_twice_does_not_double_count(cache):
    """Two workers can compute the same node at once; the second is a no-op."""
    cache.put_many("ds", 0, {1: RUNS})
    before = cache.counters()

    assert cache.put_many("ds", 0, {1: RUNS}) == 0
    assert cache.counters() == before


def test_counters_track_what_was_stored(cache):
    cache.put_many("ds", 0, {1: RUNS, 2: RUNS})
    n_keys, n_bytes = cache.counters()
    assert n_keys == 2
    assert n_bytes > 0
    assert cache.stats()["fraction_keys"] == pytest.approx(2 / 1000)


# --- the ceiling ----------------------------------------------------------


def test_a_full_cache_refuses_every_request(tmp_path):
    """Not eviction: the request fails, so somebody has to look at it."""
    cache = l2cache.L2Cache(str(tmp_path / "l2.sqlite"), max_keys=2, max_bytes=10**9)
    cache.check_health()  # empty, so fine

    cache.put_many("ds", 0, {1: RUNS, 2: RUNS})
    assert cache.is_full()

    with pytest.raises(HTTPException) as raised:
        cache.check_health()
    assert raised.value.status_code == 503


def test_the_error_says_how_to_recover(tmp_path):
    """A wedged cache is an operator's problem, so the 503 has to be actionable."""
    cache = l2cache.L2Cache(str(tmp_path / "l2.sqlite"), max_keys=1, max_bytes=10**9)
    cache.put_many("ds", 0, {1: RUNS})

    with pytest.raises(HTTPException) as raised:
        cache.check_health()
    detail = raised.value.detail
    assert "L2CacheMaxBytes" in detail
    assert "--clear" in detail
    assert "L2CacheEnabled" in detail


def test_the_byte_ceiling_also_trips(tmp_path):
    cache = l2cache.L2Cache(str(tmp_path / "l2.sqlite"), max_keys=10**9, max_bytes=1)
    cache.put_many("ds", 0, {1: RUNS})
    assert cache.is_full()
    with pytest.raises(HTTPException):
        cache.check_health()


def test_a_full_cache_can_be_emptied(tmp_path):
    cache = l2cache.L2Cache(str(tmp_path / "l2.sqlite"), max_keys=1, max_bytes=10**9)
    cache.put_many("ds", 0, {1: RUNS})
    assert cache.is_full()

    assert cache.clear() == 1
    assert cache.counters() == (0, 0)
    assert not cache.is_full()
    cache.check_health()


def test_clearing_one_dataset_leaves_the_others(cache):
    cache.put_many("a", 0, {1: RUNS})
    cache.put_many("b", 0, {1: RUNS})

    cache.clear(dataset="a", scale=0)
    assert cache.get_many("a", 0, [1]) == {}
    assert 1 in cache.get_many("b", 0, [1])
    # Recounted from what survived, not decremented from what went.
    assert cache.counters()[0] == 1


# --- wiring ---------------------------------------------------------------


def test_caching_can_be_switched_off(monkeypatch):
    """None is a supported state: callers fall back to computing live."""
    monkeypatch.setattr(config, "L2CacheEnabled", False)
    monkeypatch.setattr(l2cache, "_cache", None)
    assert l2cache.get_cache() is None


def test_the_handle_is_shared_within_a_process(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "L2CacheEnabled", True)
    monkeypatch.setattr(config, "L2_CACHE_PATH", str(tmp_path / "l2.sqlite"))
    monkeypatch.setattr(l2cache, "_cache", None)
    try:
        assert l2cache.get_cache() is l2cache.get_cache()
    finally:
        l2cache.reset_cache()


@pytest.mark.parametrize("value", [0, 1, 2**63 - 1, 2**63, 2**64 - 1])
def test_label_encoding_is_reversible(value):
    assert l2cache.to_unsigned(l2cache.to_signed(value)) == value


@pytest.mark.parametrize("value", [-1, 2**64])
def test_labels_outside_uint64_are_rejected(value):
    with pytest.raises(ValueError):
        l2cache.to_signed(value)
