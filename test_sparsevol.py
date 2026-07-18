"""Tests for the supervoxel -> RLE index.

Entirely offline: a synthetic index is built into a temporary store and served
through the real endpoints, so the read path, the byte-range planning and the
codec are all exercised without a graphene volume or network access.
"""

import io
import os
import shutil
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config, sparsevol
from app.rle import (
    COORD_DTYPE,
    decode_rle,
    encode_rle,
    merge_runs,
    pack_runs,
    rle_voxel_count,
    union_rle,
    unpack_runs,
)
from app.sparsevol_build import write_chunk

client = TestClient(app)

BASE = "transform-service/sparsevol/dataset/test_index/s/0"

# The test dataset declares spatial_bits=8, layer_bits=8, so a label is
# layer(8) | x(8) | y(8) | z(8) | segid(32).
LAYER_SHIFT = 56
SEGID_BITS = 32


def make_sv(chunk, segid):
    """A layer-1 supervoxel ID sitting in the given chunk."""
    x, y, z = chunk
    return (
        (1 << LAYER_SHIFT)
        | (x << (SEGID_BITS + 16))
        | (y << (SEGID_BITS + 8))
        | (z << SEGID_BITS)
        | segid
    )


def voxel_block(origin, shape):
    """Every voxel of a solid box, as (N, 3) coordinates."""
    ranges = [np.arange(o, o + s) for o, s in zip(origin, shape)]
    grid = np.meshgrid(*ranges, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=-1)


def sort_zyx(coords):
    """Deduplicate and sort into the (z, y, x) order decode_rle emits."""
    coords = np.unique(np.asarray(coords), axis=0)
    return coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]


@pytest.fixture(scope="module")
def index():
    """Build a small two-chunk index and return the truth it was built from."""
    store = config.SPARSEVOL_DATASOURCES["test_index"]["index"]
    shutil.rmtree(store[len("file://") :], ignore_errors=True)

    truth = {}
    for chunk, base in ((1, 2, 3), (0, 0, 0)), ((4, 5, 6), (2000, 100, 50)):
        fragments = {}
        for i in range(12):
            sv = make_sv(chunk, 1000 + i)
            coords = voxel_block((base[0] + i * 10, base[1] + i, base[2]), (7, 3, 2))
            fragments[sv] = encode_rle(coords)
            truth[sv] = coords
        write_chunk(store, chunk, 0, fragments)

    # Drop the process-level index cache so each run reads the fresh store.
    sparsevol._stores.pop("test_index", None)
    return truth


def post(supervoxels, fmt=None, headers=None):
    params = {"fmt": fmt} if fmt else None
    return client.post(
        BASE + "/supervoxels",
        json={"supervoxels": [str(sv) for sv in supervoxels]},
        params=params,
        headers=headers,
    )


def runs_from(response):
    return np.frombuffer(response.content, dtype="<i4").reshape(-1, 4).astype(np.int64)


# --- codec ----------------------------------------------------------------


def test_rle_round_trip():
    coords = voxel_block((5, 6, 7), (4, 3, 2))
    runs = encode_rle(coords)
    # 3 y-positions x 2 z-positions, each a solid run of 4 along x.
    assert runs.shape == (6, 4)
    assert np.all(runs[:, 3] == 4)
    np.testing.assert_array_equal(decode_rle(runs), sort_zyx(coords))
    assert rle_voxel_count(runs) == len(coords)


def test_rle_handles_duplicates_and_empty():
    coords = np.array([[1, 1, 1], [1, 1, 1], [2, 1, 1]])
    np.testing.assert_array_equal(encode_rle(coords), [[1, 1, 1, 2]])
    assert encode_rle(np.zeros((0, 3))).shape == (0, 4)
    assert decode_rle(np.zeros((0, 4))).shape == (0, 3)


def test_merge_runs_joins_abutting_fragments():
    # Two supervoxels meeting along X describe one unbroken stretch.
    merged = merge_runs(np.array([[0, 0, 0, 5], [5, 0, 0, 3]]))
    np.testing.assert_array_equal(merged, [[0, 0, 0, 8]])

    # A gap of one voxel must not be bridged.
    apart = merge_runs(np.array([[0, 0, 0, 5], [6, 0, 0, 3]]))
    assert len(apart) == 2

    # Different scanlines never merge, however adjacent their x ranges.
    rows = merge_runs(np.array([[0, 0, 0, 5], [5, 1, 0, 3]]))
    assert len(rows) == 2

    # A run wholly inside another must not shorten the result.
    nested = merge_runs(np.array([[0, 0, 0, 10], [2, 0, 0, 3]]))
    np.testing.assert_array_equal(nested, [[0, 0, 0, 10]])


def test_merge_runs_absorbs_overlaps_without_duplicating():
    # A nested run must not pull the coverage frontier backwards and let the
    # run after it open a second, overlapping block.
    merged = merge_runs(np.array([[0, 0, 0, 10], [1, 0, 0, 1], [5, 0, 0, 1]]))
    np.testing.assert_array_equal(merged, [[0, 0, 0, 10]])
    assert rle_voxel_count(merged) == 10
    assert len(np.unique(decode_rle(merged), axis=0)) == 10

    # The frontier must reset per scanline, not carry across them.
    across = merge_runs(np.array([[0, 0, 0, 10], [2, 1, 0, 1], [3, 1, 0, 1]]))
    np.testing.assert_array_equal(across, [[0, 0, 0, 10], [2, 1, 0, 2]])


def test_merge_runs_never_duplicates_a_voxel(random=np.random.default_rng(0)):
    """Random overlapping runs must still decode to distinct voxels."""
    for _ in range(200):
        n = int(random.integers(1, 12))
        runs = np.stack(
            [
                random.integers(0, 15, n),  # x
                random.integers(0, 2, n),  # y
                random.integers(0, 2, n),  # z
                random.integers(1, 8, n),  # length
            ],
            axis=-1,
        )
        merged = merge_runs(runs)
        coords = decode_rle(merged)
        truth = sort_zyx(decode_rle(runs))
        np.testing.assert_array_equal(coords, truth)
        assert rle_voxel_count(merged) == len(truth)


def test_encoding_does_not_widen_narrow_coordinates():
    """32-bit coordinates must survive the encoder without a detour via int64.

    The coordinate array is the large one in a live request -- tens of millions
    of rows -- so widening it inside encode_rle would undo the caller's choice
    to hold it narrow.
    """
    coords = voxel_block((5, 6, 7), (40, 8, 4)).astype(COORD_DTYPE)
    assert COORD_DTYPE().itemsize == 4

    narrow = encode_rle(coords)
    wide = encode_rle(coords.astype(np.int64))
    np.testing.assert_array_equal(narrow, wide)
    np.testing.assert_array_equal(decode_rle(narrow), sort_zyx(coords))


def test_encoding_handles_negative_coordinates():
    """Signed, not unsigned: a negative voxel offset must not wrap."""
    coords = np.array([[-5, -2, -1], [-4, -2, -1], [-3, -2, -1]], dtype=COORD_DTYPE)
    np.testing.assert_array_equal(encode_rle(coords), [[-5, -2, -1, 3]])
    np.testing.assert_array_equal(decode_rle(encode_rle(coords)), sort_zyx(coords))


def test_union_matches_encoding_the_whole_thing_at_once():
    left = voxel_block((0, 0, 0), (6, 4, 2))
    right = voxel_block((6, 0, 0), (6, 4, 2))
    together = encode_rle(np.concatenate([left, right]))
    unioned = union_rle([encode_rle(left), encode_rle(right)])
    np.testing.assert_array_equal(unioned, together)


@pytest.mark.parametrize("compress", [True, False])
def test_fragment_pack_round_trip(compress):
    runs = encode_rle(voxel_block((17, 300, 9), (5, 4, 3)))
    blob = pack_runs(runs, compress=compress)
    np.testing.assert_array_equal(unpack_runs(blob, compressed=compress), runs)


def test_pack_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        pack_runs(np.array([[2**40, 0, 0, 1]]))


def test_packing_is_deterministic_and_smaller_than_raw():
    runs = encode_rle(voxel_block((0, 0, 0), (40, 40, 20)))
    blob = pack_runs(runs)
    assert blob == pack_runs(runs)
    assert len(blob) < runs.astype("<i4").nbytes


# --- index format ---------------------------------------------------------


def test_index_round_trip():
    records = np.zeros(3, dtype=sparsevol.INDEX_DTYPE)
    records["sv"] = [10, 20, 30]
    records["offset"] = [0, 64, 200]
    records["nbytes"] = [64, 136, 12]
    records["n_runs"] = [4, 9, 1]

    decoded, mip, compressed = sparsevol.decode_index(
        sparsevol.encode_index(records, 2)
    )
    np.testing.assert_array_equal(decoded, records)
    assert mip == 2 and compressed is True


def test_index_rejects_unsorted_records():
    records = np.zeros(2, dtype=sparsevol.INDEX_DTYPE)
    records["sv"] = [30, 10]
    with pytest.raises(ValueError):
        sparsevol.encode_index(records, 0)


def test_decode_index_rejects_foreign_bytes():
    with pytest.raises(ValueError):
        sparsevol.decode_index(b"not an index at all")


def test_chunk_layout_decodes_position():
    layout = sparsevol.ChunkLayout(spatial_bits=8, layer_bits=8)
    ids = [make_sv((1, 2, 3), 99), make_sv((250, 0, 7), 1)]
    np.testing.assert_array_equal(layout.decode(ids), [[1, 2, 3], [250, 0, 7]])


def test_group_by_chunk_splits_and_sorts(index):
    a, b = make_sv((1, 2, 3), 5), make_sv((4, 5, 6), 2)
    grouped = sparsevol.group_by_chunk("test_index", [b, a, a])
    assert set(grouped) == {"1_2_3", "4_5_6"}
    np.testing.assert_array_equal(grouped["1_2_3"], [a])


# --- serve path -----------------------------------------------------------


def test_datasets_are_listed():
    response = client.get("transform-service/sparsevol/datasets")
    assert response.status_code == 200
    assert "test_index" in response.json()


def test_supervoxel_lookup_returns_the_indexed_voxels(index):
    sv = next(iter(index))
    response = post([sv])
    assert response.status_code == 200

    np.testing.assert_array_equal(decode_rle(runs_from(response)), sort_zyx(index[sv]))
    assert response.headers["X-Sparsevol-Missing"] == "0"
    assert response.headers["X-Sparsevol-Fragments"] == "1"


def test_union_spans_chunks(index):
    everything = list(index)
    response = post(everything)
    assert response.status_code == 200

    expected = sort_zyx(np.concatenate([index[sv] for sv in everything]))
    np.testing.assert_array_equal(decode_rle(runs_from(response)), expected)

    assert response.headers["X-Sparsevol-Chunks"] == "2"
    assert response.headers["X-Sparsevol-Fragments"] == str(len(everything))
    assert int(response.headers["X-Sparsevol-Voxels"]) == len(expected)


def test_neighbouring_fragments_are_read_together(index):
    """Fragments close together in the data file cost one read, not one each."""
    chunk_svs = [sv for sv in index if sparsevol.ChunkLayout(8).decode([sv])[0][0] == 1]
    assert len(chunk_svs) == 12

    stats = post(chunk_svs, fmt="json").json()["stats"]
    assert stats["n_fragments"] == 12
    # All 12 are small and adjacent in the data file, so they coalesce into one.
    assert stats["n_range_reads"] == 1


def test_index_reads_are_cached_across_requests(index):
    sv = next(iter(index))
    post([sv])  # warm the cache, whatever state earlier tests left it in
    stats = post([sv], fmt="json").json()["stats"]
    assert stats["n_index_reads"] == 0
    assert stats["n_index_cached"] == 1


def test_unknown_supervoxels_are_reported_not_fatal(index):
    known = next(iter(index))
    response = post([known, make_sv((1, 2, 3), 999999), make_sv((9, 9, 9), 1)])
    assert response.status_code == 200
    assert response.headers["X-Sparsevol-Missing"] == "2"
    assert response.headers["X-Sparsevol-Fragments"] == "1"


def test_empty_request_returns_no_runs(index):
    response = post([])
    assert response.status_code == 200
    assert response.headers["X-Sparsevol-Runs"] == "0"
    assert response.content == b""


def test_formats_agree(index):
    svs = list(index)[:3]
    reference = runs_from(post(svs))

    as_npy = np.load(io.BytesIO(post(svs, fmt="npy").content))
    np.testing.assert_array_equal(as_npy, reference)

    as_json = post(svs, fmt="json").json()
    np.testing.assert_array_equal(np.array(as_json["runs"]), reference)
    assert as_json["stats"]["n_runs"] == len(reference)

    as_coords = np.frombuffer(
        post(svs, fmt="coords").content, dtype="<i4"
    ).reshape(-1, 3)
    np.testing.assert_array_equal(as_coords, decode_rle(reference))


def test_rle_is_smaller_than_coordinates(index):
    """The point of the format: solid blocks collapse along X."""
    svs = list(index)
    rle = post(svs).content
    coords = post(svs, fmt="coords").content
    assert len(rle) < len(coords)


def test_gzip_is_offered_when_the_client_accepts_it(index):
    svs = list(index)
    plain = post(svs, headers={"accept-encoding": "identity"})
    assert "content-encoding" not in plain.headers

    compressed = client.post(
        BASE + "/supervoxels",
        json={"supervoxels": [str(sv) for sv in svs]},
        headers={"accept-encoding": "gzip"},
    )
    # TestClient decodes transparently, so compare against the raw body.
    assert compressed.headers.get("content-encoding") == "gzip"
    np.testing.assert_array_equal(runs_from(compressed), runs_from(plain))


def test_unindexed_scale_is_rejected(index):
    response = client.post(
        "transform-service/sparsevol/dataset/test_index/s/5/supervoxels",
        json={"supervoxels": ["1"]},
    )
    assert response.status_code == 400
    assert "not indexed" in response.json()["detail"]


def test_root_lookup_needs_a_graphene_source(index):
    """The test dataset has no graph, so root resolution must fail clearly."""
    response = client.get(BASE + "/root/720575940626838909")
    assert response.status_code == 400
    assert "graphene" in response.json()["detail"]


def test_too_many_supervoxels_is_refused(index, monkeypatch):
    monkeypatch.setitem(config.SPARSEVOL_DATASOURCES["test_index"], "max_supervoxels", 2)
    response = post(list(index))
    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_out_of_range_supervoxel_ids_are_rejected(index):
    for bad in ["-1", str(2**64)]:
        response = client.post(BASE + "/supervoxels", json={"supervoxels": [bad]})
        assert response.status_code == 400, bad
        assert "64-bit" in response.json()["detail"]

    response = client.post(BASE + "/supervoxels", json={"supervoxels": ["not a number"]})
    assert response.status_code == 400


# --- failures must not look like empty results ----------------------------
#
# A store that is failing and a store that is merely incomplete produce very
# different correct answers. Confusing them returns a truncated neuron with
# HTTP 200, which a client is invited to cache forever.


class BrokenFiles:
    """A CloudFiles stand-in that fails whichever reads the test names."""

    def __init__(self, real, fail_on):
        self.real = real
        self.fail_on = fail_on

    def get(self, paths, **kwargs):
        results = self.real.get(paths, **kwargs)
        for result in results:
            if result["path"].endswith(self.fail_on):
                result["error"] = OSError("503 from the store")
                result["content"] = None
        return results


@pytest.fixture
def failing_store(index, monkeypatch):
    store = sparsevol.get_store("test_index")
    store._cache.clear()  # force real index reads so they can be made to fail

    def break_on(suffix):
        monkeypatch.setattr(store, "_cf", BrokenFiles(store._files(), suffix))

    yield break_on
    store._cache.clear()


def test_failed_index_read_is_an_error_not_an_empty_answer(index, failing_store):
    failing_store(".idx")
    response = post(list(index))
    assert response.status_code == 502
    assert "index" in response.json()["detail"]


def test_failed_fragment_read_is_an_error_not_a_partial_neuron(index, failing_store):
    failing_store(".dat")
    response = post(list(index))
    assert response.status_code == 502


def test_index_disagreeing_with_data_is_reported(index):
    """A fragment the index promises but the data file cannot supply."""
    store_path = config.SPARSEVOL_DATASOURCES["test_index"]["index"][len("file://") :]
    sv = make_sv((7, 7, 7), 1)

    records = np.zeros(1, dtype=sparsevol.INDEX_DTYPE)
    records["sv"], records["offset"], records["nbytes"], records["n_runs"] = sv, 0, 64, 4
    with open(os.path.join(store_path, "0", "7_7_7.idx"), "wb") as f:
        f.write(sparsevol.encode_index(records, 0))
    with open(os.path.join(store_path, "0", "7_7_7.dat"), "wb") as f:
        f.write(b"\0" * 8)  # far short of the 64 bytes promised

    sparsevol.get_store("test_index")._cache.clear()
    response = post([sv])
    assert response.status_code == 502


def test_layout_lookup_does_not_deadlock(monkeypatch):
    """get_layout must not take a lock it then re-enters via get_volume."""

    class FakeMeta:
        def spatial_bit_count(self, level):
            return 8

        n_bits_for_layer_id = 8

    monkeypatch.setitem(
        config.SPARSEVOL_DATASOURCES,
        "deadlock_check",
        {"index": "file:///tmp/none", "scales": [0], "graphene": "graphene://fake"},
    )
    monkeypatch.setitem(
        sparsevol._volumes, "deadlock_check", type("V", (), {"meta": FakeMeta()})()
    )

    done = []
    thread = threading.Thread(
        target=lambda: done.append(sparsevol.get_layout("deadlock_check"))
    )
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "get_layout deadlocked"
    assert done[0].spatial_bits == 8

    # The lock must be free afterwards for every other dataset.
    assert sparsevol.get_store("test_index") is not None
    sparsevol._layouts.pop("deadlock_check", None)


# --- build path -----------------------------------------------------------


class FakeVolume:
    """Just enough of a graphene CloudVolume to build one chunk from.

    Chunks are 64^3, so chunk (1, 0, 0) owns x in [64, 128).
    """

    def __init__(self):
        self.volume = np.zeros((256, 256, 256), dtype=np.uint64)
        self.a = make_sv((1, 0, 0), 11)
        self.b = make_sv((1, 0, 0), 22)
        self.volume[70:75, 3:5, 2:4] = self.a
        self.volume[80:83, 3:4, 2:3] = self.b
        # A supervoxel owned by the neighbouring chunk, sitting just outside.
        self.foreign = make_sv((0, 0, 0), 33)
        self.volume[60:64, 3:4, 2:3] = self.foreign
        self.meta = self

    # -- metadata surface the builder uses
    watershed_mip = 0
    chunks_start_at_voxel_offset = False
    graph_chunk_size = np.array([64, 64, 64])

    def bounds(self, mip):
        from cloudvolume.lib import Bbox

        return Bbox([0, 0, 0], [256, 256, 256])

    def decode_chunk_position(self, label):
        return sparsevol.ChunkLayout(8).decode([label])[0]

    def download(self, box, mip=0, agglomerate=False):
        cut = self.volume[
            box.minpt[0] : box.maxpt[0],
            box.minpt[1] : box.maxpt[1],
            box.minpt[2] : box.maxpt[2],
        ]
        return cut[..., np.newaxis]


def test_build_from_dense_then_serve(index):
    """The whole pipeline: dense read -> fragments -> ranged read -> response."""
    from app.sparsevol_build import fragments_for_chunk

    cv = FakeVolume()
    fragments = fragments_for_chunk(cv, (1, 0, 0), mip=0)

    # A supervoxel belongs to exactly one chunk, so the neighbour's voxels --
    # which a rounded chunk box sweeps up at coarse mips -- must not be written
    # here as a second, partial fragment.
    assert set(fragments) == {cv.a, cv.b}

    store = config.SPARSEVOL_DATASOURCES["test_index"]["index"]
    write_chunk(store, (1, 0, 0), 0, fragments)
    sparsevol.get_store("test_index")._cache.clear()

    response = post([cv.a, cv.b])
    assert response.status_code == 200
    np.testing.assert_array_equal(
        decode_rle(runs_from(response)),
        sort_zyx(np.argwhere((cv.volume == cv.a) | (cv.volume == cv.b))),
    )
