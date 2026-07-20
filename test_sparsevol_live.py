"""Tests for sparse voxels computed live from a locally-stored segmentation.

A small neuroglancer_precomputed volume is written to disk and read back
through the real endpoint, so the chunk geometry, the masking and the encoding
are all exercised for real. Only the chunkedgraph is substituted -- it is the
one piece that is genuinely remote.
"""

import shutil
import threading
import time
from contextlib import contextmanager

import numpy as np

# Imported eagerly, not via the np.testing attribute hook: numpy 1.26 probes
# for SVE support by shelling out the first time numpy.testing is loaded, and
# that abort()s once the read pool has threads running.
import numpy.testing  # noqa: F401
import pytest
import tensorstore as ts
from fastapi.testclient import TestClient

from app.main import app
from app import config, datasource, l2cache, l2cache_warm, sparsevol_live
from app.chunks import ChunkLayout, chunk_boxes
from app.rle import COORD_DTYPE, decode_rle

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Give every test its own empty cache, never the deployment's.

    Autouse because the cached path is the default: a test that forgot this
    would read a cache another test wrote, and pass or fail on the order they
    happened to run in.
    """
    monkeypatch.setattr(config, "L2_CACHE_PATH", str(tmp_path / "l2.sqlite"))
    monkeypatch.setattr(l2cache, "_cache", None)
    yield
    l2cache.reset_cache()

BASE = "transform-service/sparsevol/dataset/test_segmentation/s/0"
VOLUME = "sparsevol_test_volume"

CHUNK = np.array([64, 64, 32])
SHAPE = np.array([128, 128, 64])

LAYER_SHIFT, SEGID_BITS = 56, 32


def label(layer, chunk, segid):
    x, y, z = chunk
    return (
        (layer << LAYER_SHIFT)
        | (x << (SEGID_BITS + 16))
        | (y << (SEGID_BITS + 8))
        | (z << SEGID_BITS)
        | segid
    )


def supervoxel(chunk, segid):
    return label(1, chunk, segid)


def sort_zyx(coords):
    coords = np.unique(np.asarray(coords), axis=0)
    return coords[np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))]


# Three supervoxels: two in different chunks making up one "neuron", and a
# third sharing a chunk with the first so masking has something to exclude.
SV_A = supervoxel((0, 0, 0), 11)
SV_B = supervoxel((1, 0, 0), 22)
SV_OTHER = supervoxel((0, 0, 0), 33)

# The layer-2 node owning each of them. L2_A and L2_OTHER share a chunk, which
# is what a neuron passing through one twice looks like.
L2_A = label(2, (0, 0, 0), 1)
L2_B = label(2, (1, 0, 0), 1)
L2_OTHER = label(2, (0, 0, 0), 2)
L2_MEMBERS = {L2_A: [SV_A], L2_B: [SV_B], L2_OTHER: [SV_OTHER]}

# Root 999 spans two chunks with one node each; root 888 has two nodes in one.
ROOT_NODES = {999: [L2_A, L2_B], 888: [L2_A, L2_OTHER]}

REGIONS = {
    SV_A: (slice(10, 20), slice(5, 8), slice(2, 4)),
    SV_B: (slice(70, 75), slice(5, 7), slice(2, 3)),
    SV_OTHER: (slice(30, 35), slice(5, 8), slice(2, 4)),
}


class FakeGraph:
    """The chunkedgraph, reduced to what the live path actually asks of it.

    Deliberately holds no voxels: if the service ever tried to read image data
    through the graph rather than from the local volume, this would fail.
    """

    def __init__(self):
        self.meta = self
        self.leaf_calls = []

    watershed_mip = 0
    chunks_start_at_voxel_offset = False
    graph_chunk_size = CHUNK

    def voxel_offset(self, mip):
        return np.zeros(3, dtype=np.int64)

    def bounds(self, mip):
        from cloudvolume.lib import Bbox

        return Bbox([0, 0, 0], SHAPE)

    def bbox_to_mip(self, box, mip, to_mip):
        return box

    def get_leaves(self, root_id, bbox, mip, stop_layer=None):
        self.leaf_calls.append((int(root_id), stop_layer))
        if stop_layer == 2:
            return np.array(
                ROOT_NODES.get(int(root_id), [L2_A, L2_B]), dtype=np.uint64
            )
        # Asked about a layer-2 node, answer for that node alone. The real
        # graph does the same, and it is what makes per-node caching possible:
        # a whole-root manifest could never say who owns which voxel.
        if int(root_id) in L2_MEMBERS:
            return np.array(L2_MEMBERS[int(root_id)], dtype=np.uint64)
        return np.array([SV_A, SV_B], dtype=np.uint64)


@pytest.fixture(scope="module")
def volume():
    """Write a small uint64 segmentation to disk and point the config at it."""
    shutil.rmtree(VOLUME, ignore_errors=True)

    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": VOLUME},
            "multiscale_metadata": {
                "type": "segmentation",
                "data_type": "uint64",
                "num_channels": 1,
            },
            "scale_metadata": {
                "size": SHAPE.tolist(),
                "encoding": "raw",
                "chunk_size": CHUNK.tolist(),
                "resolution": [16, 16, 45],
                "voxel_offset": [0, 0, 0],
            },
            "create": True,
            "delete_existing": True,
        }
    ).result()

    data = np.zeros(SHAPE, dtype=np.uint64)
    for sv, region in REGIONS.items():
        data[region] = sv
    store[:, :, :, 0].write(data).result()

    # Drop handles so the fresh volume and stand-in graph are picked up.
    datasource.open_n5_mip.pop(("test_segmentation", 0), None)
    graph = FakeGraph()
    sparsevol_live._graphs["test_segmentation"] = graph

    yield data, graph

    sparsevol_live._graphs.pop("test_segmentation", None)
    datasource.open_n5_mip.pop(("test_segmentation", 0), None)


def truth_for(data, labels):
    mask = np.zeros(data.shape, dtype=bool)
    for sv in labels:
        mask |= data == sv
    return sort_zyx(np.argwhere(mask))


def post(supervoxels, fmt=None):
    return client.post(
        BASE + "/supervoxels",
        json={"supervoxels": [str(sv) for sv in supervoxels]},
        params={"fmt": fmt} if fmt else None,
    )


def runs_from(response):
    return np.frombuffer(response.content, dtype="<i4").reshape(-1, 4).astype(np.int64)


# --- the core claim -------------------------------------------------------


def test_supervoxel_lookup_reads_the_local_volume(volume):
    data, graph = volume
    response = post([SV_A])
    assert response.status_code == 200

    np.testing.assert_array_equal(
        decode_rle(runs_from(response)), truth_for(data, [SV_A])
    )
    # The chunk came out of the ID, so the graph was never consulted.
    assert graph.leaf_calls == []
    assert response.headers["X-Sparsevol-Chunks"] == "1"


def test_masking_excludes_other_labels_in_the_same_chunk(volume):
    data, _ = volume
    coords = decode_rle(runs_from(post([SV_A])))
    other = set(map(tuple, np.argwhere(data == SV_OTHER)))
    assert other and not (set(map(tuple, coords)) & other)


def test_root_lookup_uses_the_graph_for_the_manifest(volume):
    data, graph = volume
    graph.leaf_calls.clear()

    response = client.get(BASE + "/root/999")
    assert response.status_code == 200
    np.testing.assert_array_equal(
        decode_rle(runs_from(response)), truth_for(data, [SV_A, SV_B])
    )

    # One call for the layer-2 nodes, then one manifest per node.
    assert (999, 2) in graph.leaf_calls
    assert sorted(call for call in graph.leaf_calls if call[1] is None) == [
        (L2_A, None),
        (L2_B, None),
    ]
    assert response.headers["X-Sparsevol-Chunks"] == "2"
    assert response.headers["X-Sparsevol-Supervoxels"] == "2"


def test_only_the_occupied_chunks_are_read(volume):
    """The graph bounds the read: 2 chunks of the volume's 8, not all of it."""
    response = client.get(BASE + "/root/999")
    read = int(response.headers["X-Sparsevol-Voxels-Read"])
    assert read == 2 * int(np.prod(CHUNK))
    assert read < int(np.prod(SHAPE))


def test_response_is_far_smaller_than_the_chunks_it_read(volume):
    """The whole point: the client gets runs, not the dense chunks."""
    response = client.get(BASE + "/root/999")
    dense_bytes = int(response.headers["X-Sparsevol-Voxels-Read"]) * 8  # uint64
    assert len(response.content) < dense_bytes / 100
    assert float(response.headers["X-Sparsevol-Reduction"]) > 100


def test_multiple_supervoxels_span_chunks(volume):
    data, _ = volume
    response = post([SV_A, SV_B])
    np.testing.assert_array_equal(
        decode_rle(runs_from(response)), truth_for(data, [SV_A, SV_B])
    )
    assert response.headers["X-Sparsevol-Chunks"] == "2"


def test_unknown_supervoxel_yields_nothing_but_does_not_fail(volume):
    response = post([supervoxel((0, 0, 0), 987654)])
    assert response.status_code == 200
    assert response.headers["X-Sparsevol-Runs"] == "0"
    assert response.content == b""


def test_supervoxel_outside_the_volume_is_clipped_away(volume):
    """A chunk position past the volume bounds must not raise."""
    response = post([supervoxel((250, 250, 250), 1)])
    assert response.status_code == 200
    assert response.headers["X-Sparsevol-Runs"] == "0"


def test_formats_agree(volume):
    reference = runs_from(post([SV_A, SV_B]))
    as_json = post([SV_A, SV_B], fmt="json").json()
    np.testing.assert_array_equal(np.array(as_json["runs"]), reference)
    coords = np.frombuffer(post([SV_A, SV_B], fmt="coords").content, dtype="<i4")
    np.testing.assert_array_equal(coords.reshape(-1, 3), decode_rle(reference))


# --- guards ---------------------------------------------------------------


def test_chunk_budget_is_enforced(volume, monkeypatch):
    monkeypatch.setitem(config.DATASOURCES["test_segmentation"], "max_chunks", 1)
    response = post([SV_A, SV_B])
    assert response.status_code == 400
    assert "over the 1 limit" in response.json()["detail"]


def test_voxel_budget_is_enforced(volume, monkeypatch):
    monkeypatch.setitem(config.DATASOURCES["test_segmentation"], "max_voxels", 10)
    response = post([SV_A])
    assert response.status_code == 400
    assert "voxels" in response.json()["detail"]


def test_coordinates_are_accumulated_narrow(volume):
    """The array that scales with neuron size stays 32-bit end to end."""
    data, graph = volume
    stats = sparsevol_live.LiveStats()
    boxes = chunk_boxes(graph.meta, [[0, 0, 0], [1, 0, 0]], mip=0)
    coords = sparsevol_live.read_and_mask(
        "test_segmentation", 0, boxes, [SV_A, SV_B], stats
    )

    assert coords.dtype == COORD_DTYPE
    assert coords.dtype.itemsize == 4
    np.testing.assert_array_equal(sort_zyx(coords), truth_for(data, [SV_A, SV_B]))

    # And an empty read must agree on dtype rather than falling back to int64.
    empty = sparsevol_live.read_and_mask("test_segmentation", 0, [], [SV_A], stats)
    assert empty.dtype == COORD_DTYPE


# --- concurrency cap ------------------------------------------------------


@pytest.fixture
def slots(monkeypatch):
    """Replace the process-wide slot pool with one sized for the test."""

    def resize(count, queue_seconds=20):
        monkeypatch.setattr(
            sparsevol_live, "_read_slots", threading.BoundedSemaphore(count)
        )
        monkeypatch.setattr(config, "SparseVolQueueSeconds", queue_seconds)

    return resize


def test_reads_beyond_the_cap_are_shed_not_queued_forever(volume, slots):
    """With every slot held, a request gives up and says so."""
    slots(1, queue_seconds=1)

    held = threading.Event()
    released = threading.Event()

    def hog():
        with sparsevol_live.read_slot(sparsevol_live.LiveStats()):
            held.set()
            released.wait(timeout=10)

    holder = threading.Thread(target=hog)
    holder.start()
    assert held.wait(timeout=5)

    try:
        response = post([SV_A])
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"
        assert "in flight" in response.json()["detail"]
    finally:
        released.set()
        holder.join(timeout=5)


def test_a_slot_is_returned_after_the_read(volume, slots):
    """A shed request must not leak the slot it never got."""
    slots(1, queue_seconds=1)
    for _ in range(3):
        assert post([SV_A]).status_code == 200

    # And after a failure inside the read, too.
    with pytest.raises(Exception):
        with sparsevol_live.read_slot(sparsevol_live.LiveStats()):
            raise RuntimeError("boom")
    assert post([SV_A]).status_code == 200


def test_concurrent_reads_are_limited_to_the_cap(volume, slots):
    """Never more than `cap` reads inside the masking step at once."""
    slots(2)

    live = 0
    peak = 0
    guard = threading.Lock()
    real_slot = sparsevol_live.read_slot

    @contextmanager
    def counting_slot(stats):
        # Counted inside the real slot, so this measures occupancy rather than
        # arrivals -- the cap only claims to bound the former.
        nonlocal live, peak
        with real_slot(stats):
            with guard:
                live += 1
                peak = max(peak, live)
            try:
                time.sleep(0.05)  # long enough that threads genuinely overlap
                yield
            finally:
                with guard:
                    live -= 1

    sparsevol_live.read_slot = counting_slot
    try:
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(post([SV_A]).status_code))
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        sparsevol_live.read_slot = real_slot

    assert results == [200] * 6
    assert peak <= 2, "ran {} reads at once, cap was 2".format(peak)


def test_queue_wait_is_reported(volume, slots):
    slots(4)
    stats = post([SV_A], fmt="json").json()["stats"]
    assert "queued_seconds" in stats
    assert stats["queued_seconds"] >= 0


def test_dataset_without_sparsevol_service_is_refused(volume):
    """'test' is a transform volume, so it is not a sparsevol dataset at all."""
    response = client.post(
        "transform-service/sparsevol/dataset/test/s/7/supervoxels",
        json={"supervoxels": ["1"]},
    )
    assert response.status_code == 422  # rejected by the dataset enum


def test_index_and_live_dataset_names_cannot_collide():
    """A name in both dicts would route to a backend nobody chose."""
    from app import main

    assert not set(config.SPARSEVOL_DATASOURCES) & set(main.SPARSEVOL_LIVE_DATASETS)


def test_backend_dispatch_follows_the_config():
    from app import main

    assert main.sparsevol_backend("test_segmentation") is sparsevol_live
    assert main.sparsevol_backend("test_index") is not sparsevol_live


def test_datasets_listing_reports_the_mode(volume):
    listed = client.get("transform-service/sparsevol/datasets").json()
    assert listed["test_segmentation"]["mode"] == "live"
    assert listed["wclee_aedes_brain"]["mode"] == "live"
    assert listed["test_index"]["mode"] == "index"


def test_chunk_layout_matches_the_ids_we_build(volume):
    layout = ChunkLayout(spatial_bits=8, layer_bits=8)
    np.testing.assert_array_equal(layout.decode([SV_A, SV_B]), [[0, 0, 0], [1, 0, 0]])


# --- the layer-2 cache ----------------------------------------------------


def test_the_second_request_reads_no_voxels(volume):
    """The whole point of the cache: the dense read happens once."""
    first = client.get(BASE + "/root/999")
    assert int(first.headers["X-Sparsevol-Voxels-Read"]) > 0
    assert first.headers["X-Sparsevol-L2-Computed"] == "2"

    second = client.get(BASE + "/root/999")
    assert second.headers["X-Sparsevol-L2-Cached"] == "2"
    assert second.headers["X-Sparsevol-L2-Computed"] == "0"
    assert second.headers["X-Sparsevol-Voxels-Read"] == "0"
    assert second.content == first.content


def test_the_cached_answer_is_the_same_answer(volume, monkeypatch):
    """Caching must not change the voxels, only what it cost to get them.

    Worth stating as a test because the two paths genuinely differ: uncached
    masks every chunk against the whole neuron at once, cached masks each chunk
    against one L2 node at a time and merges the runs afterwards.
    """
    data, _ = volume
    monkeypatch.setattr(config, "L2CacheEnabled", False)
    monkeypatch.setattr(l2cache, "_cache", None)
    uncached = runs_from(client.get(BASE + "/root/999"))

    monkeypatch.setattr(config, "L2CacheEnabled", True)
    monkeypatch.setattr(l2cache, "_cache", None)
    cached = runs_from(client.get(BASE + "/root/999"))

    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_array_equal(decode_rle(cached), truth_for(data, [SV_A, SV_B]))


def test_only_the_missing_nodes_are_recomputed(volume):
    """What makes an edit cheap: the untouched L2 nodes are already there."""
    cache = l2cache.get_cache()
    stats = sparsevol_live.LiveStats()
    known = sparsevol_live.compute_l2_fragments("test_segmentation", 0, [L2_A], stats)
    cache.put_many("test_segmentation", 0, known)

    response = client.get(BASE + "/root/999")
    assert response.headers["X-Sparsevol-L2-Cached"] == "1"
    assert response.headers["X-Sparsevol-L2-Computed"] == "1"
    # And only the one chunk holding the uncached node was read.
    assert response.headers["X-Sparsevol-Chunks"] == "1"


def test_two_nodes_in_one_chunk_are_attributed_separately(volume):
    """The chunk is read once, but its voxels still land in the right node.

    This is the case the per-node masking exists for. A single mask over the
    whole neuron would be cheaper here and completely useless: there would be
    no way to say afterwards which of the two nodes a voxel belonged to, and so
    nothing that could be cached against either of them.
    """
    data, _ = volume
    response = client.get(BASE + "/root/888")

    assert response.status_code == 200
    assert response.headers["X-Sparsevol-Chunks"] == "1"
    assert response.headers["X-Sparsevol-L2-Computed"] == "2"
    np.testing.assert_array_equal(
        decode_rle(runs_from(response)), truth_for(data, [SV_A, SV_OTHER])
    )

    cached = l2cache.get_cache().get_many("test_segmentation", 0, [L2_A, L2_OTHER])
    np.testing.assert_array_equal(decode_rle(cached[L2_A]), truth_for(data, [SV_A]))
    np.testing.assert_array_equal(
        decode_rle(cached[L2_OTHER]), truth_for(data, [SV_OTHER])
    )


def test_a_shared_node_is_reused_across_neurons(volume):
    """Two roots sharing an L2 node: the second gets it for free.

    The reason a cache keyed below the root is worth having at all -- a root
    key would treat these as entirely unrelated requests.
    """
    client.get(BASE + "/root/999")  # caches L2_A and L2_B
    response = client.get(BASE + "/root/888")  # shares L2_A

    assert response.headers["X-Sparsevol-L2-Cached"] == "1"
    assert response.headers["X-Sparsevol-L2-Computed"] == "1"


def test_a_full_cache_fails_every_request(volume):
    """Including requests it could have answered, and ones it never touches."""
    cache = l2cache.get_cache()
    cache.put_many("test_segmentation", 0, {L2_A: np.zeros((0, 4), dtype=np.int64)})
    cache.max_keys = 1

    root = client.get(BASE + "/root/999")
    assert root.status_code == 503
    assert "full" in root.json()["detail"]

    # The supervoxel endpoint does not use the cache at all, but a wedged cache
    # should take the service down visibly rather than by halves.
    assert post([SV_A]).status_code == 503


def test_the_cache_endpoint_reports_usage(volume):
    client.get(BASE + "/root/999")
    reported = client.get("transform-service/sparsevol/cache").json()

    assert reported["enabled"] is True
    assert reported["n_keys"] == 2
    assert reported["full"] is False
    assert reported["max_bytes"] == config.L2CacheMaxBytes


def test_the_cache_endpoint_says_when_caching_is_off(volume, monkeypatch):
    monkeypatch.setattr(config, "L2CacheEnabled", False)
    monkeypatch.setattr(l2cache, "_cache", None)
    assert client.get("transform-service/sparsevol/cache").json() == {"enabled": False}


def test_warming_up_leaves_nothing_to_compute(volume):
    """The warm-up and the request path have to agree on the key, or it is moot."""
    assert l2cache_warm.main(["test_segmentation", "999", "--scale", "0"]) == 0

    response = client.get(BASE + "/root/999")
    assert response.status_code == 200
    assert response.headers["X-Sparsevol-L2-Cached"] == "2"
    assert response.headers["X-Sparsevol-Voxels-Read"] == "0"


def test_a_dry_run_reads_nothing(volume):
    assert (
        l2cache_warm.main(["test_segmentation", "999", "--scale", "0", "--dry-run"]) == 0
    )
    assert l2cache.get_cache().counters() == (0, 0)


def test_warming_stops_when_the_cache_is_full(volume, monkeypatch):
    monkeypatch.setattr(config, "L2CacheMaxKeys", 1)
    # One chunk per pass, so the first pass fills it and the second is refused.
    code = l2cache_warm.main(
        ["test_segmentation", "999", "--scale", "0", "--chunks", "1"]
    )
    assert code == 2
