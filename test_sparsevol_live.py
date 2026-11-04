"""Tests for sparse voxels computed live from a locally-stored segmentation.

A small neuroglancer_precomputed volume is written to disk and read back
through the real endpoint, so the chunk geometry, the masking and the encoding
are all exercised for real. Only the chunkedgraph is substituted -- it is the
one piece that is genuinely remote.
"""

import io
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from fastapi import HTTPException

from app import (
    config,
    datasource,
    l2cache,
    l2cache_warm,
    sparsevol,
    sparsevol_live,
)
from app.chunks import ChunkLayout, chunk_boxes
from app.rle import COORD_DTYPE, decode_rle, decode_rle_labelled, encode_rle_labelled
from conftest import runs_from, sort_rows

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

# Which supervoxels each root owns, for the reverse (bounded) lookup the box
# endpoint uses. 999 and 888 are the same roots the manifest tests use, so the
# two paths describe the same neurons.
ROOT_SUPERVOXELS = {999: [SV_A, SV_B], 888: [SV_A, SV_OTHER]}


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
        if int(root_id) in ROOT_SUPERVOXELS and stop_layer is None and bbox is not None:
            # The reverse lookup: only the supervoxels whose voxels actually
            # fall inside the box. This is the behaviour the box endpoint leans
            # on, so the stand-in has to model it rather than return everything.
            lo = np.asarray(bbox.minpt, dtype=np.int64)
            hi = np.asarray(bbox.maxpt, dtype=np.int64)
            inside = []
            for sv in ROOT_SUPERVOXELS[int(root_id)]:
                region = REGIONS[sv]
                starts = np.array([s.start for s in region])
                stops = np.array([s.stop for s in region])
                if np.all(stops > lo) and np.all(starts < hi):
                    inside.append(sv)
            return np.array(inside, dtype=np.uint64)
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


def truth_for(data, labels, window=None, down=None):
    """The voxels these labels own, in the order the decoders emit.

    ``window`` restricts to a half-open box and ``down`` floors onto a coarse
    grid -- the box endpoint's OR-downsampling written out by hand, rather than
    computed the way the service computes it.
    """
    mask = np.zeros(data.shape, dtype=bool)
    for sv in labels:
        mask |= data == sv
    if window is not None:
        (x0, y0, z0), (x1, y1, z1) = window
        keep = np.zeros(data.shape, dtype=bool)
        keep[x0:x1, y0:y1, z0:z1] = True
        mask &= keep
    coords = np.argwhere(mask)
    if down is not None:
        coords = coords // np.asarray(down)
    return sort_rows(coords)


def post(supervoxels, fmt=None):
    return client.post(
        BASE + "/supervoxels",
        json={"supervoxels": [str(sv) for sv in supervoxels]},
        params={"fmt": fmt} if fmt else None,
    )


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


def test_an_over_budget_request_asks_the_graph_for_no_manifests(volume, monkeypatch):
    """Refuse first, ask later.

    A manifest is one HTTP GET per layer-2 node against a shared production
    chunkedgraph, and a cold neuron has hundreds. A request that is going to be
    rejected for being over budget must not spend them on its way to the 400 --
    and it does not have to, since which chunk a node occupies is packed into
    its ID rather than something the graph has to be asked for.
    """
    _, graph = volume
    monkeypatch.setitem(config.DATASOURCES["test_segmentation"], "max_chunks", 1)
    del graph.leaf_calls[:]

    response = client.get(BASE + "/root/999")
    assert response.status_code == 400

    # The stop_layer=2 lookup that found the nodes is expected; one call per
    # node afterwards is what this guards against.
    assert graph.leaf_calls == [(999, 2)]


def test_a_node_clipped_away_by_the_bounds_needs_no_manifest(volume):
    """Planning from the IDs alone also spares the graph the nodes that read
    nothing: a chunk falling outside the volume resolves to empty without ever
    being asked what it contains."""
    outside = label(2, (7, 7, 7), 1)
    stats = sparsevol_live.LiveStats()
    _, graph = volume
    del graph.leaf_calls[:]

    fragments = sparsevol_live.compute_l2_fragments(
        "test_segmentation", 0, [outside], stats
    )
    assert fragments[outside].shape == (0, 4)
    assert graph.leaf_calls == []
    assert stats.n_chunks == 0


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
    np.testing.assert_array_equal(sort_rows(coords), truth_for(data, [SV_A, SV_B]))

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


# --- box lookups ----------------------------------------------------------
#
# A different question from the ones above: not "where is this segment?" but
# "what is inside this box?", for several objects at once, with the answer
# OR-downsampled and labelled.


def box_runs(bbox, roots=None, objects=None, down=None):
    """Call the box path directly and return its (M, 5) runs."""
    return sparsevol_live.box_to_runs(
        "test_segmentation", 0, bbox, roots=roots, objects=objects, down=down
    )


def rows_for_label(runs, label):
    """The coarse voxels one label claims, as sorted (N, 3) coordinates."""
    rows = decode_rle_labelled(runs)
    return sort_rows(rows[rows[:, 3] == label][:, :3])


def test_box_returns_each_object_separately(volume):
    """The core claim: one read, one response, every object labelled apart."""
    data, _ = volume
    bbox = [[0, 0, 0], [128, 128, 64]]
    runs, stats = box_runs(bbox, objects=[[SV_A], [SV_OTHER]])

    assert runs.shape[1] == 5
    assert stats.n_objects == 2
    np.testing.assert_array_equal(
        rows_for_label(runs, 1), truth_for(data, [SV_A], window=bbox)
    )
    np.testing.assert_array_equal(
        rows_for_label(runs, 2), truth_for(data, [SV_OTHER], window=bbox)
    )


def test_labels_are_one_based_indices_into_the_request(volume):
    """Position in the request, not the ID -- the response carries no IDs."""
    data, _ = volume
    runs, _ = box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_OTHER], [SV_A]])
    assert set(np.unique(runs[:, 4])) == {1, 2}
    np.testing.assert_array_equal(
        rows_for_label(runs, 1),
        truth_for(data, [SV_OTHER], window=[[0, 0, 0], [128, 128, 64]]),
    )


def test_voxels_are_clipped_to_the_box(volume):
    """SV_A spans x 10-20; a box ending at 15 must return only half of it."""
    data, _ = volume
    bbox = [[0, 0, 0], [15, 128, 64]]
    runs, _ = box_runs(bbox, objects=[[SV_A]])

    coords = rows_for_label(runs, 1)
    assert coords[:, 0].max() == 14
    np.testing.assert_array_equal(coords, truth_for(data, [SV_A], window=bbox))


def test_or_downsampling_keeps_every_object_in_a_coarse_voxel(volume):
    """The whole reason for OR rather than mode.

    SV_A spans x 10-20 and SV_OTHER x 30-35, so the coarse voxel covering fine
    x 16-31 holds four voxels of SV_A against two of SV_OTHER, in the same y
    and z. SV_OTHER is the minority there; taking a majority would erase it,
    which is exactly how a thin neurite beside a thick one disappears. OR
    keeps both.
    """
    data, _ = volume
    bbox = [[0, 0, 0], [128, 128, 64]]
    down = [16, 1, 1]
    runs, _ = box_runs(bbox, objects=[[SV_A], [SV_OTHER]], down=down)

    for label, sv in ((1, SV_A), (2, SV_OTHER)):
        np.testing.assert_array_equal(
            rows_for_label(runs, label), truth_for(data, [sv], window=bbox, down=down)
        )

    # And they really do share a coarse voxel, or the test above proves nothing.
    shared = set(map(tuple, rows_for_label(runs, 1))) & set(
        map(tuple, rows_for_label(runs, 2))
    )
    assert shared


def test_coarse_coordinates_are_absolute_not_box_relative(volume):
    """floor(voxel / down) of the absolute voxel, so a cache survives the box.

    Two boxes that both contain the object must describe it with the same
    coordinates; if they were relative to the corner, they would not.
    """
    wide, _ = box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]], down=[2, 2, 1])
    narrow, _ = box_runs([[8, 4, 0], [24, 12, 8]], objects=[[SV_A]], down=[2, 2, 1])
    np.testing.assert_array_equal(rows_for_label(wide, 1), rows_for_label(narrow, 1))


def test_downsampled_coordinates_do_not_depend_on_block_boundaries(
    volume, monkeypatch
):
    """A coarse voxel split across two read blocks is still one coarse voxel."""
    data, _ = volume
    bbox = [[0, 0, 0], [128, 128, 64]]
    down = [4, 4, 1]
    whole, _ = box_runs(bbox, objects=[[SV_A], [SV_B]], down=down)

    # One block per storage chunk, so the box is read in many pieces.
    monkeypatch.setattr(config, "SparseVolBoxBlockVoxels", 1)
    split, _ = box_runs(bbox, objects=[[SV_A], [SV_B]], down=down)

    np.testing.assert_array_equal(whole, split)
    np.testing.assert_array_equal(
        rows_for_label(whole, 1), truth_for(data, [SV_A], window=bbox, down=down)
    )


def test_the_packed_keys_arrive_in_the_order_the_encoder_expects(volume):
    """The box path tells the encoder its input is already sorted and unique.

    That holds only because the key packs (label, z, y, x) in that bit order,
    so one np.unique over the keys leaves exactly the order the encoder would
    have sorted into. If it ever stopped holding, the encoder would not raise
    -- it would quietly emit more runs than necessary, each still correct --
    so the claim is checked here rather than trusted.
    """
    runs, _ = box_runs(
        [[0, 0, 0], [128, 128, 64]], objects=[[SV_A], [SV_OTHER], [SV_B]], down=[4, 2, 1]
    )
    rows = decode_rle_labelled(runs)
    resorted = encode_rle_labelled(rows[:, :3], rows[:, 3])
    np.testing.assert_array_equal(runs, resorted)


def test_the_coarse_grid_matches_flooring_every_voxel_by_hand():
    """The packing against its definition, over random boxes and block splits.

    The service never materialises per-voxel coordinates -- it packs a label
    and a flat coarse index into one integer and unravels them at the end -- so
    the thing worth checking is that the round trip equals the obvious thing:
    floor each voxel onto the coarse grid and collect the distinct (label,
    voxel) pairs. Random block splits because the packing has to be independent
    of where a block boundary happens to fall.
    """
    rng = np.random.default_rng(5)
    for _ in range(10):
        lo = rng.integers(0, 50, 3)
        hi = lo + rng.integers(5, 40, 3)
        down = rng.integers(1, 6, 3)
        n_objects = int(rng.integers(1, 8))
        grid = sparsevol_live.CoarseGrid.for_box(lo, hi, down, n_objects)

        volume = rng.integers(0, n_objects + 1, size=tuple(hi - lo)).astype(np.uint64)
        cuts = [
            np.unique(np.concatenate([[0], rng.integers(1, s, 2), [s]]))
            for s in hi - lo
        ]

        pieces = []
        for i in range(len(cuts[0]) - 1):
            for j in range(len(cuts[1]) - 1):
                for k in range(len(cuts[2]) - 1):
                    corner = np.array([cuts[0][i], cuts[1][j], cuts[2][k]])
                    block = volume[
                        cuts[0][i] : cuts[0][i + 1],
                        cuts[1][j] : cuts[1][j + 1],
                        cuts[2][k] : cuts[2][k + 1],
                    ]
                    pieces.append(grid.pack(block.reshape(-1), block.shape, lo + corner))

        keys, _ = sparsevol_live._merge_keys(pieces)
        coords, labels = grid.unpack(keys)
        got = {(int(label), tuple(c)) for label, c in zip(labels, coords)}

        occupied = np.argwhere(volume > 0)
        want = {
            (int(volume[tuple(v)]), tuple((v + lo) // down)) for v in occupied
        }
        assert got == want


def test_the_labelling_fallback_agrees_with_fastremap(volume, monkeypatch):
    """Both label a block identically, so the optional dependency is optional.

    fastremap remaps a block in one hash pass; without it the fallback has to
    sort the block to build an inverse index. Two code paths behind an import
    is two chances to be wrong, and only one of them runs on any given machine.
    """
    # Real graphene supervoxel ids, not small integers. The fast path clears
    # non-members by magnitude -- an unmapped value comes back as its own id,
    # and a graphene label carries its layer in the top bits, so it cannot be
    # mistaken for an object label. That invariant is the thing under test as
    # much as the two paths agreeing, so the fixture has to honour it.
    rng = np.random.default_rng(0)
    pool = np.array(
        [supervoxel((0, 0, 0), i) for i in range(1, 12)], dtype=np.uint64
    )
    block = rng.choice(pool, size=(8, 8, 8)).astype(np.uint64)
    ids = np.sort(pool[:6])
    labels = np.array([1, 2, 2, 3, 3, 3], dtype=np.uint64)

    assert ids.min() > len(labels), "ids must be graphene labels, not indices"
    assert sparsevol_live.fastremap is not None
    fast = sparsevol_live._label_block(block.copy(), ids, labels, 3)

    monkeypatch.setattr(sparsevol_live, "fastremap", None)
    slow = sparsevol_live._label_block(block.copy(), ids, labels, 3)

    np.testing.assert_array_equal(fast, slow)
    # Both labelled something and left something out, or this proves nothing.
    assert np.count_nonzero(fast)
    assert np.count_nonzero(fast == 0)
    assert set(np.unique(fast)) <= {0, 1, 2, 3}


def test_an_object_with_nothing_in_the_box_returns_no_runs(volume):
    """An answer, not an error: most candidates have left the box."""
    runs, stats = box_runs([[0, 0, 0], [25, 128, 64]], objects=[[SV_A], [SV_B]])
    assert set(np.unique(runs[:, 4])) == {1}
    assert stats.n_objects == 2


def test_roots_are_resolved_against_the_box(volume):
    """The reverse lookup: one bounded get_leaves per object.

    Root 999 owns SV_A and SV_B, which are far apart in x. A box around SV_A
    only must come back with SV_A's voxels, and the graph must have been asked
    with the box rather than with the whole volume.
    """
    data, graph = volume
    del graph.leaf_calls[:]
    bbox = [[0, 0, 0], [25, 128, 64]]
    runs, stats = box_runs(bbox, roots=[999])

    assert [root for root, _ in graph.leaf_calls] == [999]
    assert stats.n_supervoxels == 1
    np.testing.assert_array_equal(
        rows_for_label(runs, 1), truth_for(data, [SV_A], window=bbox)
    )


def test_a_box_outside_the_volume_is_clipped_away(volume):
    runs, stats = box_runs([[500, 500, 500], [520, 520, 520]], objects=[[SV_A]])
    assert runs.shape == (0, 5)
    assert stats.voxels_read == 0


def test_only_the_blocks_covering_the_box_are_read(volume):
    """The box bounds the read -- not the chunks the objects occupy."""
    _, stats = box_runs([[0, 0, 0], [16, 16, 16]], objects=[[SV_A]])
    assert stats.voxels_read < int(np.prod(SHAPE))


@pytest.mark.parametrize(
    "bbox, down, message",
    [
        # JSON carries no integer width, so a value past int64 reaches numpy as
        # an ordinary Python int and raises OverflowError rather than ValueError.
        ([[0, 0, 0], [10**30, 10, 10]], None, "corner"),
        ([[0, 0, 0], [10, 10, 10]], [10**30, 1, 1], "down must be"),
        ([[0, 0, 0], [10, 10]], None, "corner"),
        ([[10, 0, 0], [10, 10, 10]], None, "half-open"),
        ([[0, 0, 0], [10, 10, 10]], [0, 1, 1], "down factors"),
        ([[0, 0, 0], [10, 10, 10]], [1, 1], "down must be"),
        ([[0, 0, 0], [10, 10, 10]], [999, 1, 1], "down factors"),
    ],
)
def test_the_request_shape_is_validated(volume, bbox, down, message):
    with pytest.raises(HTTPException) as caught:
        box_runs(bbox, objects=[[SV_A]], down=down)
    assert caught.value.status_code == 400
    assert message in caught.value.detail


def test_exactly_one_of_roots_and_objects(volume):
    for kwargs in ({}, {"roots": [999], "objects": [[SV_A]]}):
        with pytest.raises(HTTPException) as caught:
            box_runs([[0, 0, 0], [10, 10, 10]], **kwargs)
        assert caught.value.status_code == 400
        assert "exactly one" in caught.value.detail


def test_a_supervoxel_claimed_by_two_objects_is_refused(volume):
    """Under OR semantics, attributing it to the first would silently drop it."""
    with pytest.raises(HTTPException) as caught:
        box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A], [SV_A, SV_B]])
    assert caught.value.status_code == 400


def test_a_box_over_the_edge_is_charged_only_for_what_it_reads(volume, monkeypatch):
    """An open end near the surface of the brain makes a box that overhangs.

    The part outside the volume is never read, so charging the budget for it
    would refuse a request that is perfectly affordable.
    """
    inside = int(np.prod(SHAPE))
    monkeypatch.setattr(config, "SparseVolBoxMaxVoxels", inside)

    # Twice the volume on every axis; only the eighth inside it can be read.
    runs, _ = box_runs([[0, 0, 0], SHAPE.tolist()], objects=[[SV_A]])
    overhanging, _ = box_runs([[0, 0, 0], (SHAPE * 2).tolist()], objects=[[SV_A]])
    np.testing.assert_array_equal(runs, overhanging)


def test_a_root_listed_twice_is_refused_by_name(volume):
    """Each root is one label, so a repeat has no meaning.

    It would otherwise surface as the supervoxel-collision error, which is true
    of the leaves but tells the caller nothing it can act on.
    """
    with pytest.raises(HTTPException) as caught:
        box_runs([[0, 0, 0], [128, 128, 64]], roots=[999, 999])
    assert caught.value.status_code == 400
    assert "more than once" in caught.value.detail


def test_the_box_budget_is_enforced(volume, monkeypatch):
    monkeypatch.setattr(config, "SparseVolBoxMaxVoxels", 100)
    with pytest.raises(HTTPException) as caught:
        box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]])
    assert caught.value.status_code == 400
    assert "over the" in caught.value.detail


def test_the_answer_size_is_capped_independently_of_the_box(volume, monkeypatch):
    """The read ceiling cannot imply this one.

    A sparse box and a dense box of the same size cost the same to read and
    differ by orders of magnitude in what they return, so the pairs that
    survive need their own limit.
    """
    monkeypatch.setattr(config, "SparseVolBoxMaxCoarseVoxels", 5)
    with pytest.raises(HTTPException) as caught:
        box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]])
    assert caught.value.status_code == 400
    assert "coarse voxels" in caught.value.detail

    # Generous enough and the same request succeeds, so the cap is what spoke.
    monkeypatch.setattr(config, "SparseVolBoxMaxCoarseVoxels", 10**9)
    runs, _ = box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]])
    assert len(runs)


def test_chunkedgraph_calls_are_bounded_across_requests(monkeypatch):
    """The ceiling is on the process, not on one request.

    SparseVolBoxLookupWorkers limits one request's concurrency; this limits how
    many requests can be doing that at once. Without it the only bound on what
    the service points at a shared chunkedgraph is how many requests arrive.
    """
    monkeypatch.setattr(
        sparsevol_live, "_graph_slots", threading.BoundedSemaphore(3)
    )
    live = []
    peak = []
    lock = threading.Lock()

    def slow_call():
        with lock:
            live.append(1)
            peak.append(len(live))
        time.sleep(0.02)
        with lock:
            live.pop()
        return "ok"

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(lambda _: sparsevol_live.call_graph(slow_call), range(32)))

    assert results == ["ok"] * 32
    assert max(peak) <= 3, "saw {} concurrent calls past a ceiling of 3".format(max(peak))


def test_a_transient_chunkedgraph_failure_is_retried(monkeypatch):
    """One blip should not throw away every other lookup in the request."""
    monkeypatch.setattr(config, "SparseVolChunkedGraphRetries", 2)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("connection reset")
        return "ok"

    assert sparsevol_live.call_graph(flaky) == "ok"
    assert len(calls) == 3


def test_a_client_error_from_the_chunkedgraph_is_not_retried(monkeypatch):
    """A bad root stays bad; retrying it only wastes round trips."""
    monkeypatch.setattr(config, "SparseVolChunkedGraphRetries", 3)
    calls = []

    class Response:
        status_code = 400

    class BadRequest(Exception):
        response = Response()

    def rejected():
        calls.append(1)
        raise BadRequest("no such root")

    with pytest.raises(BadRequest):
        sparsevol_live.call_graph(rejected)
    assert len(calls) == 1


def test_a_lookup_that_never_recovers_still_fails(monkeypatch):
    """Retries are bounded, and the slot is given back each time."""
    monkeypatch.setattr(config, "SparseVolChunkedGraphRetries", 2)
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        sparsevol_live.call_graph(broken)
    assert len(calls) == 3

    # The semaphore is back to full, so a later request is not starved.
    assert sparsevol_live.call_graph(lambda: "ok") == "ok"


def test_the_supervoxel_budget_is_enforced(volume, monkeypatch):
    """The one ceiling that bounds the request rather than the work.

    An `objects` request spells out every supervoxel, so without this nothing
    caps how large a body the service will accept -- and the proxy's own limit
    would reject it as an opaque 413 rather than a 400 naming the problem.
    """
    monkeypatch.setattr(config, "SparseVolBoxMaxSupervoxels", 1)
    with pytest.raises(HTTPException) as caught:
        box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A, SV_B]])
    assert caught.value.status_code == 400
    assert "supervoxels" in caught.value.detail

    # Refused before anything is read: the lookup is all it cost.
    _, stats = box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]])
    assert stats.voxels_read > 0


def test_many_objects_cost_little_and_pack_correctly(volume):
    """A thousand objects in one box, which is what a mask store sends.

    The block is labelled once however many objects are asked about, so the
    thing worth checking is not speed but that the label still fits beside the
    voxel position in the packed key, and that the few objects with voxels in
    the box are still found among the many without.
    """
    wanted = {400: SV_A, 900: SV_B}
    objects = [[] for _ in range(1000)]
    for position, sv in wanted.items():
        objects[position] = [sv]

    data, _ = volume
    bbox = [[0, 0, 0], [128, 128, 64]]
    runs, stats = box_runs(bbox, objects=objects)

    assert stats.n_objects == 1000
    assert set(np.unique(runs[:, 4])) == {p + 1 for p in wanted}
    for position, sv in wanted.items():
        np.testing.assert_array_equal(
            rows_for_label(runs, position + 1), truth_for(data, [sv], window=bbox)
        )


def test_the_object_budget_is_enforced(volume, monkeypatch):
    monkeypatch.setattr(config, "SparseVolBoxMaxObjects", 1)
    with pytest.raises(HTTPException) as caught:
        box_runs([[0, 0, 0], [128, 128, 64]], objects=[[SV_A], [SV_B]])
    assert caught.value.status_code == 400
    assert "objects" in caught.value.detail


def test_the_index_backend_refuses_box_queries():
    """Honest 501 rather than a wrong answer from the wrong data structure."""
    with pytest.raises(HTTPException) as caught:
        sparsevol.box_to_runs("test_index", 0, [[0, 0, 0], [10, 10, 10]], roots=[1])
    assert caught.value.status_code == 501


# --- the box endpoint -----------------------------------------------------


def post_box(bbox, roots=None, objects=None, down=None, fmt=None):
    body = {"bbox": bbox}
    if roots is not None:
        body["roots"] = [str(root) for root in roots]
    if objects is not None:
        body["objects"] = [{"supervoxels": [str(sv) for sv in g]} for g in objects]
    if down is not None:
        body["down"] = down
    return client.post(BASE + "/box", json=body, params={"fmt": fmt} if fmt else None)


def test_the_box_endpoint_reaches_the_service(volume):
    """What the route adds: the request body arrives, the runs come back.

    The voxels themselves are the service's claim and are checked against dense
    truth above; what is route-level is that `objects` and `roots` in JSON reach
    the right arguments, and that the response describes its own width.
    """
    data, _ = volume
    bbox = [[0, 0, 0], [25, 128, 64]]

    by_object = post_box(bbox, objects=[[SV_A]])
    assert by_object.status_code == 200
    assert by_object.headers["X-Sparsevol-Columns"] == "5"

    np.testing.assert_array_equal(
        rows_for_label(runs_from(by_object), 1), truth_for(data, [SV_A], window=bbox)
    )

    # And the same answer asked for by root, which exercises the other branch
    # of the request model. IDs cross the wire as strings, since they are uint64
    # and a JavaScript client loses precision past 2^53.
    assert SV_A > 2**53
    np.testing.assert_array_equal(runs_from(post_box(bbox, roots=[999])), runs_from(by_object))


def test_box_formats_agree(volume):
    bbox = [[0, 0, 0], [128, 128, 64]]
    runs = runs_from(post_box(bbox, objects=[[SV_A], [SV_B]]))

    as_json = post_box(bbox, objects=[[SV_A], [SV_B]], fmt="json").json()
    np.testing.assert_array_equal(np.array(as_json["runs"]), runs)

    as_npy = np.load(io.BytesIO(post_box(bbox, objects=[[SV_A], [SV_B]], fmt="npy").content))
    np.testing.assert_array_equal(as_npy, runs)

    # coords spells out one row per (label, voxel) pair: x, y, z, label. The
    # header describes the body, so it says 4 here where 'rle' said 5.
    coords_response = post_box(bbox, objects=[[SV_A], [SV_B]], fmt="coords")
    assert coords_response.headers["X-Sparsevol-Columns"] == "4"
    np.testing.assert_array_equal(runs_from(coords_response), decode_rle_labelled(runs))


def test_every_sparsevol_answer_may_be_cached_forever(volume):
    """Roots, supervoxels, the box and down are all immutable.

    Proofreading mints new root IDs rather than changing the voxels under an
    existing one, so no sparsevol answer can go stale. Only `/root` is a GET,
    where a shared cache will act on this; on the two POST endpoints it is
    advisory, but the claim is the same one and is worth stating uniformly.
    """
    for response in (
        post_box([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]]),
        post([SV_A]),
        client.get(BASE + "/root/999"),
    ):
        assert response.status_code == 200
        assert "immutable" in response.headers["Cache-Control"]


def test_a_json_result_past_the_limit_is_refused(volume, monkeypatch):
    """The one format that cannot be serialized concurrently.

    orjson holds the GIL for its whole run, so a huge JSON response blocks
    every other client this worker is serving -- which is why it is refused
    rather than served slowly. The binary formats have no such limit.
    """
    monkeypatch.setattr(config, "SparseVolMaxJsonRuns", 1)

    refused = post_box([[0, 0, 0], [128, 128, 64]], objects=[[SV_A]], fmt="json")
    assert refused.status_code == 400
    assert "fmt=json" in refused.json()["detail"]

    # The binary formats answer the same request unaffected.
    for fmt in (None, "npy", "coords"):
        assert post_box(
            [[0, 0, 0], [128, 128, 64]], objects=[[SV_A]], fmt=fmt
        ).status_code == 200

    # And it applies to the segment endpoints too, not just the box.
    assert post([SV_A], fmt="json").status_code == 400


def test_the_segment_endpoints_still_report_four_columns(volume):
    """The labelled layout must not leak into the endpoints that predate it."""
    response = post([SV_A])
    assert response.headers["X-Sparsevol-Columns"] == "4"
    assert runs_from(response).shape[1] == 4

    # And their coords form is still three columns, with no label.
    coords = post([SV_A], fmt="coords")
    assert coords.headers["X-Sparsevol-Columns"] == "3"
    np.testing.assert_array_equal(
        runs_from(coords), decode_rle(runs_from(response))
    )


def test_the_box_endpoint_validates_its_request(volume):
    # Shape errors are the request model's to catch, and come back as 422.
    assert client.post(BASE + "/box", json={"bbox": [[0, 0], [1, 1]]}).status_code == 422
    assert (
        client.post(
            BASE + "/box", json={"bbox": [[0, 0, 0], [8, 8, 8]], "down": [1, 1]}
        ).status_code
        == 422
    )

    # The service's own checks are tested directly above; what matters here is
    # that its HTTPException reaches the client as a 400 rather than a 500.
    assert post_box([[8, 0, 0], [8, 8, 8]], objects=[[SV_A]]).status_code == 400

    bad = client.post(
        BASE + "/box", json={"bbox": [[0, 0, 0], [8, 8, 8]], "roots": ["-1"]}
    )
    assert bad.status_code == 400
    assert "unsigned 64-bit" in bad.json()["detail"]


def test_stats_headers_cover_the_box_accounting(volume):
    """BoxStats counters reach the client without main.py naming any of them."""
    response = post_box([[0, 0, 0], [128, 128, 64]], objects=[[SV_A], [SV_B]])
    assert response.headers["X-Sparsevol-Objects"] == "2"
    assert len(runs_from(response))
    for name in ("Blocks", "Supervoxels", "Lookup-Seconds", "Voxels-Read", "Runs"):
        assert "X-Sparsevol-" + name in response.headers


def test_the_index_backend_refuses_the_box_route():
    """Dispatch reaches the stub, so the refusal is an HTTP answer not a 500."""
    response = client.post(
        "transform-service/sparsevol/dataset/test_index/s/0/box",
        json={"bbox": [[0, 0, 0], [8, 8, 8]], "roots": ["1"]},
    )
    assert response.status_code == 501
    assert "precomputed index" in response.json()["detail"]
