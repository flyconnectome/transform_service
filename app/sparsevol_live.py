"""Sparse voxels computed on the fly from a locally-stored segmentation.

The dense read happens here, per request, and that is the point rather than a
compromise. These watershed volumes sit on the same machine as the service, so
reading a few hundred chunks costs a local disk read; sending those same chunks
to the client would cost gigabytes over the wire. Doing the masking server-side
turns a multi-gigabyte download into a few hundred kilobytes of runs.

So this trades the client's bandwidth for the server's I/O, deliberately. What
makes that affordable is locality, not cleverness.

The dense read is then cached per layer-2 node (see :mod:`app.l2cache`), so a
neuron asked about twice, or two neurons sharing a branch, pay for it once.
Caching is optional and off-by-configuration: with it disabled every request
re-reads and re-sparsifies, which is what this service did originally and still
does for the supervoxel endpoint.

**How it knows where to read.** The segmentation is flat -- the stored labels
are watershed supervoxels -- but a PyChunkedGraph sits on top of it, and its
labels are graphene labels with the chunk position packed into their high bits.
So ``get_leaves(root, stop_layer=2)`` yields the neuron's occupied chunks
without reading a single voxel, and the read is confined to those chunks. A
supervoxel query needs even less: the chunk falls straight out of the ID
arithmetic, with no call to the graph at all.

The graph is used for the manifest and for chunk geometry only. Image data is
always read from the local volume, never through graphene.
"""

import threading
import time
from contextlib import contextmanager

import numpy as np
from fastapi import HTTPException
from concurrent.futures import ThreadPoolExecutor

from . import config
from . import datasource
from . import l2cache
from .chunks import ChunkLayout, chunk_bbox, chunk_boxes, l2_chunk_positions
from .rle import COORD_DTYPE, encode_rle, rle_voxel_count, union_rle

_graphs = {}
_layouts = {}
_graph_lock = threading.Lock()

# Reads are admitted through here rather than run on arrival. One request can
# hold a gigabyte of chunks, so the thing that needs limiting is how many are
# reading at once, not how fast each one goes.
_read_slots = threading.BoundedSemaphore(config.SparseVolMaxConcurrent)


@contextmanager
def read_slot(stats):
    """Hold one of the concurrent-read slots for the duration of a read.

    Waits briefly for a slot, so a burst queues rather than failing, then sheds
    load with a 503 and a Retry-After. Shedding is the kind thing to do here:
    the alternative is admitting work the machine cannot hold and having the
    kernel pick which worker dies.
    """
    started = time.time()
    if not _read_slots.acquire(timeout=config.SparseVolQueueSeconds):
        raise HTTPException(
            status_code=503,
            detail=(
                "Too many sparse volume reads in flight; waited {}s for a slot. "
                "Retry shortly.".format(config.SparseVolQueueSeconds)
            ),
            headers={"Retry-After": str(config.SparseVolQueueSeconds)},
        )
    stats.queued_seconds = time.time() - started
    try:
        yield
    finally:
        _read_slots.release()


class LiveStats:
    """What the request actually read, and what it managed to avoid sending."""

    def __init__(self):
        self.n_l2_nodes = 0
        # How the L2 nodes were answered. Together these add up to n_l2_nodes
        # on the cached path, and are the quickest read on whether the cache is
        # earning its disk: cached high and computed low is the whole point.
        self.n_l2_cached = 0
        self.n_l2_computed = 0
        self.n_supervoxels = 0
        self.n_chunks = 0
        self.n_chunks_empty = 0
        self.voxels_read = 0
        self.n_runs = 0
        self.n_voxels = 0
        self.queued_seconds = 0.0
        self.seconds = 0.0

    @property
    def reduction(self):
        """Dense voxels read per voxel kept -- what the client did not receive."""
        if self.n_voxels == 0:
            return 0.0
        return self.voxels_read / self.n_voxels

    def as_dict(self):
        out = {k: v for k, v in self.__dict__.items()}
        out["seconds"] = round(self.seconds, 4)
        out["queued_seconds"] = round(self.queued_seconds, 4)
        out["reduction"] = round(self.reduction, 1)
        return out


def get_live_info(dataset):
    """Config for a dataset that can be sparsified live."""
    info = datasource.get_datasource_info(dataset)
    if "sparsevol" not in info.get("services", []):
        raise HTTPException(
            status_code=400,
            detail="Dataset {} does not provide sparse volume services.".format(dataset),
        )
    if not info.get("graphene"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset {} has no graphene source configured, so there is no "
                "way to tell which chunks hold a segment.".format(dataset)
            ),
        )
    return info


def get_graph(dataset):
    """The graphene volume: manifest and chunk geometry only, never image data."""
    info = get_live_info(dataset)
    with _graph_lock:
        if dataset not in _graphs:
            from cloudvolume import CloudVolume

            try:
                _graphs[dataset] = CloudVolume(
                    info["graphene"], use_https=True, progress=False, fill_missing=True
                )
            except Exception as exc:
                # Nearly always a bad URL or missing CAVE credentials. Say so,
                # rather than surfacing a connection error as a server fault.
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Could not open the chunkedgraph for {} at {}: {}. "
                        "Check the graphene source and CAVE credentials.".format(
                            dataset, info["graphene"], exc
                        )
                    ),
                )
        return _graphs[dataset]


def get_layout(dataset):
    info = get_live_info(dataset)
    layout = info.get("chunk_layout")
    if layout is not None:
        return ChunkLayout(layout["spatial_bits"], layout.get("layer_bits", 8))

    cached = _layouts.get(dataset)
    if cached is not None:
        return cached

    # Resolved outside the lock, since get_graph takes the same one.
    meta = get_graph(dataset).meta
    with _graph_lock:
        _layouts[dataset] = ChunkLayout.from_metadata(meta)
        return _layouts[dataset]


def check_budget(info, boxes, stats):
    """Refuse a read that is too large, naming the limit that stopped it.

    A request that cannot finish should say so immediately rather than occupy a
    worker until something times out.
    """
    max_chunks = info.get("max_chunks", config.SparseVolMaxChunks)
    if max_chunks and len(boxes) > max_chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "Segment spans {} chunks, over the {} limit for this dataset. "
                "Use a coarser scale.".format(len(boxes), max_chunks)
            ),
        )

    voxels = int(sum(np.prod(box.size3()) for box in boxes))
    max_voxels = info.get("max_voxels", config.SparseVolMaxVoxels)
    if max_voxels and voxels > max_voxels:
        raise HTTPException(
            status_code=400,
            detail=(
                "Segment would require reading {:,} voxels, over the {:,} limit "
                "for this dataset. Use a coarser scale.".format(voxels, max_voxels)
            ),
        )
    return voxels


def info_workers(dataset):
    """Concurrent chunk reads. Each one holds a whole decompressed chunk."""
    info = datasource.get_datasource_info(dataset)
    return max(1, int(info.get("max_workers", config.SparseVolMaxWorkers)))


def read_and_mask(dataset, scale, boxes, labels, stats):
    """Read each chunk from the local volume and keep only ``labels``.

    The masking is what makes the whole exercise worthwhile: it is the step
    that turns chunks into a neuron, and doing it here rather than in the
    client is the difference between sending gigabytes and sending kilobytes.
    """
    store = datasource.get_datastore(dataset, scale)
    domain = store.domain
    labels = np.asarray(labels, dtype=np.uint64)

    def read(box):
        lo = np.maximum(np.asarray(box.minpt, dtype=np.int64), domain.inclusive_min[:3])
        hi = np.minimum(np.asarray(box.maxpt, dtype=np.int64), domain.exclusive_max[:3])
        if np.any(hi <= lo):
            return 0, np.zeros((0, 3), dtype=COORD_DTYPE)

        block = store[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].read().result()
        block = np.asarray(block)
        if block.ndim == 4:
            block = block[..., 0]

        keep = _mask_to_labels(block, labels)
        # argwhere hands back int64. Narrowed straight away, because these
        # arrays are what accumulate across every chunk of the neuron and then
        # get concatenated, deduplicated and sorted -- each of those steps pays
        # for the width. The intermediate is one chunk's worth and short-lived.
        return int(block.size), (np.argwhere(keep) + lo).astype(COORD_DTYPE)

    pieces = []
    workers = info_workers(dataset)
    # The slot is taken here rather than at the top of the request so that the
    # graph calls and the budget check -- which hold nothing -- are not counted
    # against the limit. Only the part that holds chunks in memory queues.
    with read_slot(stats), ThreadPoolExecutor(max_workers=workers) as pool:
        for read_voxels, coords in pool.map(read, boxes):
            stats.voxels_read += read_voxels
            if coords.shape[0] == 0:
                stats.n_chunks_empty += 1
            else:
                pieces.append(coords)

    if not pieces:
        return np.zeros((0, 3), dtype=COORD_DTYPE)
    return np.concatenate(pieces, axis=0)


def _mask_to_labels(block, labels):
    """Boolean mask of the voxels carrying one of ``labels``."""
    try:
        import fastremap

        # mask_except zeroes everything else in one pass over the block, which
        # matters when the block is 100x the size of what we keep.
        return fastremap.mask_except(block, list(labels), in_place=False, value=0) != 0
    except ImportError:
        return np.isin(block, labels)


def _to_runs(coords, stats, started):
    """Deduplicate, sort and encode -- the shape every entry point returns."""
    if coords.shape[0]:
        # Chunk boxes round outward onto the mip grid at coarse scales and can
        # overlap their neighbours, so repeats are expected, not exceptional.
        coords = np.unique(coords, axis=0)
    runs = encode_rle(coords, assume_unique=True)

    stats.n_runs = len(runs)
    stats.n_voxels = rle_voxel_count(runs)
    stats.seconds = time.time() - started
    return runs, stats


def l2_manifests(graph, l2_ids, workers):
    """``{l2_id: supervoxels}``, one chunkedgraph call each, fetched concurrently.

    There is no batch endpoint -- ``get_leaves`` takes a single node -- so a
    cold neuron costs one HTTP GET per L2 node. That is the price of being able
    to attribute a voxel to an L2 node at all: a supervoxel ID reveals its
    chunk by arithmetic, but never its owner, and a chunk routinely holds
    several of a neuron's L2 nodes.

    It is paid once per node ever, since the runs it produces are then cached.
    """
    bounds = graph.meta.bounds(0)

    def fetch(l2_id):
        leaves = graph.get_leaves(int(l2_id), bounds, 0)
        return int(l2_id), np.asarray(leaves, dtype=np.uint64).ravel()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(fetch, [int(i) for i in l2_ids]))


def read_and_split(dataset, scale, groups, stats):
    """Read each chunk once and attribute its voxels to the L2 node that owns them.

    ``groups`` is ``[(box, [(l2_id, supervoxels), ...]), ...]``. Grouping by
    chunk is what keeps this honest: a neuron passing through a chunk as three
    branches has three L2 nodes there, and reading the chunk once for all of
    them is the difference between this and the naive per-node loop.
    """
    store = datasource.get_datastore(dataset, scale)
    domain = store.domain

    def read(group):
        box, members = group
        lo = np.maximum(np.asarray(box.minpt, dtype=np.int64), domain.inclusive_min[:3])
        hi = np.minimum(np.asarray(box.maxpt, dtype=np.int64), domain.exclusive_max[:3])
        empty = [(l2_id, np.zeros((0, 3), dtype=COORD_DTYPE)) for l2_id, _ in members]
        if np.any(hi <= lo):
            return 0, empty

        block = store[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].read().result()
        block = np.asarray(block)
        if block.ndim == 4:
            block = block[..., 0]

        # One masking pass per L2 node, rather than a single pass building an
        # owner map for all of them. A chunk holds only a handful of a neuron's
        # nodes, so the extra passes are cheap, while an owner map would need
        # full-width temporaries several times the size of the block -- and the
        # block is already the thing this service budgets its memory around.
        out = []
        for l2_id, supervoxels in members:
            keep = _mask_to_labels(block, supervoxels)
            out.append((l2_id, (np.argwhere(keep) + lo).astype(COORD_DTYPE)))
        return int(block.size), out

    coords = {}
    workers = info_workers(dataset)
    with read_slot(stats), ThreadPoolExecutor(max_workers=workers) as pool:
        for read_voxels, members in pool.map(read, groups):
            stats.voxels_read += read_voxels
            if not any(c.shape[0] for _, c in members):
                stats.n_chunks_empty += 1
            coords.update(members)
    return coords


def compute_l2_fragments(dataset, scale, l2_ids, stats, enforce_budget=True):
    """Runs for each of ``l2_ids``, read and sparsified from the local volume.

    Returns ``{l2_id: (M, 4) runs}`` covering every id asked for, including the
    ones that turn out to be empty -- see :meth:`L2Cache.put_many` for why
    those are worth recording.

    ``enforce_budget`` is off for the offline warm-up, which deliberately reads
    far more chunks per call than any single request is allowed to.
    """
    info = get_live_info(dataset)
    graph = get_graph(dataset)
    layout = get_layout(dataset)

    workers = max(1, int(info.get("manifest_workers", config.L2CacheManifestWorkers)))
    try:
        manifests = l2_manifests(graph, l2_ids, workers)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not resolve layer-2 supervoxel manifests: {}".format(exc),
        )
    stats.n_supervoxels += sum(len(s) for s in manifests.values())

    groups = {}
    for l2_id, supervoxels in manifests.items():
        position = tuple(int(v) for v in layout.decode([l2_id])[0])
        groups.setdefault(position, []).append((l2_id, supervoxels))

    boxes = []
    fragments = {}
    for position, members in sorted(groups.items()):
        box = chunk_bbox(graph.meta, position, mip=scale)
        if box.subvoxel():
            # Clipped away entirely against the volume bounds. That is a real
            # answer, not a failure, so it is recorded as empty rather than
            # left to be recomputed on every future request.
            for l2_id, _ in members:
                fragments[l2_id] = np.zeros((0, 4), dtype=np.int64)
            continue
        boxes.append((box, members))

    stats.n_chunks += len(boxes)
    if enforce_budget:
        check_budget(info, [box for box, _ in boxes], stats)

    coords = read_and_split(dataset, scale, boxes, stats) if boxes else {}
    for l2_id, points in coords.items():
        fragments[l2_id] = encode_rle(points)
    return fragments


def root_to_runs(dataset, scale, root_id):
    """Sparse voxels for a root ID, computed from the local segmentation.

    ``stop_layer=2`` gives the neuron's layer-2 nodes without reading a voxel.
    From there the answer is either assembled from cached per-node runs, or
    computed by reading the chunks those nodes occupy -- see the two functions
    below for why those are separate paths rather than one.
    """
    info = get_live_info(dataset)
    started = time.time()
    stats = LiveStats()

    cache = l2cache.get_cache()
    if cache is not None:
        # Before any work, and even for a request that would have been served
        # entirely from cache: a wedged cache should be impossible to miss.
        cache.check_health()

    graph = get_graph(dataset)
    layout = get_layout(dataset)

    try:
        l2_ids = graph.get_leaves(int(root_id), graph.meta.bounds(0), 0, stop_layer=2)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Could not resolve root {}: {}".format(root_id, exc)
        )

    l2_ids = np.unique(np.asarray(l2_ids, dtype=np.uint64).ravel())
    if l2_ids.size == 0:
        raise HTTPException(
            status_code=404,
            detail="Root {} has no layer-2 nodes. Is it a current root ID?".format(
                root_id
            ),
        )
    stats.n_l2_nodes = len(l2_ids)

    if cache is None:
        return _root_runs_uncached(
            dataset, scale, root_id, l2_ids, info, graph, layout, stats, started
        )
    return _root_runs_cached(dataset, scale, l2_ids, cache, stats, started)


def _root_runs_uncached(
    dataset, scale, root_id, l2_ids, info, graph, layout, stats, started
):
    """Mask every chunk against the whole root at once, attributing nothing.

    Kept for when caching is off, because it is strictly cheaper there: one
    chunkedgraph call for the entire manifest, against one per L2 node. Nothing
    it produces could be cached anyway -- a single mask over the whole neuron
    cannot say which L2 node a given voxel belonged to.
    """
    try:
        supervoxels = graph.get_leaves(int(root_id), graph.meta.bounds(0), 0)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Could not resolve root {}: {}".format(root_id, exc)
        )

    if len(supervoxels) == 0:
        raise HTTPException(
            status_code=404,
            detail="Root {} has no supervoxels. Is it a current root ID?".format(root_id),
        )
    stats.n_supervoxels = len(supervoxels)

    positions = l2_chunk_positions(layout, l2_ids)
    boxes = chunk_boxes(graph.meta, positions, mip=scale)
    stats.n_chunks = len(boxes)
    check_budget(info, boxes, stats)

    coords = read_and_mask(dataset, scale, boxes, supervoxels, stats)
    return _to_runs(coords, stats, started)


def _root_runs_cached(dataset, scale, l2_ids, cache, stats, started):
    """Serve what is cached, compute the rest, and remember it.

    A fully cached neuron reads no image data and makes no manifest call at
    all: the only chunkedgraph traffic is the ``stop_layer=2`` lookup that got
    us here.
    """
    fragments = cache.get_many(dataset, scale, l2_ids)
    stats.n_l2_cached = len(fragments)

    missing = [int(i) for i in l2_ids.tolist() if int(i) not in fragments]
    stats.n_l2_computed = len(missing)

    if missing:
        computed = compute_l2_fragments(dataset, scale, missing, stats)
        cache.put_many(dataset, scale, computed)
        fragments.update(computed)

    # Merged rather than concatenated: L2 nodes meet along chunk boundaries, so
    # two of them routinely hold halves of what is really one unbroken run.
    runs = union_rle(list(fragments.values()))
    stats.n_runs = len(runs)
    stats.n_voxels = rle_voxel_count(runs)
    stats.seconds = time.time() - started
    return runs, stats


def supervoxels_to_runs(dataset, scale, sv_ids):
    """Sparse voxels for explicit supervoxel IDs, with no call to the graph.

    The chunk position is arithmetic on the ID, and the labels to keep are the
    IDs themselves, so this path never contacts the graph server at all.
    """
    info = get_live_info(dataset)
    started = time.time()
    stats = LiveStats()

    # This path never touches the cache -- an arbitrary set of supervoxels does
    # not line up with L2 nodes -- but it fails alongside it anyway, so a full
    # cache takes the whole service down visibly rather than half of it.
    cache = l2cache.get_cache()
    if cache is not None:
        cache.check_health()

    sv_ids = np.unique(np.asarray(sv_ids, dtype=np.uint64))
    sv_ids = sv_ids[sv_ids != 0]
    stats.n_supervoxels = len(sv_ids)
    if len(sv_ids) == 0:
        return _to_runs(np.zeros((0, 3), dtype=COORD_DTYPE), stats, started)

    layout = get_layout(dataset)
    positions = np.unique(layout.decode(sv_ids), axis=0)
    boxes = chunk_boxes(get_graph(dataset).meta, positions, mip=scale)
    stats.n_chunks = len(boxes)
    check_budget(info, boxes, stats)

    coords = read_and_mask(dataset, scale, boxes, sv_ids, stats)
    return _to_runs(coords, stats, started)
