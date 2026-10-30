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

try:
    import fastremap
except ImportError:  # pragma: no cover - cloud-volume depends on it
    # Kept as a name rather than an import inside the hot functions. It arrives
    # as a dependency of cloud-volume, which this module cannot work without,
    # so in practice this never binds None -- but resolving it here means a
    # genuine ImportError from somewhere inside a labelling pass cannot be
    # mistaken for a missing dependency and silently take the slow path.
    fastremap = None

from . import config
from . import datasource
from . import l2cache
from .chunks import ChunkLayout, chunk_bbox, chunk_boxes, l2_chunk_positions
from .rle import (
    COORD_DTYPE,
    encode_rle,
    encode_rle_labelled,
    rle_voxel_count,
    union_rle,
)

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


class _Stats:
    """Shared accounting for the read paths, and what becomes of it.

    Subclasses declare their own counters; this holds only what every path
    reports the same way. ``app.main.stats_headers`` turns ``as_dict`` into
    response headers without knowing which class it was given, so a counter
    added to a subclass appears on that endpoint and nowhere else -- which is
    the point, since the paths genuinely account for different things.

    ``_rounded`` names the float fields, so a subclass that adds one gets it
    rounded without touching ``as_dict``.
    """

    _rounded = ("seconds", "queued_seconds")

    def __init__(self):
        # Defined here because this class already depends on them: reduction
        # reads two, as_dict rounds the others, and read_slot writes
        # queued_seconds on whatever it is handed. A subclass that forgot one
        # would otherwise fail inside as_dict rather than at construction.
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
        out = dict(self.__dict__)
        for key in self._rounded:
            out[key] = round(out[key], 4)
        out["reduction"] = round(self.reduction, 1)
        return out


class LiveStats(_Stats):
    """What the request actually read, and what it managed to avoid sending."""

    def __init__(self):
        super().__init__()
        self.n_l2_nodes = 0
        # How the L2 nodes were answered. Together these add up to n_l2_nodes
        # on the cached path, and are the quickest read on whether the cache is
        # earning its disk: cached high and computed low is the whole point.
        self.n_l2_cached = 0
        self.n_l2_computed = 0
        self.n_supervoxels = 0
        self.n_chunks = 0
        self.n_chunks_empty = 0


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


def clip_to_domain(domain, lo, hi):
    """``[lo, hi)`` trimmed to what the volume actually holds.

    The domain carries a channel axis the boxes do not, hence the ``[:3]``.
    """
    return (
        np.maximum(np.asarray(lo, dtype=np.int64), domain.inclusive_min[:3]),
        np.minimum(np.asarray(hi, dtype=np.int64), domain.exclusive_max[:3]),
    )


def _read_blocks(dataset, scale, items, kernel, stats):
    """Read each box from the local volume and reduce it with ``kernel``.

    Every read path in this module does the same four things -- clip the box to
    what the volume actually holds, read it, drop the channel axis, and do that
    for many boxes at once under a bounded pool. What differs between them is
    only what happens to a block once it is in memory: mask it to one label
    set, split it between the layer-2 nodes that own it, or pack it into coarse
    keys. That part is ``kernel(block, lo, payload)``, where ``payload`` is
    whatever the caller attached to the box.

    A block clipped away entirely is handed to the kernel as an empty array
    rather than skipped, so every caller gets one result per box and none of
    them has to carry a second code path for the edge of the volume.

    The caller takes the concurrency slot, not this function, because how far
    it should extend differs. The box path holds it across the assembly that
    follows, where the merged keys and the coordinates unpacked from them are
    together several times a block. The segment paths release it as soon as the
    last block lands and then concatenate, deduplicate and encode unadmitted --
    which is what they did before this function existed, and is worth revisiting
    against the per-slot budget in ``config``, but is not something to change
    here.
    """
    store = datasource.get_datastore(dataset, scale)
    domain = store.domain

    def read(item):
        box, payload = item
        lo, hi = clip_to_domain(domain, box.minpt, box.maxpt)
        if np.any(hi <= lo):
            return 0, kernel(np.zeros((0, 0, 0), dtype=np.uint64), lo, payload)

        block = store[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].read().result()
        block = np.asarray(block)
        if block.ndim == 4:
            block = block[..., 0]
        return int(block.size), kernel(block, lo, payload)

    results = []
    with ThreadPoolExecutor(max_workers=info_workers(dataset)) as pool:
        for read_voxels, value in pool.map(read, items):
            stats.voxels_read += read_voxels
            results.append(value)
    return results


def read_and_mask(dataset, scale, boxes, labels, stats):
    """Read each chunk from the local volume and keep only ``labels``.

    The masking is what makes the whole exercise worthwhile: it is the step
    that turns chunks into a neuron, and doing it here rather than in the
    client is the difference between sending gigabytes and sending kilobytes.
    """
    labels = np.asarray(labels, dtype=np.uint64)

    def kernel(block, lo, _):
        keep = _mask_to_labels(block, labels)
        # argwhere hands back int64. Narrowed straight away, because these
        # arrays are what accumulate across every chunk of the neuron and then
        # get concatenated, deduplicated and sorted -- each of those steps pays
        # for the width. The intermediate is one chunk's worth and short-lived.
        return (np.argwhere(keep) + lo).astype(COORD_DTYPE)

    # The slot is taken here rather than at the top of the request so that the
    # graph calls and the budget check -- which hold nothing -- are not counted
    # against the limit. Only the part that holds chunks in memory queues.
    with read_slot(stats):
        blocks = _read_blocks(
            dataset, scale, [(box, None) for box in boxes], kernel, stats
        )

    pieces = []
    for coords in blocks:
        if coords.shape[0] == 0:
            stats.n_chunks_empty += 1
        else:
            pieces.append(coords)

    if not pieces:
        return np.zeros((0, 3), dtype=COORD_DTYPE)
    return np.concatenate(pieces, axis=0)


def _mask_to_labels(block, labels):
    """Boolean mask of the voxels carrying one of ``labels``."""
    if fastremap is None:
        return np.isin(block, labels)
    # mask_except zeroes everything else in one pass over the block, which
    # matters when the block is 100x the size of what we keep.
    return fastremap.mask_except(block, list(labels), in_place=False, value=0) != 0


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

    def kernel(block, lo, members):
        # One masking pass per L2 node, rather than a single pass building an
        # owner map for all of them. A chunk holds only a handful of a neuron's
        # nodes, so the extra passes are cheap, while an owner map would need
        # full-width temporaries several times the size of the block -- and the
        # block is already the thing this service budgets its memory around.
        out = []
        for l2_id, supervoxels in members:
            keep = _mask_to_labels(block, supervoxels)
            out.append((l2_id, (np.argwhere(keep) + lo).astype(COORD_DTYPE)))
        return out

    with read_slot(stats):
        blocks = _read_blocks(dataset, scale, groups, kernel, stats)

    coords = {}
    for members in blocks:
        if not any(c.shape[0] for _, c in members):
            stats.n_chunks_empty += 1
        coords.update(members)
    return coords


def compute_l2_fragments(dataset, scale, l2_ids, stats, enforce_budget=True):
    """Runs for each of ``l2_ids``, read and sparsified from the local volume.

    Returns ``{l2_id: (M, 4) runs}`` covering every id asked for, including the
    ones that turn out to be empty -- see :meth:`L2Cache.put_many` for why
    those are worth recording.

    The read is planned, and refused, before the graph is asked anything. That
    ordering matters: a manifest is one HTTP GET per node against a shared
    production server, and a cold neuron has hundreds of nodes, so a request
    that is going to be rejected for being over budget must not spend them on
    its way to the 400. Nothing in the plan needs a manifest -- which chunk a
    node lives in is packed into its ID -- so none of it has to be paid for up
    front.

    ``enforce_budget`` is off for the offline warm-up, which deliberately reads
    far more chunks per call than any single request is allowed to.
    """
    info = get_live_info(dataset)
    graph = get_graph(dataset)
    layout = get_layout(dataset)

    l2_ids = [int(i) for i in l2_ids]
    positions = layout.decode(np.asarray(l2_ids, dtype=np.uint64))

    groups = {}
    for l2_id, position in zip(l2_ids, positions):
        groups.setdefault(tuple(int(v) for v in position), []).append(l2_id)

    fragments = {}
    boxes = []
    for position, members in sorted(groups.items()):
        box = chunk_bbox(graph.meta, position, mip=scale)
        if box.subvoxel():
            # Clipped away entirely against the volume bounds. That is a real
            # answer, not a failure, so it is recorded as empty rather than
            # left to be recomputed on every future request. These nodes need
            # no manifest either, so they never reach the graph at all.
            for l2_id in members:
                fragments[l2_id] = np.zeros((0, 4), dtype=np.int64)
            continue
        boxes.append((box, members))

    stats.n_chunks += len(boxes)
    if enforce_budget:
        check_budget(info, [box for box, _ in boxes], stats)
    if not boxes:
        return fragments

    workers = max(1, int(info.get("manifest_workers", config.L2CacheManifestWorkers)))
    wanted = [l2_id for _, members in boxes for l2_id in members]
    try:
        manifests = l2_manifests(graph, wanted, workers)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not resolve layer-2 supervoxel manifests: {}".format(exc),
        )
    stats.n_supervoxels += sum(len(s) for s in manifests.values())

    coords = read_and_split(
        dataset,
        scale,
        [(box, [(i, manifests[i]) for i in members]) for box, members in boxes],
        stats,
    )
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


# --- box lookups -----------------------------------------------------------
#
# A different question from the two above, and worth saying why it is not just
# a variation on them. Those ask "where is this segment?", and the answer is
# bounded by what the segment occupies -- the chunks it passes through. This one
# asks "what is inside this box?", for up to a hundred objects at once, and the
# answer is bounded by the box.
#
# That inverts the economics. Reading the box once costs the same whether one
# object is asked about or a hundred, so the per-object cost collapses; and the
# chunks the objects occupy are irrelevant, because anything outside the box is
# not wanted even if the object extends there. Hence a separate read path, its
# own ceilings, and labels -- the one thing the segment endpoints never need,
# since there the whole answer belongs to one segment by construction.


class BoxStats(_Stats):
    """What a box request read, and for how many objects."""

    _rounded = ("seconds", "queued_seconds", "lookup_seconds")

    def __init__(self):
        super().__init__()
        self.n_objects = 0
        self.n_objects_empty = 0
        self.n_supervoxels = 0
        self.n_blocks = 0
        self.n_blocks_empty = 0
        self.lookup_seconds = 0.0


class CoarseGrid:
    """The downsampled grid a box answer lives in, and its packed key.

    Surviving voxels are accumulated as one int64 each: the object label in the
    high bits, a flat index into the coarse box below it. That packing does
    three jobs at once.

    It gives OR-downsampling for free -- a coarse voxel holding fine voxels of
    three objects produces three distinct keys, so all three survive, where a
    majority vote would keep one and erase exactly the thin neurites this
    endpoint exists to find.

    It sorts correctly by construction: the flat index runs x fastest and z
    slowest, so sorting the keys sorts by (label, z, y, x), which is the order
    :func:`encode_rle_labelled` would have sorted into -- hence it can be told
    to skip its own sort.

    And it keeps coordinates absolute, ``floor(voxel / down)`` of the real
    voxel rather than an offset into the box, so a result does not move when
    the box does and blocks need no alignment to the coarse grid.

    Both directions of the packing live here because they are one decision: a
    mismatched unpack yields plausible coordinates rather than an error.
    """

    def __init__(self, down, origin, shape, shift):
        self.down = down
        self.origin = origin
        self.shape = shape
        self.shift = shift

    @classmethod
    def for_box(cls, lo, hi, down, n_objects):
        origin = lo // down
        shape = (hi - 1) // down - origin + 1
        shift = int(np.prod(shape)).bit_length()
        # Checked rather than assumed: a box and an object count that together
        # overflow the key would silently mix two objects' voxels.
        if shift + int(n_objects + 1).bit_length() >= 63:
            raise HTTPException(
                status_code=400,
                detail="Box is too large to label at this downsampling; coarsen 'down'.",
            )
        return cls(down, origin, shape, shift)

    def pack(self, labelled, shape, lo):
        """Coarse keys for the labelled voxels of one block.

        ``labelled`` is the block flattened, one object index per voxel.
        """
        flat = np.flatnonzero(labelled)
        if flat.size == 0:
            return np.zeros(0, dtype=np.int64)
        # Reinterpreted rather than converted: the gathered values are small
        # object labels, so the bit pattern is the same either way, and the
        # shift then happens in place instead of in two more N-wide arrays.
        keys = labelled[flat].view(np.int64)
        keys <<= self.shift

        # One table per axis, giving the coarse coordinate of every fine index
        # along it, already multiplied by its stride in the flat key. Built at
        # the size of the block's edge, so the per-voxel work is three lookups
        # and two adds -- no (N, 3) coordinate array, and no separate division
        # and origin pass over every surviving voxel.
        strides = (1, int(self.shape[0]), int(self.shape[0] * self.shape[1]))
        tables = [
            ((np.arange(dim, dtype=np.int64) + lo[axis]) // self.down[axis]
             - self.origin[axis]) * strides[axis]
            for axis, dim in enumerate(shape)
        ]

        # Unravelled in place, each axis folded in as soon as it is peeled off.
        coord = flat % shape[2]
        keys += tables[2][coord]
        flat //= shape[2]
        coord = flat % shape[1]
        keys += tables[1][coord]
        flat //= shape[1]
        keys += tables[0][flat]

        # Survivors arrive in flat order and the last axis is the fastest, so
        # the voxels that fall into one coarse voxel along it are adjacent
        # here. Collapsing equal neighbours first hands the sort roughly
        # down[z] times fewer keys; np.unique then finishes the job for the
        # duplicates that are not adjacent.
        keep = np.empty(len(keys), dtype=bool)
        keep[0] = True
        np.not_equal(keys[1:], keys[:-1], out=keep[1:])
        return np.unique(keys[keep])

    def unpack(self, keys):
        """Inverse of :meth:`pack`: absolute coarse coordinates and labels.

        Coordinates come back at ``COORD_DTYPE`` rather than int64. This is the
        one array in a box request that holds a row per voxel instead of a row
        per run, and :mod:`app.rle` keeps whatever width it is handed, so the
        choice here is the one that decides the encoder's peak.
        """
        labels = (keys >> self.shift).astype(np.uint32)
        flat = keys & ((1 << self.shift) - 1)

        coords = np.empty((len(flat), 3), dtype=COORD_DTYPE)
        # Each divmod writes its quotient back over its own input, so the
        # unravel needs one scratch array rather than three.
        remainder = np.empty_like(flat)
        np.divmod(flat, int(self.shape[0]), out=(flat, remainder))
        coords[:, 0] = remainder
        np.divmod(flat, int(self.shape[1]), out=(flat, remainder))
        coords[:, 1] = remainder
        coords[:, 2] = flat
        coords += self.origin.astype(COORD_DTYPE)
        return coords, labels


def parse_box(bbox):
    """``(lo, hi)`` as int64 arrays, from a half-open ``[[x0,y0,z0],[x1,y1,z1]]``."""
    try:
        box = np.asarray(bbox, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=400, detail="bbox must be two integer corners")

    if box.shape != (2, 3):
        raise HTTPException(
            status_code=400,
            detail="bbox must be [[x0, y0, z0], [x1, y1, z1]], got shape {}".format(
                tuple(box.shape)
            ),
        )
    lo, hi = box[0], box[1]
    if np.any(hi <= lo):
        raise HTTPException(
            status_code=400,
            detail="bbox must be half-open with hi > lo on every axis",
        )
    return lo, hi


def parse_down(down):
    """The OR-downsampling factor, as a (3,) int64 array of positive factors."""
    if down is None:
        return np.ones(3, dtype=np.int64)
    try:
        factors = np.asarray(down, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=400, detail="down must be three integers")

    if factors.shape != (3,):
        raise HTTPException(
            status_code=400, detail="down must be [dx, dy, dz], got {}".format(down)
        )
    if np.any(factors < 1) or np.any(factors > config.SparseVolBoxMaxDown):
        raise HTTPException(
            status_code=400,
            detail="down factors must be between 1 and {}".format(
                config.SparseVolBoxMaxDown
            ),
        )
    return factors


def box_to_mip0(graph, lo, hi, scale):
    """The box as mip-0 voxels, for ``get_leaves``' bounds.

    ``bbox_to_mip`` rounds outward -- floor on the near corner, ceil on the far
    one -- which is the behaviour this needs and the reason to use it rather
    than scale the corners here. The chunkedgraph is asked which supervoxels of
    a root fall in the box, and the voxels are masked against the box again
    during the read, so a bounds slightly too large costs nothing while one
    slightly too small silently drops supervoxels along the edge. It is also
    the conversion :func:`app.chunks.chunk_bbox` already uses, so a dataset with
    an odd pyramid is converted the same way everywhere.

    CloudVolume's ``get_leaves`` passes its bbox to CAVE verbatim and uses its
    ``mip`` argument only to clamp, so the conversion has to happen here. It is
    returned as a Bbox, which is what every other call to the graph in this
    module passes -- CAVE's own client takes a (3, 2) array instead, so this is
    the one place the two conventions could be confused.
    """
    from cloudvolume.lib import Bbox

    return graph.meta.bbox_to_mip(Bbox(lo, hi), mip=scale, to_mip=0)


def leaves_in_box(graph, roots, bounds, workers):
    """``get_leaves(root, bounds=box)`` for each root, concurrently.

    The reverse lookup, and the whole reason this endpoint takes roots rather
    than supervoxels: it is one call per object whatever the box holds, and it
    is exact, so the result is the object's true membership rather than
    whatever the voxels happen to say.
    """

    def fetch(root):
        return np.asarray(graph.get_leaves(int(root), bounds, 0), dtype=np.uint64).ravel()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            return list(pool.map(fetch, [int(r) for r in roots]))
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not resolve roots against the box: {}".format(exc),
            )


def label_lookup(groups):
    """Sorted supervoxel ids and the label each one carries.

    Two parallel arrays, which is the shape both labelling paths want: a
    remap table for fastremap, or a ``searchsorted`` target for the fallback.
    Labels are 1-based, so 0 is free to mean "belongs to none of them".
    """
    ids, labels = [], []
    for i, supervoxels in enumerate(groups):
        supervoxels = np.unique(np.asarray(supervoxels, dtype=np.uint64))
        supervoxels = supervoxels[supervoxels != 0]
        ids.append(supervoxels)
        labels.append(np.full(len(supervoxels), i + 1, dtype=np.uint64))

    if not ids:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint64)

    ids = np.concatenate(ids)
    labels = np.concatenate(labels)
    order = np.argsort(ids, kind="stable")
    ids, labels = ids[order], labels[order]

    # A supervoxel has exactly one root, so roots cannot collide. Explicit
    # object lists can, and attributing such a voxel to whichever group was
    # listed first would quietly drop it from the other -- which, under OR
    # semantics, is the one thing this endpoint promises not to do.
    if np.any(ids[1:] == ids[:-1]):
        raise HTTPException(
            status_code=400,
            detail="A supervoxel is listed under more than one object.",
        )
    return ids, labels


def block_boxes(store, lo, hi, target_voxels):
    """Tile ``[lo, hi)`` into boxes aligned to the volume's storage chunks.

    Aligned so no stored chunk is decompressed by two blocks, and sized so each
    block is roughly ``target_voxels`` -- the unit this service budgets memory
    in, since a block is held whole while it is labelled.

    Unlike :func:`app.chunks.chunk_boxes` this tiles the request's own box on
    the *storage* grid; the graph's chunk grid is a different thing entirely and
    is irrelevant here, since the box bounds the read rather than the segments.
    """
    from cloudvolume.lib import Bbox

    chunk = np.maximum(np.asarray(store.chunk_layout.read_chunk.shape[:3]), 1)
    origin = np.asarray(store.domain.inclusive_min[:3], dtype=np.int64)

    # Grow the block by whole chunks on whichever axis is relatively smallest,
    # until it is big enough or it covers the box.
    size = chunk.copy()
    while np.prod(size) < target_voxels and np.any(size < (hi - lo)):
        axis = int(np.argmin(size / chunk))
        size[axis] += chunk[axis]

    starts = [
        np.arange(origin[a] + (lo[a] - origin[a]) // size[a] * size[a], hi[a], size[a])
        for a in range(3)
    ]
    return [
        Bbox(np.maximum([x, y, z], lo), np.minimum(np.array([x, y, z]) + size, hi))
        for x in starts[0]
        for y in starts[1]
        for z in starts[2]
    ]


def check_box_budget(info, lo, hi, n_objects):
    """Refuse a box that is too large, naming the limit that stopped it.

    Charged against the box as it will actually be read -- already clipped to
    the volume -- so a box hanging over the edge of the brain is not refused
    for voxels that do not exist.
    """
    voxels = int(np.prod(np.maximum(hi - lo, 0)))
    max_voxels = info.get("box_max_voxels", config.SparseVolBoxMaxVoxels)
    if max_voxels and voxels > max_voxels:
        raise HTTPException(
            status_code=400,
            detail=(
                "Box covers {:,} voxels at this scale, over the {:,} limit for "
                "this dataset. Use a smaller box or a coarser scale.".format(
                    voxels, max_voxels
                )
            ),
        )

    max_objects = info.get("box_max_objects", config.SparseVolBoxMaxObjects)
    if max_objects and n_objects > max_objects:
        raise HTTPException(
            status_code=400,
            detail="Request covers {} objects, over the {} limit.".format(
                n_objects, max_objects
            ),
        )


def _object_labels(distinct, ids, labels):
    """The label each of ``distinct`` carries, 0 for a non-member."""
    if len(ids) == 0:
        return np.zeros(len(distinct), dtype=np.uint64)
    position = np.clip(np.searchsorted(ids, distinct), 0, len(ids) - 1)
    return np.where(ids[position] == distinct, labels[position], 0).astype(np.uint64)


def _label_block(block, ids, labels, n_objects):
    """Every voxel of a block as its object's label, flattened. 0 for none.

    The fast path remaps the block against the lookup in one hash pass, in
    place. fastremap leaves values it was not given alone, so a voxel belonging
    to none of the objects comes back holding its own supervoxel id -- and
    those are graphene labels, which carry the layer in their top bits and are
    therefore always astronomically larger than an object label, so one
    comparison clears them. Taking the block's distinct values first instead
    would cost a second pass over the block and a second block-sized array, on
    top of the block this service already budgets its memory around.

    The fallback has to sort the block to build an inverse index, measured at
    ~5.7x the block in transient memory.
    """
    # remap_from_array_kv is fused-type: the block, the keys and the values
    # must all be the same width. Graphene labels need 64 bits so every
    # segmentation this service reads is uint64, but a dataset that is not
    # takes the fallback rather than a 500 -- and the fallback is also what
    # keeps this function's result uint64 either way, which CoarseGrid.pack
    # relies on when it reinterprets the labels as int64.
    if fastremap is None or block.dtype != np.uint64:
        distinct, inverse = np.unique(block, return_inverse=True)
        return _object_labels(distinct, ids, labels)[inverse.reshape(-1)]

    out = fastremap.remap_from_array_kv(block.reshape(-1), ids, labels)
    out[out > n_objects] = 0
    return out


class _KeptBudget:
    """Refuses a box whose answer grows past what the service will hold.

    The read ceiling cannot imply this one: a sparse box and a dense box of the
    same size cost the same to read and differ by orders of magnitude in what
    they return. Counted across blocks as they land, so a runaway box is
    refused partway through rather than once everything is already in memory.
    """

    def __init__(self, limit):
        self.limit = int(limit or 0)
        self.kept = 0
        self._lock = threading.Lock()

    def add(self, n):
        if not self.limit:
            return
        with self._lock:
            self.kept += n
            if self.kept > self.limit:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Box holds over {:,} labelled coarse voxels, past the "
                        "{:,} limit for this dataset. Use a coarser 'down' or a "
                        "smaller box.".format(self.kept, self.limit)
                    ),
                )


def _merge_keys(pieces):
    """``(keys, n_empty)``: one sorted, distinct key array from the blocks'.

    Takes ownership of ``pieces`` and empties it as it goes. Those arrays are
    the second largest thing the request holds and the concatenation below
    doubles them, so a reference kept elsewhere would put the peak exactly
    where the next step needs the room -- which is why the caller passes the
    list straight in rather than naming it. ``np.unique`` is spelled out for
    the same reason: it would flatten into a copy before sorting.
    """
    n_empty = sum(1 for piece in pieces if piece.size == 0)
    pieces[:] = [piece for piece in pieces if piece.size]
    if not pieces:
        return np.zeros(0, dtype=np.int64), n_empty

    merged = np.concatenate(pieces)
    del pieces[:]
    # Stable, because every piece is already ascending: this merges a handful
    # of sorted runs rather than sorting the concatenation from scratch.
    merged.sort(kind="stable")

    keep = np.empty(len(merged), dtype=bool)
    keep[0] = True
    np.not_equal(merged[1:], merged[:-1], out=keep[1:])
    if keep.all():
        # The usual case: blocks are disjoint in fine space, so two of them can
        # only produce the same key where a coarse voxel straddles their
        # boundary -- and never at all when down is 1. Worth testing for,
        # because the alternative is copying the whole array to delete nothing.
        return merged, n_empty
    return merged[keep], n_empty


def _box_result(runs, stats, started):
    """Finish a box request: the counters every exit from it has to set."""
    stats.n_runs = len(runs)
    stats.n_voxels = rle_voxel_count(runs)
    stats.seconds = time.time() - started
    return runs, stats


def box_to_runs(dataset, scale, bbox, roots=None, objects=None, down=None):
    """Per-object sparse voxels inside a box, OR-downsampled and labelled.

    Returns ``(M, 5)`` runs of ``x, y, z, length, label``, where ``label`` is
    the 1-based index into the objects as given. Coordinates are in the
    downsampled grid, absolute rather than relative to the box.

    Exactly one of ``roots`` and ``objects`` is given: roots are resolved here
    through ``get_leaves(root, bounds=box)``, while ``objects`` is a list of
    supervoxel lists for a caller that already holds them.

    An object with no voxels in the box simply contributes no runs. That is an
    answer, not an error -- most of the candidates around an open end will have
    left the box before it.
    """
    info = get_live_info(dataset)
    started = time.time()
    stats = BoxStats()

    # Validated before anything is opened or asked, so a request that cannot
    # succeed fails as the 400 it is rather than as whatever the cache or the
    # graph happens to say first.
    if (roots is None) == (objects is None):
        raise HTTPException(
            status_code=400, detail="Give exactly one of 'roots' and 'objects'."
        )
    lo, hi = parse_box(bbox)
    down = parse_down(down)

    cache = l2cache.get_cache()
    if cache is not None:
        # This path never reads or writes the cache -- a box-clipped answer is
        # not a whole L2 node, so it could not be stored -- but it fails
        # alongside it, so a wedged cache takes the service down visibly rather
        # than leaving one endpoint quietly working.
        cache.check_health()

    # Clipped to what the volume actually holds, before the budget rather than
    # after it. A box hanging over the edge is ordinary -- an open end near the
    # surface of the brain produces one -- and the part inside is still a real
    # answer, so it should not be charged for the part outside.
    store = datasource.get_datastore(dataset, scale)
    read_lo, read_hi = clip_to_domain(store.domain, lo, hi)

    stats.n_objects = len(roots if roots is not None else objects)
    check_box_budget(info, read_lo, read_hi, stats.n_objects)

    groups = objects
    if roots is not None:
        # Each root becomes one label, so a repeated one has no meaning and
        # would otherwise surface as label_lookup's supervoxel collision --
        # which is true of the leaves but says nothing the caller can act on.
        if len({int(root) for root in roots}) != len(roots):
            raise HTTPException(
                status_code=400,
                detail="A root is listed more than once; objects must be distinct.",
            )

        # Only the root path needs the graph. Opening it on the objects path
        # would give a caller that already holds its supervoxels a dependency,
        # and a failure mode, that it deliberately does without.
        graph = get_graph(dataset)
        lookup_started = time.time()
        groups = leaves_in_box(
            graph,
            roots,
            box_to_mip0(graph, lo, hi, scale),
            max(1, int(info.get("box_lookup_workers", config.SparseVolBoxLookupWorkers))),
        )
        stats.lookup_seconds = time.time() - lookup_started

    ids, labels = label_lookup(groups)
    stats.n_supervoxels = len(ids)

    # Checked after the lookup, since a `roots` request does not know its own
    # supervoxel count until the graph has answered -- but still before any
    # voxel is read, so an over-large request costs nothing but the lookup.
    max_supervoxels = info.get(
        "box_max_supervoxels", config.SparseVolBoxMaxSupervoxels
    )
    if max_supervoxels and len(ids) > max_supervoxels:
        raise HTTPException(
            status_code=400,
            detail=(
                "Request covers {:,} supervoxels, over the {:,} limit for this "
                "dataset. Split the objects across several requests, or ask by "
                "root and let the service resolve them.".format(
                    len(ids), max_supervoxels
                )
            ),
        )
    stats.n_objects_empty = sum(1 for group in groups if len(group) == 0)

    if np.any(read_hi <= read_lo) or len(ids) == 0:
        # Nothing to read, or nothing to look for. Both are real answers.
        return _box_result(np.zeros((0, 5), dtype=np.int64), stats, started)

    grid = CoarseGrid.for_box(read_lo, read_hi, down, stats.n_objects)
    kept = _KeptBudget(
        info.get("box_max_coarse_voxels", config.SparseVolBoxMaxCoarseVoxels)
    )
    blocks = block_boxes(
        store,
        read_lo,
        read_hi,
        max(1, int(info.get("box_block_voxels", config.SparseVolBoxBlockVoxels))),
    )
    stats.n_blocks = len(blocks)

    def kernel(block, block_lo, _):
        if block.size == 0:
            return np.zeros(0, dtype=np.int64)
        labelled = _label_block(block, ids, labels, stats.n_objects)
        keys = grid.pack(labelled, block.shape, block_lo)
        kept.add(len(keys))
        return keys

    # The slot covers the assembly as well as the read. The merged keys and the
    # coordinates unpacked from them are together several times larger than any
    # single block, so releasing it when the last block lands would let the
    # biggest allocation of the request run unadmitted -- which is the one
    # thing the slot exists to prevent.
    with read_slot(stats):
        keys, stats.n_blocks_empty = _merge_keys(
            _read_blocks(dataset, scale, [(box, None) for box in blocks], kernel, stats)
        )
        if keys.size == 0:
            return _box_result(np.zeros((0, 5), dtype=np.int64), stats, started)

        coords, labels = grid.unpack(keys)
        del keys
        # The keys were distinct and ascending, and they pack (label, z, y, x)
        # in that bit order, so the encoder has nothing left to sort or
        # deduplicate -- which is most of what it would otherwise cost.
        runs = encode_rle_labelled(
            coords, labels, assume_unique=True, assume_sorted=True
        )

    return _box_result(runs, stats, started)
