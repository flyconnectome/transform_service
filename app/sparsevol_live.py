"""Sparse voxels computed on the fly from a locally-stored segmentation.

The dense read happens here, per request, and that is the point rather than a
compromise. These watershed volumes sit on the same machine as the service, so
reading a few hundred chunks costs a local disk read; sending those same chunks
to the client would cost gigabytes over the wire. Doing the masking server-side
turns a multi-gigabyte download into a few hundred kilobytes of runs.

So this trades the client's bandwidth for the server's I/O, deliberately. There
is no index and no cache: every request re-reads and re-sparsifies. What makes
that affordable is locality, not cleverness.

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
from .chunks import ChunkLayout, chunk_boxes, l2_chunk_positions
from .rle import COORD_DTYPE, encode_rle, rle_voxel_count

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


def root_to_runs(dataset, scale, root_id):
    """Sparse voxels for a root ID, computed from the local segmentation.

    Two graph calls and no image data: ``stop_layer=2`` gives the chunks to
    read, and the full leaf set gives the supervoxels to keep. Everything after
    that is local.
    """
    info = get_live_info(dataset)
    started = time.time()
    stats = LiveStats()

    graph = get_graph(dataset)
    layout = get_layout(dataset)

    try:
        l2_ids = graph.get_leaves(int(root_id), graph.meta.bounds(0), 0, stop_layer=2)
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

    stats.n_l2_nodes = len(l2_ids)
    stats.n_supervoxels = len(supervoxels)

    positions = l2_chunk_positions(layout, l2_ids)
    boxes = chunk_boxes(graph.meta, positions, mip=scale)
    stats.n_chunks = len(boxes)
    check_budget(info, boxes, stats)

    coords = read_and_mask(dataset, scale, boxes, supervoxels, stats)
    return _to_runs(coords, stats, started)


def supervoxels_to_runs(dataset, scale, sv_ids):
    """Sparse voxels for explicit supervoxel IDs, with no call to the graph.

    The chunk position is arithmetic on the ID, and the labels to keep are the
    IDs themselves, so this path never contacts the graph server at all.
    """
    info = get_live_info(dataset)
    started = time.time()
    stats = LiveStats()

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
