"""Supervoxel -> RLE index: the serve path.

PyChunkedGraph stores the graph, not voxels. Asking it for a neuron's voxels
degrades to "read dense blocks, mask to one root, sparsify", which touches
100-1000x more voxels than it keeps -- for one FlyWire neuron, ~20 billion
voxels at mip 0 to end up with ~30 million.

This service does not do that. It reads a precomputed index keyed on
**supervoxel ID**, and a request becomes:

    get_leaves(root) -> group supervoxels by chunk -> ranged reads -> union

No dense voxels are read, and none cross the wire. The dense read still exists,
but it happens once, offline, when the index is built (see
:mod:`app.sparsevol_build`); it never happens per request.

Keying on supervoxels rather than roots is what makes the index static.
Supervoxel IDs are immutable -- proofreading re-agglomerates them into new roots
but never changes them -- so a fragment written today stays correct through
every future edit, exactly like the per-L2 mesh fragments this sits beside.

**Why the index is grouped by chunk.** A graphene label packs its chunk position
into its high bits, so sorting fragments by raw supervoxel ID groups them by
chunk for free. One FlyWire root's 27,839 supervoxels fall into 587 chunks, so
storing per chunk turns 27,839 point lookups into 587 ranged reads without
changing the key.
"""

import struct
import threading
import time
from collections import OrderedDict

import numpy as np
from fastapi import HTTPException

from . import config
from .chunks import ChunkLayout, chunk_key
from .rle import union_rle, unpack_runs, rle_voxel_count

# --- Fragment file format -------------------------------------------------
#
# Each chunk of the segmentation gets two objects in the store:
#
#   {mip}/{x}_{y}_{z}.idx   header + a table of (supervoxel, offset, nbytes)
#   {mip}/{x}_{y}_{z}.dat   the fragments those offsets point into
#
# The index is a separate object so a request can read every index it needs in
# one parallel round, work out the byte ranges, and then read only those ranges.
# Indices are small and shared across roots, so they cache well in process;
# packing them into the head of the data file instead would make every request
# either guess a prefix length or pay a second round trip.

INDEX_MAGIC = b"SVRLEIDX"
INDEX_VERSION = 1
INDEX_HEADER_SIZE = 24
INDEX_DTYPE = np.dtype(
    [("sv", "<u8"), ("offset", "<u8"), ("nbytes", "<u4"), ("n_runs", "<u4")]
)

FLAG_GZIP = 1

# Two fragments this close in the data file are fetched as one read. A short
# gap costs fewer bytes than a second request costs in latency.
COALESCE_GAP = 8192


_tqdm_lock_set = False


def open_cloudfiles(path):
    """A CloudFiles handle that keeps its progress machinery in-process.

    cloudfiles builds a tqdm bar for every batch it schedules, even a disabled
    one, and tqdm's default lock is a *multiprocessing* lock -- creating it
    spawns the resource-tracker subprocess. Inside a threaded server that is at
    best pointless, and on macOS spawning after threads exist aborts the
    process outright. Handing tqdm a plain threading lock avoids all of it.
    """
    from cloudfiles import CloudFiles
    from tqdm import tqdm

    global _tqdm_lock_set
    if not _tqdm_lock_set:
        tqdm.set_lock(threading.RLock())
        _tqdm_lock_set = True

    return CloudFiles(path, progress=False)


def encode_index(records, mip, compressed=True):
    """Serialize an index table to the bytes of a ``.idx`` object."""
    records = np.asarray(records, dtype=INDEX_DTYPE)
    if np.any(np.diff(records["sv"].astype(np.int64)) <= 0):
        raise ValueError("index records must be sorted by supervoxel and unique")
    header = INDEX_MAGIC + struct.pack(
        "<IIiI", INDEX_VERSION, len(records), int(mip), FLAG_GZIP if compressed else 0
    )
    return header + records.tobytes()


def decode_index(blob):
    """Inverse of :func:`encode_index`. Returns ``(records, mip, compressed)``."""
    if len(blob) < INDEX_HEADER_SIZE or blob[:8] != INDEX_MAGIC:
        raise ValueError("not a supervoxel RLE index")
    version, count, mip, flags = struct.unpack("<IIiI", blob[8:INDEX_HEADER_SIZE])
    if version != INDEX_VERSION:
        raise ValueError("unsupported index version {}".format(version))

    body = blob[INDEX_HEADER_SIZE:]
    expected = count * INDEX_DTYPE.itemsize
    if len(body) < expected:
        raise ValueError("index is truncated")
    records = np.frombuffer(body[:expected], dtype=INDEX_DTYPE)
    return records, mip, bool(flags & FLAG_GZIP)


class FetchStats:
    """What a request actually moved.

    Reported on every response because the whole design is a bandwidth claim,
    and a claim that is not measured tends to stop being true.
    """

    def __init__(self):
        self.n_supervoxels = 0
        self.n_chunks = 0
        self.n_fragments = 0
        self.n_missing = 0
        self.n_index_reads = 0
        self.n_index_cached = 0
        self.n_range_reads = 0
        self.bytes_read = 0
        self.n_runs = 0
        self.n_voxels = 0
        self.seconds = 0.0

    def as_dict(self):
        out = dict(self.__dict__)
        out["seconds"] = round(self.seconds, 4)
        return out


class FragmentStore:
    """Ranged reads against one dataset's fragment store.

    Backed by CloudFiles, so ``path`` may be a local directory, ``gs://``,
    ``s3://`` or plain HTTP. Index objects are cached in process; they are
    small, immutable, and reused by every root passing through the same chunk.
    """

    def __init__(self, path, cache_size=1024):
        self.path = path
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._cf = None

    def _files(self):
        if self._cf is None:
            self._cf = open_cloudfiles(self.path)
        return self._cf

    def _cache_get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def _cache_put(self, key, value):
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def read_indices(self, mip, keys, stats):
        """Fetch the index for each chunk key, in one parallel round."""
        found = {}
        missing = []
        for key in keys:
            cached = self._cache_get((mip, key))
            if cached is None:
                missing.append(key)
            else:
                found[key] = cached
                stats.n_index_cached += 1

        if missing:
            paths = ["{}/{}.idx".format(mip, key) for key in missing]
            results = self._files().get(paths, raise_errors=False)
            for result in results:
                key = result["path"].split("/")[-1][: -len(".idx")]
                if result.get("error") is not None:
                    # A store that is failing is not a store that is empty.
                    # Treating the two alike would answer a broken read with a
                    # partial neuron and HTTP 200, which a client would cache.
                    raise HTTPException(
                        status_code=502,
                        detail="Could not read index for chunk {}: {}".format(
                            key, result["error"]
                        ),
                    )
                blob = result.get("content")
                if not blob:
                    # A chunk with no index is a chunk the build never wrote.
                    # Its supervoxels simply resolve to nothing.
                    continue
                stats.n_index_reads += 1
                stats.bytes_read += len(blob)
                entry = decode_index(blob)
                self._cache_put((mip, key), entry)
                found[key] = entry

        return found

    def read_fragments(self, mip, wanted, stats):
        """Look up fragments for ``{chunk_key: supervoxel ids}``.

        Returns a list of ``(M, 4)`` run arrays. Fragments not present in the
        index are counted as missing rather than raising: an index built over a
        subset of the volume should answer for the part it covers.
        """
        indices = self.read_indices(mip, list(wanted.keys()), stats)

        # Plan the reads first, then issue them all at once, so one request
        # costs one round trip against the store rather than one per chunk.
        reads = []
        for key, sv_ids in wanted.items():
            entry = indices.get(key)
            if entry is None:
                stats.n_missing += len(sv_ids)
                continue

            records, _, compressed = entry
            selected = _select_records(records, sv_ids)
            stats.n_missing += len(sv_ids) - len(selected)
            if len(selected) == 0:
                continue

            for start, end, group in _coalesce(selected):
                reads.append(
                    {
                        "path": "{}/{}.dat".format(mip, key),
                        "start": int(start),
                        "end": int(end),
                        "group": group,
                        "compressed": compressed,
                    }
                )

        if not reads:
            return []

        stats.n_range_reads = len(reads)
        results = self._files().get(
            [{"path": r["path"], "start": r["start"], "end": r["end"]} for r in reads],
            raise_errors=False,
        )

        # Return order is not guaranteed, so match responses back to plans by
        # (path, byte range) rather than by position.
        by_range = {}
        for result in results:
            if result.get("error") is not None:
                raise HTTPException(
                    status_code=502,
                    detail="Could not read fragments from {}: {}".format(
                        result["path"], result["error"]
                    ),
                )
            if not result.get("content"):
                continue
            byte_range = result.get("byte_range") or (None, None)
            by_range[(result["path"], int(byte_range[0]))] = result["content"]

        fragments = []
        for read in reads:
            blob = by_range.get((read["path"], read["start"]))
            if blob is None:
                # The index said these fragments are here. If the data file
                # cannot produce them the index and data disagree, which is
                # corruption rather than an unindexed supervoxel.
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Index for {} lists {} fragments at bytes {}-{} that the "
                        "data file does not have.".format(
                            read["path"], len(read["group"]), read["start"], read["end"]
                        )
                    ),
                )
            stats.bytes_read += len(blob)
            for record in read["group"]:
                offset = int(record["offset"]) - read["start"]
                piece = blob[offset : offset + int(record["nbytes"])]
                runs = _decode_fragment(piece, record, read)
                fragments.append(runs)
                stats.n_fragments += 1

        return fragments


def _decode_fragment(piece, record, read):
    """Decode one fragment, checking it against what the index promised.

    A short read slices silently in Python, so the byte count and the run count
    are both verified: the alternative is answering with a quietly truncated
    neuron, which is worse than answering with an error.
    """
    if len(piece) != int(record["nbytes"]):
        raise HTTPException(
            status_code=502,
            detail="Fragment {} in {} is {} bytes, expected {}.".format(
                int(record["sv"]), read["path"], len(piece), int(record["nbytes"])
            ),
        )

    try:
        runs = unpack_runs(piece, compressed=read["compressed"])
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Fragment {} in {} could not be decoded: {}".format(
                int(record["sv"]), read["path"], exc
            ),
        )

    if len(runs) != int(record["n_runs"]):
        raise HTTPException(
            status_code=502,
            detail="Fragment {} in {} holds {} runs, index says {}.".format(
                int(record["sv"]), read["path"], len(runs), int(record["n_runs"])
            ),
        )
    return runs


def _select_records(records, sv_ids):
    """The index rows for ``sv_ids``, dropping ids the index does not carry."""
    keys = records["sv"]
    if len(keys) == 0:
        return records[:0]
    positions = np.searchsorted(keys, sv_ids)
    positions = np.clip(positions, 0, len(keys) - 1)
    hit = keys[positions] == sv_ids
    return records[positions[hit]]


def _coalesce(records, gap=COALESCE_GAP):
    """Group index rows into byte ranges, merging rows separated by < ``gap``.

    Fragments for one root inside one chunk are scattered through the data
    file, but supervoxels of the same neuron cluster, so the rows are often
    near-adjacent. Merging pulls a few unwanted bytes to avoid a round trip.
    """
    order = np.argsort(records["offset"], kind="stable")
    records = records[order]

    groups = []
    start = int(records[0]["offset"])
    end = start + int(records[0]["nbytes"])
    current = [records[0]]

    for record in records[1:]:
        offset = int(record["offset"])
        if offset - end <= gap:
            current.append(record)
            end = max(end, offset + int(record["nbytes"]))
        else:
            groups.append((start, end, current))
            start = offset
            end = offset + int(record["nbytes"])
            current = [record]
    groups.append((start, end, current))
    return groups


# --- Dataset handles ------------------------------------------------------

_stores = {}
_volumes = {}
_layouts = {}
_handle_lock = threading.Lock()


def get_sparsevol_info(dataset):
    if dataset not in config.SPARSEVOL_DATASOURCES:
        raise HTTPException(
            status_code=400, detail="Sparsevol dataset {} not found".format(dataset)
        )
    return config.SPARSEVOL_DATASOURCES[dataset]


def get_store(dataset):
    info = get_sparsevol_info(dataset)
    with _handle_lock:
        if dataset not in _stores:
            _stores[dataset] = FragmentStore(info["index"])
        return _stores[dataset]


def get_volume(dataset):
    """The graphene volume, opened lazily.

    Only needed to resolve a root to its supervoxels and to decode chunk
    positions -- never to read image data. Datasets that only serve supervoxel
    lookups do not need it configured at all.
    """
    info = get_sparsevol_info(dataset)
    if not info.get("graphene"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset {} has no graphene source configured, so root IDs "
                "cannot be resolved. Query supervoxels directly.".format(dataset)
            ),
        )

    with _handle_lock:
        if dataset not in _volumes:
            from cloudvolume import CloudVolume

            try:
                _volumes[dataset] = CloudVolume(
                    info["graphene"], use_https=True, progress=False, fill_missing=True
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Could not open the chunkedgraph for {} at {}: {}. "
                        "Check the graphene source and CAVE credentials.".format(
                            dataset, info["graphene"], exc
                        )
                    ),
                )
        return _volumes[dataset]


def check_scale(info, mip):
    if mip not in info["scales"]:
        raise HTTPException(
            status_code=400,
            detail="Scale {} not indexed for this dataset. Available: {}".format(
                mip, info["scales"]
            ),
        )


def get_layout(dataset):
    """The chunk bit layout, from config if given, else from graphene metadata."""
    info = get_sparsevol_info(dataset)
    layout = info.get("chunk_layout")
    if layout is not None:
        return ChunkLayout(layout["spatial_bits"], layout.get("layer_bits", 8))

    cached = _layouts.get(dataset)
    if cached is not None:
        return cached

    # Opened outside the lock: get_volume takes the same lock, and it is not
    # reentrant. Two threads racing here just build the same layout twice.
    meta = get_volume(dataset).meta
    with _handle_lock:
        _layouts[dataset] = ChunkLayout.from_metadata(meta)
        return _layouts[dataset]


def group_by_chunk(dataset, sv_ids):
    """Group supervoxel IDs by the chunk their high bits encode."""
    sv_ids = np.asarray(sv_ids, dtype=np.uint64)
    positions = get_layout(dataset).decode(sv_ids)

    # Distinct chunks number in the hundreds against tens of thousands of
    # supervoxels, so collapse to unique rows and map each id by index.
    unique, inverse = np.unique(positions, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)  # shape of this varies across numpy versions

    grouped = {}
    for i, position in enumerate(unique):
        # Fragments are looked up by binary search, which needs sorted input.
        grouped[chunk_key(position)] = np.unique(sv_ids[inverse == i])
    return grouped


# --- Query entry points ---------------------------------------------------


def supervoxels_to_runs(dataset, mip, sv_ids):
    """Union the RLE fragments of an explicit supervoxel list."""
    info = get_sparsevol_info(dataset)
    check_scale(info, mip)

    started = time.time()
    stats = FetchStats()

    sv_ids = np.unique(np.asarray(sv_ids, dtype=np.uint64))
    sv_ids = sv_ids[sv_ids != 0]
    stats.n_supervoxels = len(sv_ids)

    limit = info.get("max_supervoxels", config.MaxSupervoxels)
    if limit and len(sv_ids) > limit:
        raise HTTPException(
            status_code=400,
            detail="Request covers {} supervoxels, over the {} limit.".format(
                len(sv_ids), limit
            ),
        )

    if len(sv_ids) == 0:
        stats.seconds = time.time() - started
        return np.zeros((0, 4), dtype=np.int64), stats

    grouped = group_by_chunk(dataset, sv_ids)
    stats.n_chunks = len(grouped)

    store = get_store(dataset)
    runs = union_rle(store.read_fragments(mip, grouped, stats))

    stats.n_runs = len(runs)
    stats.n_voxels = rle_voxel_count(runs)
    stats.seconds = time.time() - started
    return runs, stats


def root_to_runs(dataset, mip, root_id):
    """Sparse voxels for a root ID: manifest lookup, then ranged reads.

    The manifest (``get_leaves``) is the one irreducibly live piece -- it has to
    reflect the current agglomeration. Everything after it is static.
    """
    info = get_sparsevol_info(dataset)
    check_scale(info, mip)

    cv = get_volume(dataset)
    try:
        sv_ids = cv.get_leaves(int(root_id), cv.meta.bounds(0), 0)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not resolve root {}: {}".format(root_id, exc),
        )

    if len(sv_ids) == 0:
        raise HTTPException(
            status_code=404,
            detail="Root {} has no supervoxels. Is it a current root ID?".format(root_id),
        )

    return supervoxels_to_runs(dataset, mip, sv_ids)
