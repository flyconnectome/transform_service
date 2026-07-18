"""Run-length encoding of sparse voxel sets, and the on-disk fragment codec.

A run is ``(x, y, z, length)``: ``length`` voxels starting at ``(x, y, z)`` and
extending along +X. This is DVID's ``sparsevol`` layout, and it suits
segmentation because X is the fastest-varying axis -- a neurite's cross-section
collapses to a handful of runs per scanline.

Coordinates carry no intrinsic scale. Whatever space a fragment was encoded in
(here: voxels at the mip the index was built for) is the space it decodes back
into, so the index records the mip alongside the runs rather than in them.

The codec is shared with the ``pcg_sparse`` prototype, which validated the
round trip against dense reads of real FlyWire neurons.
"""

import gzip

import numpy as np

RLE_DTYPE = np.int64

# Fragments are stored as int32. Coordinates are mip-level voxels, which for a
# whole-brain EM volume stay well inside 2^31 even at mip 0.
PACK_DTYPE = np.int32

# What callers should hold large coordinate arrays in. A neuron runs to tens of
# millions of voxels and the encoder makes several passes over that array, so
# accumulating at half the width of numpy's default int64 halves the peak.
#
# Signed rather than unsigned, despite both being four bytes: encode_rle takes
# differences between neighbouring coordinates, and on an unsigned type a
# decreasing coordinate wraps to a huge positive value rather than going
# negative. A volume with a negative voxel offset would go wrong the same way.
COORD_DTYPE = np.int32


def encode_rle(coords, assume_unique=False):
    """Encode ``(N, 3)`` voxel coordinates as ``(M, 4)`` runs along +X.

    Set ``assume_unique`` to skip the deduplication pass when the caller
    guarantees distinct coordinates. Duplicates otherwise break run detection:
    a repeated x has a step of 0, which is neither a continuation nor a clean
    new run.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N, 3), got {}".format(coords.shape))
    if coords.shape[0] == 0:
        return np.zeros((0, 4), dtype=RLE_DTYPE)

    # Kept at the caller's width if it is already a signed integer type. This
    # array is the large one -- the runs it produces are several times smaller --
    # so widening it here would undo the caller's choice to hold it narrow.
    # Only the runs that come out are normalised to RLE_DTYPE.
    if not np.issubdtype(coords.dtype, np.signedinteger):
        coords = coords.astype(RLE_DTYPE, copy=False)

    # Sort by (z, y, x) so runs along X are contiguous within a scanline.
    order = np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))
    c = coords[order]

    if not assume_unique:
        keep = np.ones(len(c), dtype=bool)
        keep[1:] = np.any(c[1:] != c[:-1], axis=1)
        c = c[keep]

    step = np.diff(c, axis=0)
    # A run continues only along the same scanline with x advancing by one.
    breaks = (step[:, 2] != 0) | (step[:, 1] != 0) | (step[:, 0] != 1)
    cut = np.nonzero(breaks)[0] + 1

    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(c)]))

    runs = np.empty((len(starts), 4), dtype=RLE_DTYPE)
    runs[:, :3] = c[starts]
    runs[:, 3] = ends - starts
    return runs


def decode_rle(runs):
    """Expand ``(M, 4)`` runs back to ``(N, 3)`` coordinates, sorted by (z, y, x)."""
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 4:
        raise ValueError("runs must be (M, 4), got {}".format(runs.shape))
    if runs.shape[0] == 0:
        return np.zeros((0, 3), dtype=RLE_DTYPE)

    runs = runs.astype(RLE_DTYPE, copy=False)
    lengths = runs[:, 3]
    if np.any(lengths < 1):
        raise ValueError("run lengths must be >= 1")

    total = int(lengths.sum())
    out = np.empty((total, 3), dtype=RLE_DTYPE)

    out[:, 1] = np.repeat(runs[:, 1], lengths)
    out[:, 2] = np.repeat(runs[:, 2], lengths)

    # Offset of each voxel within its own run: 0, 1, 2, ...
    run_start_index = np.repeat(np.cumsum(lengths) - lengths, lengths)
    out[:, 0] = np.repeat(runs[:, 0], lengths) + (np.arange(total) - run_start_index)
    return out


def rle_voxel_count(runs):
    """Voxels represented by ``runs``, without expanding them."""
    runs = np.asarray(runs)
    if runs.shape[0] == 0:
        return 0
    return int(runs[:, 3].sum())


def _running_max_within_scanline(end, y, z):
    """Cumulative maximum of ``end``, restarting at every new (z, y) scanline.

    numpy has no segmented accumulate, so each scanline is lifted into its own
    band of values -- wide enough that no scanline can reach into the next --
    a plain cumulative maximum is taken, and the offset removed. Coordinates
    are int32-bounded and scanlines are bounded by the run count, so the
    lifted values stay far inside int64.
    """
    scanline = np.zeros(len(end), dtype=RLE_DTYPE)
    scanline[1:] = np.cumsum((z[1:] != z[:-1]) | (y[1:] != y[:-1]))

    band = int(end.max() - end.min()) + 1
    lifted = end + scanline * band
    return np.maximum.accumulate(lifted) - scanline * band


def merge_runs(runs):
    """Sort runs by (z, y, x) and coalesce the ones that touch.

    This is what turns a pile of per-supervoxel fragments into one canonical
    sparse volume. Two supervoxels meeting along X leave two runs that describe
    one unbroken stretch of voxels, and a chunk boundary does the same, so
    merging is what keeps a root's RLE as compact as if it had been encoded in
    one pass.

    Fragments are disjoint by construction -- a voxel carries exactly one
    supervoxel label -- so this assumes no two runs cover the same voxel.
    Overlapping input is still absorbed rather than duplicated, but the run
    count is only minimal for disjoint input.
    """
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 4:
        raise ValueError("runs must be (M, 4), got {}".format(runs.shape))
    if runs.shape[0] == 0:
        return np.zeros((0, 4), dtype=RLE_DTYPE)

    runs = runs.astype(RLE_DTYPE, copy=False)
    order = np.lexsort((runs[:, 0], runs[:, 1], runs[:, 2]))
    r = runs[order]

    x, y, z, length = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    end = x + length

    # How far the scanline is covered so far. This has to be the running
    # maximum, not simply the previous run's end: a run nested inside an
    # earlier one would otherwise pull the frontier backwards and let the run
    # after it start a second, overlapping block.
    frontier = _running_max_within_scanline(end, y, z)

    # A run joins the previous block only on the same scanline, and only if it
    # starts no later than where that block stopped (== abuts, < overlaps).
    breaks = (z[1:] != z[:-1]) | (y[1:] != y[:-1]) | (x[1:] > frontier[:-1])
    starts = np.concatenate(([0], np.nonzero(breaks)[0] + 1))

    out = np.empty((len(starts), 4), dtype=RLE_DTYPE)
    out[:, :3] = r[starts, :3]
    # reduceat takes the max end within each merged group, so a run wholly
    # contained in its predecessor cannot shorten the result.
    out[:, 3] = np.maximum.reduceat(end, starts) - x[starts]
    return out


def union_rle(fragments):
    """Merge many fragments' runs into one canonical ``(M, 4)`` array."""
    fragments = [np.asarray(f) for f in fragments]
    fragments = [f for f in fragments if f.shape[0] > 0]
    if not fragments:
        return np.zeros((0, 4), dtype=RLE_DTYPE)
    return merge_runs(np.concatenate(fragments, axis=0))


def pack_runs(runs, compress=True):
    """Serialize ``(M, 4)`` runs to the bytes stored in a fragment.

    Columns are transposed to run-major order and delta-coded before gzip.
    Runs arrive sorted by (z, y, x), so three of the four columns are nearly
    constant down the array and the fourth advances in small steps; deltas make
    that structure visible to gzip. The prototype measured this at ~2.3 bytes
    per run on real supervoxels, against 16 bytes raw.

    Compression is per fragment rather than per file because fragments are read
    by byte range: a whole-file codec would force reading the file to reach any
    one of them. That costs roughly 28% over compressing in bulk, and buys the
    ranged read the whole design rests on.
    """
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 4:
        raise ValueError("runs must be (M, 4), got {}".format(runs.shape))
    if runs.shape[0] == 0:
        return b""

    packed = runs.astype(PACK_DTYPE, copy=False)
    if not np.array_equal(packed.astype(RLE_DTYPE), runs.astype(RLE_DTYPE)):
        raise ValueError("runs do not fit in int32; coordinates out of range")

    columns = np.ascontiguousarray(packed.T)  # (4, M)
    deltas = columns.copy()
    deltas[:, 1:] = columns[:, 1:] - columns[:, :-1]

    buf = deltas.tobytes()
    # mtime=0 so identical runs always produce identical bytes, which keeps
    # rebuilt fragments comparable and makes the tests deterministic.
    return gzip.compress(buf, mtime=0) if compress else buf


def unpack_runs(blob, compressed=True):
    """Inverse of :func:`pack_runs`."""
    if not blob:
        return np.zeros((0, 4), dtype=RLE_DTYPE)

    buf = gzip.decompress(blob) if compressed else blob
    deltas = np.frombuffer(buf, dtype=PACK_DTYPE)
    if deltas.size % 4:
        raise ValueError("fragment is not a whole number of runs")

    columns = np.cumsum(deltas.reshape(4, -1), axis=1, dtype=RLE_DTYPE)
    return np.ascontiguousarray(columns.T)
