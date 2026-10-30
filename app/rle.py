"""Run-length encoding of sparse voxel sets, and the on-disk fragment codec.

A run is ``(x, y, z, length)``: ``length`` voxels starting at ``(x, y, z)`` and
extending along +X. This is DVID's ``sparsevol`` layout, and it suits
segmentation because X is the fastest-varying axis -- a neurite's cross-section
collapses to a handful of runs per scanline.

The box service answers about several objects at once, so it uses a labelled
variant, ``(x, y, z, length, label)``. The two layouts share a column order up
to ``length``, so anything reading only those columns -- :func:`rle_voxel_count`
is the one here -- works on both. Everything else is layout-specific and says
so in its own shape check. They also differ in an assumption: unlabelled runs
are disjoint, since a voxel carries exactly one supervoxel, while labelled runs
are not, since a voxel may belong to several objects at once.

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


def _encode_runs(coords, labels, assume_unique, assume_sorted):
    """Shared body of :func:`encode_rle` and :func:`encode_rle_labelled`.

    ``labels is None`` is the unlabelled encoder. The dedup pass, the rule for
    when a run continues and the narrow-dtype policy are identical either way,
    and are the parts where a second copy would quietly drift.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N, 3), got {}".format(coords.shape))
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != coords.shape[0]:
            raise ValueError(
                "labels must be (N,) matching coords, got {} for {}".format(
                    labels.shape, coords.shape
                )
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(
                "labels must be an integer type, got {}".format(labels.dtype)
            )
        # The label column is RLE_DTYPE like the rest of a run, so a uint64
        # label past 2^63 would be stored -- and sorted -- as a negative
        # number. Labels here are small object indices, not graphene IDs, so
        # this is a caller mistake worth naming rather than a case to support:
        # supervoxel IDs need the bit-pattern round trip in app.l2cache.
        # Only uint64 can exceed the range, so nothing else pays for the check.
        if labels.dtype == np.uint64 and labels.size and int(labels.max()) >= 2**63:
            raise ValueError("labels must fit in a signed 64-bit int")

    width = 4 if labels is None else 5
    if coords.shape[0] == 0:
        return np.zeros((0, width), dtype=RLE_DTYPE)

    # Kept at the caller's width if it is already a signed integer type. This
    # array is the large one -- the runs it produces are several times smaller --
    # so widening it here would undo the caller's choice to hold it narrow.
    # Only the runs that come out are normalised to RLE_DTYPE.
    if not np.issubdtype(coords.dtype, np.signedinteger):
        coords = coords.astype(RLE_DTYPE, copy=False)

    # Sort by (z, y, x) so runs along X are contiguous within a scanline, with
    # the label ahead of all three when there is one: a run must never be
    # continued across a label boundary.
    if assume_sorted:
        c, lab = coords, labels
    else:
        keys = (coords[:, 0], coords[:, 1], coords[:, 2])
        order = np.lexsort(keys if labels is None else keys + (labels,))
        c = coords[order]
        lab = None if labels is None else labels[order]
        del order  # an N-long int64 array, dead before the filtered copies below

    if not assume_unique:
        # Column by column rather than np.any(c[1:] != c[:-1], axis=1), which
        # builds a full (N-1, 3) bool before reducing it away.
        duplicate = c[1:, 0] == c[:-1, 0]
        duplicate &= c[1:, 1] == c[:-1, 1]
        duplicate &= c[1:, 2] == c[:-1, 2]
        if lab is not None:
            # Compared rather than differenced, so labels stay at their own
            # width: an unsigned column would wrap on a decreasing step.
            duplicate &= lab[1:] == lab[:-1]

        keep = np.ones(len(c), dtype=bool)
        keep[1:] = ~duplicate
        c = c[keep]
        if lab is not None:
            lab = lab[keep]

    # A run continues only along the same scanline, with x advancing by one,
    # and -- when there are labels -- within a single label.
    breaks = c[1:, 2] != c[:-1, 2]
    breaks |= c[1:, 1] != c[:-1, 1]
    breaks |= c[1:, 0] != c[:-1, 0] + 1
    if lab is not None:
        breaks |= lab[1:] != lab[:-1]

    cut = np.nonzero(breaks)[0] + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(c)]))

    runs = np.empty((len(starts), width), dtype=RLE_DTYPE)
    runs[:, :3] = c[starts]
    runs[:, 3] = ends - starts
    if lab is not None:
        runs[:, 4] = lab[starts]
    return runs


def encode_rle(coords, assume_unique=False, assume_sorted=False):
    """Encode ``(N, 3)`` voxel coordinates as ``(M, 4)`` runs along +X.

    Set ``assume_unique`` to skip the deduplication pass when the caller
    guarantees distinct coordinates. Duplicates otherwise break run detection:
    a repeated x has a step of 0, which is neither a continuation nor a clean
    new run.

    Set ``assume_sorted`` when the caller already holds the coordinates in
    (z, y, x) order. The sort is the bulk of this function's cost, so a caller
    that gets the order for free elsewhere -- by sorting packed integer keys,
    say -- should not pay for it twice. Passing it wrongly does not raise; it
    silently produces more runs than necessary, each of them still correct.
    """
    return _encode_runs(coords, None, assume_unique, assume_sorted)


def encode_rle_labelled(coords, labels, assume_unique=False, assume_sorted=False):
    """Encode ``(N, 3)`` coordinates carrying ``(N,)`` labels as ``(M, 5)`` runs.

    A run is ``(x, y, z, length, label)``. Runs never span two labels, so this
    is :func:`encode_rle` applied independently within each label, in one pass.
    Only repeats of the same (label, voxel) pair count as duplicates.

    Output is sorted by (label, z, y, x), which keeps each label's runs
    contiguous. There is deliberately no labelled counterpart to
    :func:`merge_runs`: a labelled result is encoded in one call, from all of
    its coordinates at once, so nothing needs to merge two of them afterwards.

    ``assume_unique`` and ``assume_sorted`` are as for :func:`encode_rle`, with
    "sorted" meaning (label, z, y, x) here. The box service gets both for free:
    it accumulates its voxels as packed integer keys with the label in the high
    bits, so one ``np.unique`` over those keys leaves them deduplicated and in
    exactly this order.
    """
    return _encode_runs(coords, labels, assume_unique, assume_sorted)


def _expand_runs(runs, width, dtype):
    """Expand ``(M, 4)`` runs into an ``(N, width)`` array of one row per voxel.

    Fills the first three columns and leaves any beyond them to the caller, so
    the labelled decoder can build its wider array directly. This is the
    largest array the module produces -- one row per voxel, where everything
    else holds one row per run -- so building it twice would cost more than
    everything else here put together, and ``dtype`` is here for the same
    reason: a caller that is going to narrow the result should say so now
    rather than pay for an int64 copy first.
    """
    # Checked at the width it arrived in, before anything is narrowed, so a
    # length that would wrap on the way down cannot slip past.
    if np.any(runs[:, 3] < 1):
        raise ValueError("run lengths must be >= 1")
    total = int(runs[:, 3].sum())

    # Narrowed here, not just on the way out: otherwise every np.repeat below
    # builds a full-length int64 temporary to be cast into a narrow column, and
    # those temporaries are the thing `dtype` exists to avoid.
    runs = runs.astype(dtype, copy=False)
    lengths = runs[:, 3]
    out = np.empty((total, width), dtype=dtype)

    out[:, 1] = np.repeat(runs[:, 1], lengths)
    out[:, 2] = np.repeat(runs[:, 2], lengths)

    # x advances by one within a run, so each voxel is its run's start plus its
    # offset into the flattened output. Subtracting the run's start index
    # before the repeat keeps that arithmetic one-per-run rather than
    # one-per-voxel, which is where this function's temporaries come from.
    #
    # Note the intermediate `x - start` can be hugely negative in a narrow
    # dtype -- `start` grows to `total` while `x` is a coordinate -- and is
    # brought back by the `+= arange` below. That is exact rather than lucky:
    # both steps are modular in the same width, so the wraparound cancels. It
    # stops being exact if anything here clamps, reorders these two lines, or
    # uses an unsigned dtype.
    starts = np.cumsum(lengths, dtype=dtype) - lengths
    out[:, 0] = np.repeat(runs[:, 0] - starts, lengths)
    out[:, 0] += np.arange(total, dtype=out.dtype)
    return out


def decode_rle(runs, dtype=RLE_DTYPE):
    """Expand ``(M, 4)`` runs back to ``(N, 3)`` coordinates, sorted by (z, y, x)."""
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 4:
        raise ValueError("runs must be (M, 4), got {}".format(runs.shape))
    if runs.shape[0] == 0:
        return np.zeros((0, 3), dtype=dtype)

    return _expand_runs(runs, 3, dtype)


def is_labelled(runs):
    """Whether ``runs`` carry a label column.

    The two layouts are told apart by width, so the rule for doing that lives
    here next to the encoders rather than wherever a consumer happens to need
    it. Both encoders return the right width even for an empty result, so this
    works on a result with no runs in it.
    """
    runs = np.asarray(runs)
    if runs.ndim != 2:
        raise ValueError("runs must be (M, 4) or (M, 5), got {}".format(runs.shape))
    return runs.shape[1] == 5


def decode_rle_labelled(runs, dtype=RLE_DTYPE):
    """Expand ``(M, 5)`` labelled runs to ``(N, 4)`` rows of ``x, y, z, label``.

    One row per (label, voxel) pair, so a voxel carried by three labels appears
    three times. Sorted by (label, z, y, x), matching the encoder.
    """
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 5:
        raise ValueError("runs must be (M, 5), got {}".format(runs.shape))
    if runs.shape[0] == 0:
        return np.zeros((0, 4), dtype=dtype)

    out = _expand_runs(runs[:, :4], 4, dtype)
    # Narrowed before the repeat for the same reason as the columns above.
    out[:, 3] = np.repeat(runs[:, 4].astype(dtype, copy=False), runs[:, 3])
    return out


def rle_voxel_count(runs):
    """Voxels represented by ``runs``, without expanding them.

    Works on labelled ``(M, 5)`` runs too -- ``length`` is the same column --
    where it counts (label, voxel) pairs rather than distinct voxels, which is
    the number of rows a client gets back.
    """
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
