"""Helpers shared by the sparsevol test modules.

Both the index tests and the live tests check results against voxel sets they
build by hand, so they need the same two things: a way to spell out a solid
box of voxels, and a way to put coordinates into the order the decoders emit.
Keeping one copy here means the two modules cannot disagree about what that
order is -- which they briefly did, when only one of them knew about labels.
"""

import numpy as np


def voxel_block(origin, shape):
    """Every voxel of a solid box, as (N, 3) coordinates."""
    ranges = [np.arange(o, o + s) for o, s in zip(origin, shape)]
    grid = np.meshgrid(*ranges, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=-1)


def runs_from(response):
    """Decode a binary sparsevol response, taking its width from the header.

    Deliberately not hardcoded per endpoint: `X-Sparsevol-Columns` exists so a
    client never has to know which URL and format produce which row width, and
    a test that knows the width independently would never catch the header
    being wrong.
    """
    width = int(response.headers["X-Sparsevol-Columns"])
    return (
        np.frombuffer(response.content, dtype="<i4")
        .reshape(-1, width)
        .astype(np.int64)
    )


def sort_rows(rows):
    """Deduplicate and sort into the order the decoders emit.

    ``np.lexsort`` takes its last key as the primary one, so passing the
    columns in order sorts (N, 3) coordinates by (z, y, x) and (N, 4) labelled
    rows by (label, z, y, x) -- which is exactly what each decoder returns.
    """
    rows = np.unique(np.asarray(rows), axis=0)
    return rows[np.lexsort(rows.T)]
