"""Building the supervoxel -> RLE index.

This is the only place dense voxels are ever read, and it is deliberately not
part of the service. It runs offline, once per segmentation, and everything the
service does afterwards is a manifest lookup plus ranged reads.

Scope: this is a bootstrap builder, meant to populate an index over one neuron
or one region so the serve path can be exercised end to end. A whole-dataset
build belongs in an Igneous-style batch job running in-region next to the
bucket, where the dense reads never leave the datacentre. Two things that job
should do differently:

* Scan in **storage-block** order, not chunk order. Object storage serves whole
  blocks, and at coarse mips many graph chunks share one block, so chunk-order
  scanning re-downloads the same block repeatedly.
* Emit RLE only where runs are long. Measured against delta-coded coordinates,
  RLE wins at mip 0-3 and loses by ~1.8x at mip 4 and above, where a neurite is
  one or two voxels across and runs average barely over 1.

**Why one chunk is a complete unit of work.** A layer-1 supervoxel is a
watershed component *within* a graph chunk -- that is exactly why the chunk
position fits in the label's high bits. A supervoxel therefore never spans two
chunks, so a fragment built from one chunk's voxels is complete, and chunks can
be built independently and in any order.
"""

import argparse

import numpy as np

from .chunks import chunk_bbox, chunk_key
from .rle import encode_rle, pack_runs
from .sparsevol import (
    INDEX_DTYPE,
    encode_index,
    get_sparsevol_info,
    open_cloudfiles,
)


def fragments_for_chunk(cv, position, mip=0):
    """Read one chunk densely and return ``{supervoxel: runs}``.

    Coordinates are absolute mip-level voxels, so a fragment decodes without
    needing to know which chunk it came from.
    """
    meta = cv.meta
    box = chunk_bbox(meta, position, mip=mip)
    if box.subvoxel():
        return {}

    array = np.asarray(cv.download(box, mip=mip, agglomerate=False))[..., 0]
    local = np.argwhere(array != 0)
    if local.shape[0] == 0:
        return {}

    labels = array[local[:, 0], local[:, 1], local[:, 2]]
    coords = local + np.asarray(box.minpt, dtype=np.int64)

    # At coarse mips a chunk box rounds outward onto the mip grid and so
    # overlaps its neighbours. Voxels belonging to a neighbouring chunk's
    # supervoxels come along with the read; writing them here would produce a
    # second, partial fragment for a supervoxel that another chunk owns.
    owner = np.asarray(position, dtype=np.int64)
    owned = {
        int(label)
        for label in np.unique(labels)
        if np.array_equal(meta.decode_chunk_position(int(label)), owner)
    }

    fragments = {}
    order = np.argsort(labels, kind="stable")
    labels, coords = labels[order], coords[order]
    boundaries = np.concatenate(([0], np.nonzero(np.diff(labels))[0] + 1, [len(labels)]))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        label = int(labels[start])
        if label not in owned:
            continue
        fragments[label] = encode_rle(coords[start:end], assume_unique=True)
    return fragments


def write_chunk(store_path, position, mip, fragments, compress=True):
    """Write one chunk's ``.idx``/``.dat`` pair.

    The data object is stored uncompressed: fragments carry their own gzip so
    that a byte range addresses a fragment. Compressing the object as a whole
    would force a full read to reach any single fragment.
    """
    key = chunk_key(position)
    if not fragments:
        return 0

    records = np.zeros(len(fragments), dtype=INDEX_DTYPE)
    blobs = []
    offset = 0
    for i, sv in enumerate(sorted(fragments)):
        blob = pack_runs(fragments[sv], compress=compress)
        records[i] = (sv, offset, len(blob), len(fragments[sv]))
        blobs.append(blob)
        offset += len(blob)

    files = open_cloudfiles(store_path)
    files.puts(
        [
            {
                "path": "{}/{}.idx".format(mip, key),
                "content": encode_index(records, mip, compressed=compress),
                "compress": None,
            },
            {
                "path": "{}/{}.dat".format(mip, key),
                "content": b"".join(blobs),
                "compress": None,
            },
        ],
        raw=True,
    )
    return offset


def build_chunks(dataset, positions, mip=0, compress=True, progress=True):
    """Build the index for an explicit list of chunk positions."""
    from cloudvolume import CloudVolume

    info = get_sparsevol_info(dataset)
    cv = CloudVolume(info["graphene"], use_https=True, progress=False, fill_missing=True)

    written = 0
    for i, position in enumerate(positions):
        fragments = fragments_for_chunk(cv, position, mip=mip)
        written += write_chunk(info["index"], position, mip, fragments, compress=compress)
        if progress:
            print(
                "  [{}/{}] chunk {} -> {} fragments".format(
                    i + 1, len(positions), chunk_key(position), len(fragments)
                ),
                flush=True,
            )
    return written


def build_for_root(dataset, root_id, mip=0, compress=True, progress=True):
    """Bootstrap the index over the chunks one root occupies.

    Enough to exercise the serve path against a real neuron without building
    the whole dataset. The chunk set comes from the graph alone -- ``stop_layer=2``
    IDs carry their chunk position -- so no volume is read to plan the build.
    """
    from cloudvolume import CloudVolume

    info = get_sparsevol_info(dataset)
    cv = CloudVolume(info["graphene"], use_https=True, progress=False, fill_missing=True)

    l2_ids = cv.get_leaves(int(root_id), cv.meta.bounds(0), 0, stop_layer=2)
    positions = np.unique(
        np.array([cv.meta.decode_chunk_position(int(i)) for i in l2_ids], dtype=np.int64),
        axis=0,
    )
    if progress:
        print(
            "root {}: {} L2 nodes in {} chunks at mip {}".format(
                root_id, len(l2_ids), len(positions), mip
            )
        )
    return build_chunks(dataset, positions, mip=mip, compress=compress, progress=progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", help="key in config.SPARSEVOL_DATASOURCES")
    parser.add_argument("root_id", type=int, help="root ID whose chunks to index")
    parser.add_argument("--mip", type=int, default=0)
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()

    written = build_for_root(
        args.dataset, args.root_id, mip=args.mip, compress=not args.no_compress
    )
    print("wrote {:,} bytes of fragments".format(written))


if __name__ == "__main__":
    main()
