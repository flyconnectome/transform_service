"""Chunk geometry for graphene (PyChunkedGraph) segmentations.

A graphene label packs its layer and chunk position into its high bits, so the
set of chunks a neuron occupies is derivable from ``get_leaves(root,
stop_layer=2)`` alone, with no volume access at all. That is the fact both
sparsevol services are built on: it tells you *where* to read before you read
anything.

The chunk grid is defined in voxels at the graph's ``watershed_mip`` and, when
``chunks_start_at_voxel_offset`` is set, is anchored at the volume's voxel
offset rather than at the origin. Both conventions are read from metadata here
rather than assumed.
"""

import numpy as np

__all__ = ["ChunkLayout", "chunk_key", "chunk_bbox", "chunk_boxes", "l2_chunk_positions"]


class ChunkLayout:
    """How a graphene label packs its chunk position into its high bits.

    ``layer | x | y | z | segid``, with ``spatial_bits`` per axis. Decoding is
    pure arithmetic, so finding out which chunk a supervoxel lives in never
    requires a call to the graph server.
    """

    def __init__(self, spatial_bits, layer_bits=8):
        self.spatial_bits = int(spatial_bits)
        self.layer_bits = int(layer_bits)
        self.segid_bits = 64 - self.layer_bits - 3 * self.spatial_bits
        if self.segid_bits <= 0:
            raise ValueError("chunk layout leaves no bits for the segment id")

    @classmethod
    def from_metadata(cls, meta, level=1):
        return cls(meta.spatial_bit_count(level), meta.n_bits_for_layer_id)

    def decode(self, labels):
        """Chunk positions of ``labels`` as an ``(N, 3)`` array."""
        labels = np.asarray(labels, dtype=np.uint64)
        # Drop the layer id, then shift each axis down to the bottom.
        value = labels & np.uint64((1 << (64 - self.layer_bits)) - 1)
        mask = np.uint64((1 << self.spatial_bits) - 1)
        shifts = [
            np.uint64(self.segid_bits + 2 * self.spatial_bits),
            np.uint64(self.segid_bits + self.spatial_bits),
            np.uint64(self.segid_bits),
        ]
        return np.stack([(value >> s) & mask for s in shifts], axis=-1).astype(np.int64)


def chunk_key(position):
    return "{}_{}_{}".format(int(position[0]), int(position[1]), int(position[2]))


def chunk_bbox(meta, position, mip=0, clip=True):
    """Voxel bounding box of one graph chunk, in ``mip``-level coordinates.

    The inverse of CloudVolume's ``point_to_chunk_position``. Built at
    ``watershed_mip``, where the chunk grid is defined, and then converted, so
    an anisotropic pyramid -- fly datasets typically downsample XY only -- is
    handled by the metadata rather than by assuming an isotropic ratio.
    """
    from cloudvolume.lib import Bbox

    position = np.asarray(position, dtype=np.int64)
    size = np.asarray(meta.graph_chunk_size, dtype=np.int64)

    origin = np.zeros(3, dtype=np.int64)
    if meta.chunks_start_at_voxel_offset:
        origin = np.asarray(meta.voxel_offset(meta.watershed_mip), dtype=np.int64)

    minpt = position * size + origin
    box = Bbox(minpt, minpt + size)
    if mip != meta.watershed_mip:
        box = meta.bbox_to_mip(box, mip=meta.watershed_mip, to_mip=mip)
    if clip:
        box = Bbox.clamp(box, meta.bounds(mip))
    return box.astype(np.int64)


def chunk_boxes(meta, positions, mip=0, clip=True):
    """``chunk_bbox`` over many positions, dropping any that clip to nothing.

    Chunks on the volume's edge can fall partly or wholly outside the bounds,
    and an empty box is not something to hand to a reader.
    """
    boxes = []
    for position in np.atleast_2d(positions):
        box = chunk_bbox(meta, position, mip=mip, clip=clip)
        if not box.subvoxel():
            boxes.append(box)
    return boxes


def l2_chunk_positions(layout, l2_ids):
    """Distinct chunk positions of a set of layer-2 node IDs.

    Several L2 nodes routinely share a chunk -- a neuron passing through a
    chunk as two disconnected branches gets one L2 node per component -- so
    deduplicating here removes reads that would return the same data twice.
    """
    l2_ids = np.asarray(l2_ids, dtype=np.uint64).ravel()
    if l2_ids.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(layout.decode(l2_ids), axis=0)
