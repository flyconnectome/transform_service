"""Repack an unsharded precomputed scale into gzip-compressed shards, byte for byte.

Why this exists: a segmentation downloaded with ``gs cp`` / ``gsutil cp`` comes
down unzipped, because the bucket stores each chunk with ``Content-Encoding:
gzip`` and the download strips it. That roughly triples the size on disk, and
tensorstore cannot read gzipped chunk files in the unsharded layout -- gzip
them in place and it reports them corrupted; add a ``.gz`` suffix and it treats
every chunk as missing and returns zeros. The sharded layout is the one place
the precomputed format allows gzip, so that is where the chunks go.

Chunk files are never decoded: each file's compressed_segmentation bytes are
gzipped and placed in the shard under its compressed Morton code, which is
exactly what tensorstore and neuroglancer look up. The chunk size is
unchanged, so nothing that reads the volume needs to change either. The
source is only read.

    uv run python tools/to_sharded.py SRC_DIR DST_DIR SCALE_KEY [--workers N] [--dry-run]

Resumable: shard files are written to a .tmp name and renamed, and existing
ones are skipped on a rerun. The info file is written last, atomically.

To fill a scale straight from the bucket instead, without downloading it
unzipped first, see :mod:`fetch_sharded`.
"""

import argparse
import gzip
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from cloudvolume.datasource.precomputed.image.common import compressed_morton_code
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

NAME = re.compile(r"^(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$")

SHARDING = {
    "@type": "neuroglancer_uint64_sharded_v1",
    "hash": "identity",
    # 2^(5+3) = 256 chunks per shard (8x8x4 chunks), 32 per minishard. A dense
    # aedes shard is ~150 MB, so a worker holds a few hundred MB at most.
    "preshift_bits": 5,
    "minishard_bits": 3,
    "shard_bits": None,  # filled in from the grid size
    "minishard_index_encoding": "gzip",
    "data_encoding": "gzip",
}

# Level 6 measured the same size as 9 on real aedes chunks, in less than half
# the time.
GZIP_LEVEL = 6


def scale_grid(scale):
    """``(chunk, offset, size, grid)`` for one scale of a precomputed info."""
    chunk = np.array(scale["chunk_sizes"][0])
    offset = np.array(scale.get("voxel_offset", [0, 0, 0]))
    size = np.array(scale["size"])
    grid = np.ceil(size / chunk).astype(np.int64)
    return chunk, offset, size, grid


def sharding_for(grid):
    """The sharding spec for a scale with this chunk grid."""
    sharding = dict(SHARDING)
    total_bits = sum(int(np.ceil(np.log2(g))) for g in grid)
    sharding["shard_bits"] = max(
        0, total_bits - sharding["preshift_bits"] - sharding["minishard_bits"]
    )
    return sharding


def find_scale(info, key):
    scale = next((s for s in info["scales"] if s["key"] == key), None)
    if scale is None:
        raise SystemExit("no scale {} in the info".format(key))
    if scale.get("sharding"):
        raise SystemExit("scale {} is already sharded".format(key))
    return scale


def write_shard(dst_dir, files):
    """Write one synthesized shard file atomically; returns its size."""
    if len(files) != 1:
        raise RuntimeError("expected one shard file, got {}".format(sorted(files)))
    (fname, blob), = files.items()
    tmp = os.path.join(dst_dir, fname + ".tmp")
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, os.path.join(dst_dir, fname))
    return len(blob)


def write_info(path, info, key, sharding):
    """Mark ``key`` as sharded in ``info`` and write it to ``path`` atomically.

    Every other scale is kept exactly as it is, in the same order: the service
    opens a scale by its position (``scale_index = mip``), and another scale
    may already be sharded with settings of its own.
    """
    out = dict(info)
    out["scales"] = [
        dict(s, sharding=sharding) if s["key"] == key else s for s in info["scales"]
    ]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)


def plan(src, key):
    info = json.load(open(os.path.join(src, "info")))
    scale = find_scale(info, key)
    chunk, offset, _, grid = scale_grid(scale)
    sharding = sharding_for(grid)
    spec = ShardingSpecification.from_dict(sharding)

    groups = defaultdict(list)
    skipped = []
    for name in os.listdir(os.path.join(src, key)):
        m = NAME.match(name)
        if not m:
            skipped.append(name)
            continue
        lo = np.array([int(m.group(i)) for i in (1, 3, 5)])
        gridpt = (lo - offset) // chunk
        if np.any((lo - offset) % chunk) or np.any(gridpt < 0) or np.any(gridpt >= grid):
            skipped.append(name)
            continue
        code = int(compressed_morton_code(gridpt, grid))
        shard = spec.compute_shard_location(code).shard_number
        groups[shard].append((code, name))
    return info, sharding, groups, skipped


def build_one(args):
    src_dir, dst_dir, sharding, members = args
    # The chunks are gzipped here, at a level of our choosing, and the shard is
    # then assembled as if they were raw. The info file still says gzip, which
    # is what readers go by.
    spec = ShardingSpecification.from_dict(dict(sharding, data_encoding="raw"))
    data = {}
    raw = 0
    for code, name in members:
        with open(os.path.join(src_dir, name), "rb") as f:
            blob = f.read()
        raw += len(blob)
        data[code] = gzip.compress(blob, compresslevel=GZIP_LEVEL, mtime=0)
    written = write_shard(dst_dir, spec.synthesize_shards(data))
    return len(members), raw, written


def run_jobs(fn, jobs, workers):
    """``fn`` over ``jobs`` in order, in a process pool unless ``workers`` is 1.

    One worker runs in this process: there is nothing to gain from a pool, and
    starting one from a process that already has tensorstore's threads running
    can abort on macOS -- which is how the tests run it.
    """
    if workers <= 1:
        yield from map(fn, jobs)
        return
    with ProcessPoolExecutor(workers) as pool:
        yield from pool.map(fn, jobs, chunksize=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("key", help="scale key, e.g. 32_32_45")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    if os.path.abspath(a.src) == os.path.abspath(a.dst):
        raise SystemExit("dst must differ from src; the source is never modified")

    info, sharding, groups, skipped = plan(a.src, a.key)
    n = sum(len(v) for v in groups.values())
    print(f"{n:,} chunks -> {len(groups):,} shards; sharding={sharding}", flush=True)
    if skipped:
        print(f"skipping {len(skipped)} unrecognised files, e.g. {skipped[:3]}", flush=True)
    if a.dry_run:
        return

    dst_scale_dir = os.path.join(a.dst, a.key)
    os.makedirs(dst_scale_dir, exist_ok=True)
    src_scale_dir = os.path.join(a.src, a.key)

    existing = set(os.listdir(dst_scale_dir))
    jobs = [
        (src_scale_dir, dst_scale_dir, sharding, m)
        for shard, m in sorted(groups.items())
        if shard + ".shard" not in existing
    ]
    print(f"{len(groups) - len(jobs):,} shards already done, {len(jobs):,} to build",
          flush=True)
    started = time.time()
    done = raw = written = 0
    for i, (c, r, w) in enumerate(run_jobs(build_one, jobs, a.workers), 1):
        done, raw, written = done + c, raw + r, written + w
        if i % 50 == 0 or i == len(jobs):
            elapsed = time.time() - started
            eta = elapsed / i * (len(jobs) - i)
            print(f"{i}/{len(jobs)} shards, {done:,} chunks, "
                  f"{raw/1e9:.1f} GB -> {written/1e9:.1f} GB, "
                  f"{elapsed/60:.0f} min elapsed, ~{eta/60:.0f} min left",
                  flush=True)

    # Written last, so a half-finished run is never mistaken for a readable
    # volume. Built on the destination's info if it already has one, so a
    # scale converted into it earlier keeps its own sharding entry.
    dst_info = os.path.join(a.dst, "info")
    base = json.load(open(dst_info)) if os.path.exists(dst_info) else info
    write_info(dst_info, base, a.key, sharding)
    print("info written: {} is now sharded".format(a.key), flush=True)


if __name__ == "__main__":
    main()
