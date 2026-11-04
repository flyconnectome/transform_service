"""Download one scale of an unsharded precomputed volume straight into gzip shards.

The alternative to downloading with ``gs cp`` and then running
:mod:`to_sharded`. Chunks are requested with ``Accept-Encoding: gzip``, so the
bucket sends its stored gzip bytes unchanged, and those bytes go into the shard
as they are -- the shard's ``data_encoding`` is gzip. Nothing is unzipped,
re-zipped or staged on disk on the way. For aedes mip 0 that was 363 GiB
written and ~45 minutes, against the 1.7 TB an unzipped download took.

The scale's metadata comes from DST/info, which must already list the scale
(copy the bucket's info there first). Only that scale gains a ``sharding``
entry; every other scale, including ones already sharded, is left as it is.

    uv run python tools/fetch_sharded.py BASE_URL DST_DIR SCALE_KEY \\
        [--processes N] [--threads N] [--dry-run]

    # e.g. aedes mip 0, next to the mip 1 already in that folder
    uv run python tools/fetch_sharded.py \\
        https://storage.googleapis.com/fanc-fly_segmentation/wclee_aedes_brain \\
        /fafbz/aedes/wclee_aedes_brain_sharded 16_16_45 --processes 8 --threads 32

Throughput is bound by request latency, and one process tops out around
50 MB/s however many threads it has, so scale with ``--processes``: 4 measured
~150 MB/s from flyem1. Resumable like :mod:`to_sharded`; a chunk the bucket
does not have is counted and skipped, since tensorstore reads it as empty.
"""

import argparse
import gzip
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from cloudvolume.datasource.precomputed.image.common import compressed_morton_code
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification

from to_sharded import (
    GZIP_LEVEL,
    find_scale,
    run_jobs,
    scale_grid,
    sharding_for,
    write_info,
    write_shard,
)

RETRIES = 6


def chunk_name(lo, hi):
    return "{}-{}_{}-{}_{}-{}".format(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def plan(info, key):
    scale = find_scale(info, key)
    chunk, offset, size, grid = scale_grid(scale)
    sharding = sharding_for(grid)
    spec = ShardingSpecification.from_dict(sharding)

    gx, gy, gz = np.meshgrid(*[np.arange(g) for g in grid], indexing="ij")
    gridpts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    codes = compressed_morton_code(gridpts, grid)

    groups = defaultdict(list)
    end = offset + size
    for pt, code in zip(gridpts, codes):
        lo = offset + pt * chunk
        # Chunks on the far edge are clipped to the volume, and named so.
        hi = np.minimum(lo + chunk, end)
        groups[spec.compute_shard_location(int(code)).shard_number].append(
            (int(code), chunk_name(lo, hi))
        )
    return sharding, groups, len(gridpts)


_session = None


def session(threads):
    """One HTTP session per process, with a connection per download thread."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=threads))
    return _session


def fetch(url, threads):
    """The chunk's stored gzip bytes, or None if the bucket has no such chunk."""
    for attempt in range(RETRIES):
        try:
            r = session(threads).get(
                url, headers={"Accept-Encoding": "gzip"}, stream=True, timeout=120
            )
            if r.status_code == 404:
                return None
            if r.status_code != 200:
                raise IOError("HTTP {} for {}".format(r.status_code, url))
            blob = r.raw.read(decode_content=False)
            expected = r.headers.get("Content-Length")
            if expected is not None and int(expected) != len(blob):
                raise IOError("short read for {}: {} of {}".format(url, len(blob), expected))
            if r.headers.get("Content-Encoding", "").lower() == "gzip":
                if blob[:2] != b"\x1f\x8b":
                    raise IOError("not gzip despite Content-Encoding: {}".format(url))
                return blob
            # Stored without gzip: zip it here so the shard stays consistent.
            return gzip.compress(blob, compresslevel=GZIP_LEVEL, mtime=0)
        except Exception:
            # Broadly, on purpose: reading r.raw directly surfaces urllib3's own
            # errors (a dropped connection is urllib3.exceptions.ProtocolError),
            # which are neither requests nor IOError exceptions -- the first
            # aedes mip 0 run died on exactly that. Every failure here is a
            # transient network problem or a short read, and a chunk that still
            # fails after the last attempt stops the run loudly.
            if attempt == RETRIES - 1:
                raise
            time.sleep(min(30, 2 ** attempt))


def build_one(args):
    base_url, dst_dir, sharding, members, threads = args
    spec = ShardingSpecification.from_dict(dict(sharding, data_encoding="raw"))
    with ThreadPoolExecutor(threads) as pool:
        blobs = list(pool.map(lambda m: fetch(base_url + m[1], threads), members))
    data = {code: blob for (code, _), blob in zip(members, blobs) if blob is not None}
    missing = len(members) - len(data)
    if not data:
        return 0, missing, 0
    return len(data), missing, write_shard(dst_dir, spec.synthesize_shards(data))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("base_url", help="e.g. https://storage.googleapis.com/bucket/volume")
    p.add_argument("dst")
    p.add_argument("key", help="scale key, e.g. 16_16_45")
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--threads", type=int, default=32, help="downloads in flight per process")
    p.add_argument("--only", nargs="*", help="build just these shard numbers (testing)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    info_path = os.path.join(a.dst, "info")
    info = json.load(open(info_path))
    sharding, groups, n_chunks = plan(info, a.key)
    print("{:,} chunks -> {:,} shards; sharding={}".format(n_chunks, len(groups), sharding),
          flush=True)
    if a.dry_run:
        return

    base_url = a.base_url.rstrip("/") + "/" + a.key + "/"
    dst_dir = os.path.join(a.dst, a.key)
    os.makedirs(dst_dir, exist_ok=True)
    existing = set(os.listdir(dst_dir))
    wanted = sorted(a.only) if a.only else sorted(groups)
    jobs = [(base_url, dst_dir, sharding, groups[s], a.threads)
            for s in wanted if s + ".shard" not in existing]
    print("{:,} shards already done, {:,} to build".format(len(wanted) - len(jobs), len(jobs)),
          flush=True)

    started = time.time()
    chunks = missing = written = 0
    for i, (c, m, w) in enumerate(run_jobs(build_one, jobs, a.processes), 1):
        chunks, missing, written = chunks + c, missing + m, written + w
        if i % 50 == 0 or i == len(jobs):
            elapsed = time.time() - started
            print("{}/{} shards, {:,} chunks, {:,} missing in bucket, {:.1f} GB written, "
                  "{:.0f} min elapsed, ~{:.0f} min left".format(
                      i, len(jobs), chunks, missing, written / 1e9, elapsed / 60,
                      elapsed / i * (len(jobs) - i) / 60), flush=True)

    if a.only:
        print("--only given: info left unchanged", flush=True)
        return
    write_info(info_path, info, a.key, sharding)
    print("info updated: {} is now sharded".format(a.key), flush=True)


if __name__ == "__main__":
    main()
