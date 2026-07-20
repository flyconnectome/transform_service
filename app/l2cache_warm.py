"""Fill the layer-2 RLE cache ahead of time, for a list of root IDs.

The service populates the cache on demand, so this is never required -- it only
moves the cost of the first request for each neuron to a time of your choosing.

**Why this is not just a loop over the endpoint.** Neurons share chunks, and a
chunk read for one neuron would be read again for the next. This gathers the
layer-2 nodes of every root first, groups them by chunk, and then reads each
chunk once for all the nodes in it. On aedes that is the difference between
~120 hours and ~18 for ten thousand neurons -- and grouping needs no extra
graph traffic, since a node's chunk falls out of arithmetic on its ID.

Run it against a quiet server. It holds the same chunks in memory as a request
does, and its concurrency limit is its own -- the service's cap lives in the
service's process and knows nothing about this one.

    # what is in there now
    uv run python -m app.l2cache_warm --stats

    # warm a list of roots, one ID per line
    uv run python -m app.l2cache_warm wclee_aedes_brain --scale 1 --roots roots.txt

    # start over
    uv run python -m app.l2cache_warm --clear --dataset wclee_aedes_brain
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config
from . import l2cache
from . import sparsevol_live


def log(message):
    print(message, flush=True)


def human_bytes(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "{:.1f} {}".format(n, unit)
        n /= 1024


def report(cache):
    stats = cache.stats()
    log("cache      {}".format(stats["path"]))
    log(
        "keys       {:,} / {:,} ({:.1%})".format(
            stats["n_keys"], stats["max_keys"], stats["fraction_keys"]
        )
    )
    log(
        "accounted  {} / {} ({:.1%})".format(
            human_bytes(stats["n_bytes"]),
            human_bytes(stats["max_bytes"]),
            stats["fraction_bytes"],
        )
    )
    log("on disk    {}".format(human_bytes(stats["file_bytes"])))
    if stats["full"]:
        log("FULL -- the service is refusing every sparse volume request.")


def read_roots(args):
    roots = [int(r) for r in args.roots]
    if args.root_file:
        with open(args.root_file) as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    roots.append(int(line))
    if args.limit:
        roots = roots[: args.limit]
    # Order is irrelevant once we group by chunk, and duplicates would only
    # cost a wasted lookup.
    return sorted(set(roots))


def l2_nodes_for_roots(dataset, roots, workers):
    """Every root's layer-2 nodes, deduplicated. No image data is read."""
    graph = sparsevol_live.get_graph(dataset)
    bounds = graph.meta.bounds(0)
    done = [0]

    def fetch(root_id):
        try:
            leaves = graph.get_leaves(int(root_id), bounds, 0, stop_layer=2)
        except Exception as exc:
            log("  root {} failed: {}".format(root_id, exc))
            leaves = []
        done[0] += 1
        if done[0] % 100 == 0:
            log("  resolved {:,}/{:,} roots".format(done[0], len(roots)))
        return np.asarray(leaves, dtype=np.uint64).ravel()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = list(pool.map(fetch, roots))

    found = [f for f in found if f.size]
    if not found:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(found))


def chunk_slabs(dataset, l2_ids, chunks_per_pass):
    """Group nodes by chunk, then yield whole chunks in fixed-size slabs.

    Slabs bound how much is held at once. Chunks are never split across slabs,
    so the promise that each chunk is read exactly once survives the slicing.
    """
    layout = sparsevol_live.get_layout(dataset)
    groups = {}
    for l2_id, position in zip(l2_ids.tolist(), layout.decode(l2_ids)):
        groups.setdefault(tuple(int(v) for v in position), []).append(int(l2_id))

    slab = []
    for _, members in sorted(groups.items()):
        slab.append(members)
        if len(slab) >= chunks_per_pass:
            yield [i for group in slab for i in group]
            slab = []
    if slab:
        yield [i for group in slab for i in group]


def warm(args, cache):
    roots = read_roots(args)
    if not roots:
        log("No root IDs given. Pass them as arguments or with --roots FILE.")
        return 1
    log("{:,} root IDs".format(len(roots)))

    started = time.time()
    l2_ids = l2_nodes_for_roots(args.dataset, roots, args.graph_workers)
    log("{:,} distinct layer-2 nodes".format(len(l2_ids)))
    if not len(l2_ids):
        return 1

    present = cache.have(args.dataset, args.scale, l2_ids)
    missing = np.array([i for i in l2_ids.tolist() if i not in present], dtype=np.uint64)
    log(
        "{:,} already cached, {:,} to compute".format(len(present), len(missing))
    )
    if not len(missing):
        log("Nothing to do.")
        return 0

    slabs = list(chunk_slabs(args.dataset, missing, args.chunks))
    log("{:,} passes of up to {} chunks".format(len(slabs), args.chunks))

    if args.dry_run:
        log("--dry-run: stopping before reading any voxels.")
        return 0

    written = 0
    voxels_read = 0
    for index, ids in enumerate(slabs, start=1):
        if cache.is_full():
            log("")
            log("STOPPING: the cache hit its ceiling after {:,} new keys.".format(written))
            report(cache)
            return 2

        stats = sparsevol_live.LiveStats()
        fragments = sparsevol_live.compute_l2_fragments(
            args.dataset, args.scale, ids, stats, enforce_budget=False
        )
        written += cache.put_many(args.dataset, args.scale, fragments)
        voxels_read += stats.voxels_read

        elapsed = time.time() - started
        log(
            "pass {:,}/{:,}  {:,} nodes  {:,} chunks  {} read  {:,} keys  {}  {:.0f}s".format(
                index,
                len(slabs),
                len(ids),
                stats.n_chunks,
                human_bytes(voxels_read * 8),
                written,
                human_bytes(cache.counters()[1]),
                elapsed,
            )
        )

    log("")
    log("Done in {:.0f}s. {:,} new keys.".format(time.time() - started, written))
    report(cache)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m app.l2cache_warm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", nargs="?", help="dataset name from DATASOURCES")
    parser.add_argument("roots", nargs="*", help="root IDs to warm")
    parser.add_argument("--roots", dest="root_file", help="file of root IDs, one per line")
    parser.add_argument("--scale", type=int, default=0, help="mip level to cache")
    parser.add_argument(
        "--chunks",
        type=int,
        default=config.SparseVolMaxChunks,
        help="chunks held in one pass; the memory dial (default: %(default)s)",
    )
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=config.L2CacheManifestWorkers,
        help="concurrent chunkedgraph calls (default: %(default)s)",
    )
    parser.add_argument("--limit", type=int, help="only warm the first N roots")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be read, then stop",
    )
    parser.add_argument("--stats", action="store_true", help="report cache usage and exit")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="delete cached entries and reclaim the disk, then exit",
    )
    parser.add_argument("--dataset", dest="only_dataset", help="restrict --clear")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Built directly rather than through get_cache(), so --stats and --clear
    # still work when L2CacheEnabled has been turned off to get the service
    # running again -- which is exactly when you need them.
    cache = l2cache.L2Cache(config.L2_CACHE_PATH)

    if args.stats:
        report(cache)
        return 0

    if args.clear:
        deleted = cache.clear(
            dataset=args.only_dataset, scale=args.scale if args.only_dataset else None
        )
        log("Deleted {:,} entries.".format(deleted))
        report(cache)
        return 0

    if not args.dataset:
        build_parser().print_usage()
        log("A dataset is required unless using --stats or --clear.")
        return 1

    if not config.L2CacheEnabled:
        log("Note: config.L2CacheEnabled is False, so the service will not read this.")

    return warm(args, cache)


if __name__ == "__main__":
    sys.exit(main())
