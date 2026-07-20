# Connectome Services

This repository contains code for running various connectome-related services related currently deployed
on `flyem.mrc-lmb.cam.ac.uk` (internally `flyem1`):

- a supervoxel look-up service for the Zheng et al. CA3 volume
- a supervoxel look-up service for the Aedes aegypti EM volume
- a service for dynamically generating neuroglancer segmentation properties from tables on flytable
- a [sparse voxel service](#sparse-voxels) returning a segment's voxels as run-length encoded runs, instead of shipping raw chunks to the client

## Requirements

The project uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies and the virtual environment.

```bash
# Run to setup dependencies
uv sync
```

If you are running this server in production, you probably want a virtual environment:

```bash
# Create a virtual environment
uv env
```

In order to use the segment property service, you will also need to setup environment variables:
- `SEATABLE_SERVER`: URL for FlyTable
- `SEATABLE_TOKEN`: API token for FlyTable

_Note to self: on `flyem1` I added these to the `gunicorn_start` script._

## Sparse voxels

Returns the voxels of a segment as run-length encoded runs, the way DVID's
`sparsevol` does. The point is to stop clients downloading raw segmentation:
to get a neuron's voxels today you have to pull every chunk it touches and mask
them yourself, which is gigabytes over the wire to keep a few megabytes of
neuron. The service does that masking next to the data and sends you the
answer instead.

```bash
# Every voxel of a neuron, as runs
curl .../sparsevol/dataset/wclee_aedes_brain/s/0/root/720575940626838909 -o neuron.bin

# Or from supervoxel IDs you already hold
curl -X POST .../sparsevol/dataset/wclee_aedes_brain/s/0/supervoxels \
     -H 'Content-type: application/json' \
     -d '{"supervoxels": ["81624458144052736", "81624458144052737"]}'

# What is available, and how each dataset is served
curl .../sparsevol/datasets
```

The default response is binary: an `(M, 4)` little-endian int32 array of runs,
each `x, y, z, length`, meaning `length` voxels from `(x, y, z)` along +X.
Coordinates are voxels at the requested scale. Add `?fmt=` for `npy`, `json`,
or `coords` (an `(N, 3)` array with every voxel spelled out — far larger, so
prefer the default). Responses are gzipped when the client accepts it.

```python
import numpy as np, requests

r = requests.get(".../sparsevol/dataset/wclee_aedes_brain/s/0/root/720575940626838909")
runs = np.frombuffer(r.content, dtype="<i4").reshape(-1, 4)
print(r.headers["X-Sparsevol-Voxels"], "voxels in", len(runs), "runs")
print("read", r.headers["X-Sparsevol-Voxels-Read"], "voxels to get there")
```

Every response carries `X-Sparsevol-*` headers accounting for what the request
actually did — voxels returned, runs, chunks read, and how many dense voxels
were read and discarded to produce them.

### How it knows which chunks to read

These segmentations are flat: the stored labels are watershed supervoxels, and
nothing about a raw label says where it lives. What makes the lookup tractable
is the **chunkedgraph sitting on top of them**. Its labels are graphene labels
with the chunk position packed into the high bits, so:

- `get_leaves(root, stop_layer=2)` gives the chunks a neuron occupies without
  reading a single voxel, and
- a supervoxel query needs no graph call at all — the chunk falls straight out
  of arithmetic on the ID.

So the read is confined to the chunks the segment actually occupies. The graph
is consulted only for the manifest and the geometry; **image data is always read
from the local copy of the segmentation**, never through graphene.

Each dataset therefore needs two things in `DATASOURCES`: `sparsevol` in its
`services`, and a `graphene` source pointing at its chunkedgraph.

### Cost, and the limits on it

The dense read happens per request the first time anyone asks — there is no
precomputed index. That is affordable only because the segmentation is on the
same machine: it trades the client's bandwidth for the server's local I/O.
Repeat requests are served from the [layer-2 cache](#the-layer-2-cache) below.

It is not free, though. Graph chunks are large — aedes is 512×512×128 at mip 0,
so **one chunk is 33.5M voxels and ~268 MB once decompressed to uint64**, and
CA3 at mip 1 is about half that. Two ceilings in [`app/config.py`](app/config.py)
keep a single request from running away, both overridable per dataset:

| setting | default | meaning |
|---|---|---|
| `SparseVolMaxChunks` | 256 | chunks one request may read |
| `SparseVolMaxVoxels` | 4e9 | voxels one request may read (~32 GB, ~120 aedes chunks) |
| `SparseVolMaxWorkers` | 4 | concurrent chunk reads *within* a request; each holds a whole chunk, so ~1 GB peak |
| `SparseVolMaxConcurrent` | 2 | reads in flight at once, process-wide (~1.1 GB each) |
| `SparseVolQueueSeconds` | 20 | how long a request waits for a slot before being shed |

Going over the size limits returns a 400 naming the limit rather than occupying
a worker until something times out. If real neurons turn out to span more than
this, raise the limits per dataset — but raise them knowing what they cost.

`SparseVolMaxConcurrent` is the one that protects the machine rather than the
request. Size limits bound a single read; without a concurrency cap nothing
stops several large reads arriving together, each claiming its own chunks,
which is an out-of-memory kill rather than an error. Requests queue for a slot
and are shed with a 503 and `Retry-After` if none frees up in time. The wait is
reported in `X-Sparsevol-Queued-Seconds`, which is what to watch when tuning.

**The cap is per worker process.** Under gunicorn the real ceiling is
`(gunicorn workers) × SparseVolMaxConcurrent × ~1.5 GB`, so size it against the
machine's memory and how many workers you run, not against this number alone.

Two deployment settings matter alongside these: a read near the size limit can
take a while, so gunicorn's worker `timeout` and nginx's `proxy_read_timeout`
both need to exceed it, or a slow success arrives as a 502 with a killed worker.

Requesting a coarser scale is the cheap way to reduce work, where a dataset has
one downloaded locally.

### The layer-2 cache

Most of that dense read is repeated work: ask about the same neuron twice, or
about two neurons sharing a branch, and the same chunks get read and masked
again. So the runs are cached per **layer-2 node** — see
[`app/l2cache.py`](app/l2cache.py). A fully cached neuron reads no image data
and makes exactly one chunkedgraph call.

Layer 2 is the useful key because proofreading mostly leaves it alone. A merge
spanning two chunks adds an edge *above* layer 2 and mints no new L2 node at
all; only within-chunk merges and splits recompute one, and only for the chunks
the cut touches. So an edited neuron pays for the part that changed and reuses
the rest, and entries that do go stale are simply never asked for again.

`X-Sparsevol-L2-Cached` and `-L2-Computed` report the split per request, and
`GET /sparsevol/cache` reports how full the store is.

```bash
curl .../sparsevol/cache
```

| setting | default | meaning |
|---|---|---|
| `L2CacheEnabled` | `True` | off falls back to computing every request live |
| `L2_CACHE_PATH` | `./l2_cache.sqlite` | env-overridable; put it on a roomy disk |
| `L2CacheMaxBytes` | 50 GB | ~25,000 aedes neurons at mip 1 |
| `L2CacheMaxKeys` | 20M | the other ceiling |
| `L2CacheManifestWorkers` | 8 | concurrent chunkedgraph calls on a cold read |

**It fails loudly when full.** On reaching either ceiling the cache stops
accepting writes and *every* sparse volume request starts returning 503 —
including ones it could have served, and the supervoxel endpoint that never
touches it. This is deliberate. An LRU would quietly thrash at the ceiling and
look, from outside, exactly like a cache that is working; the thing worth
noticing is the ceiling being reached at all, and that is a capacity decision
for a person. The 503 names the three ways out: raise the limits, clear the
cache, or set `L2CacheEnabled = False`.

Sizing, measured on aedes at mip 1: ~1,650 runs per L2 node, ~3.8 kB packed,
~500 L2 nodes for a large neuron — so roughly 2 MB per neuron.

One cost worth knowing about: attributing a voxel to an L2 node needs that
node's supervoxel manifest, and there is no batch endpoint, so a cold neuron
costs one chunkedgraph GET per L2 node. It is paid once per node ever. With
caching disabled the service uses the old path instead — a single manifest call
for the whole root — which is why both paths still exist.

### Warming the cache

Populating happens on demand, so this is optional; it only moves the cost of
the first request to a time of your choosing.

```bash
uv run python -m app.l2cache_warm --stats
uv run python -m app.l2cache_warm wclee_aedes_brain --scale 1 --roots roots.txt
uv run python -m app.l2cache_warm --clear --dataset wclee_aedes_brain
```

It is not a loop over the endpoint. Neurons share chunks, so it gathers the L2
nodes of every root first, groups them by chunk — which needs no extra graph
traffic, since a node's chunk falls out of arithmetic on its ID — and reads
each chunk once for all the nodes in it. On aedes that is the difference
between ~120 hours and ~18 for ten thousand neurons. `--chunks` bounds how much
is held per pass; `--dry-run` reports the work without reading a voxel.

Run it against a quiet server: it holds the same chunks in memory as a request
does, and its concurrency limit is its own — the service's cap lives in the
service's process and knows nothing about this one.

### Precomputed indexes (optional, unused today)

The same endpoints can also be served from a precomputed **supervoxel → RLE**
index instead of reading chunks live, configured via `SPARSEVOL_DATASOURCES`.
Mode is a per-dataset deployment detail: the request and the response are
identical either way, and `/sparsevol/datasets` reports which is in use.

That path exists for a dataset too large to sparsify on demand — FlyWire, say,
where a mip-0 neuron is ~20 billion voxels and the volume is not local. It
moves the dense read to a one-time offline build keyed on supervoxel IDs, which
are immutable, so the index never needs invalidating. **No such index is built
today**, so a root query against `flywire_public` will return empty until one
exists. [`app/sparsevol_build.py`](app/sparsevol_build.py) builds one for the
chunks a single neuron occupies:

```bash
uv run python -m app.sparsevol_build flywire_public 720575940626838909 --mip 2
```

That builder is a bootstrap, not a production job: it reads serially, one box
per graph chunk, with no ceiling — see the notes at the top of the module for
what a real in-region batch build should do differently. Background and
measurements are in [`pcg_sparse`](https://github.com/schlegelp/pcg_sparse),
whose RLE codec and chunk conventions this shares.

> The `graphene` URL on the `flywire_public` entry has not been checked against
> a live datastack — confirm it with
> `CAVEclient("flywire_fafb_public").info.segmentation_source()` before relying
> on it. The aedes and CA3 sources have been verified to open and authenticate.

## Run the web service locally
```uv run uvicorn --reload app.main:app```

## Run tests
```uv run pytest```

## Run in production

We deployed this service on `flyem1` using:

- `gunicorn` as the webserver
- `supervisor` as the process control system for the gunicorn server
- a reverse proxy set up in `nginx` forwarding requests to the gunicorn webserver

Please see [this tutorial](https://dylancastillo.co/posts/fastapi-nginx-gunicorn.html) for general step-by-step instructions.

[This Slack message](https://flyconnectome.slack.com/archives/C29G9694H/p1740648367149559) contains details on how
the service is currently deployed on `flyem1`.

To restart the service (e.g. after changing the code), you can run:

```bash
sudo supervisorctl restart transform-service
```

