import os

# Number of cores used to parallel fetching of locations
MaxWorkers = 16

# Max number of locations per query
MaxLocations = 10e9

# Number of chunks for each worker to read
# Each chunk dimension is multiplied by this.
# e.g. 4 will lead to 64 (4*4*4) chunks per worker
CHUNK_MULTIPLIER = 1

# Max number of supervoxels resolved in a single sparsevol request. A large
# proofread FlyWire neuron runs to ~30k supervoxels; the ceiling is here so a
# request for something pathological fails with an explanation rather than
# looking like a hang.
MaxSupervoxels = 500_000

# Ceilings for the live sparsevol service, which reads and masks real chunks
# per request. Both are per-dataset overridable. They exist so a request that
# cannot finish says so immediately instead of occupying a worker until
# something times out.
#
# Sizing these needs the graph chunk size, which is large: aedes is 512x512x128
# at mip 0, so one chunk is 33.5M voxels and 268 MB once decompressed to uint64
# (CA3 at mip 1 is about half that). The voxel ceiling below is therefore worth
# ~16 GB of reads, or roughly 60 aedes chunks. Raise it per dataset if real
# neurons turn out to span more, but raise it knowing what it costs.
SparseVolMaxChunks = 256
SparseVolMaxVoxels = 2_000_000_000

# Chunks are read concurrently and each worker holds a whole decompressed chunk,
# so this multiplies straight into peak memory -- 4 aedes chunks is ~1 GB. Kept
# well below MaxWorkers, which is tuned for point queries that hold almost
# nothing.
SparseVolMaxWorkers = 4

# How many sparsevol reads may be in flight at once. Without this, nothing stops
# several large requests arriving together and each claiming its own set of
# chunks, which is an out-of-memory kill rather than a clean error.
#
# Budget roughly 1.1 GB per slot for aedes at mip 0: SparseVolMaxWorkers chunks
# held at once (~1 GB) plus the coordinates accumulated from them. Those are
# kept 32-bit rather than numpy's default int64, which measured at ~480 MB peak
# for a 10M-voxel neuron against ~740 MB before.
#
# NOTE: this is per worker *process*. Under gunicorn the real ceiling is
# (gunicorn workers) x SparseVolMaxConcurrent x ~1.5 GB, so size it against the
# machine's memory and the number of workers you run, not this number alone.
SparseVolMaxConcurrent = 2

# How long a request waits for a slot before giving up. Long enough to ride out
# a burst, short enough that a queue cannot outlive the client or the proxy's
# read timeout.
SparseVolQueueSeconds = 20

# Where the supervoxel -> RLE fragment stores live. Anything CloudFiles can
# read: a local path, gs://, s3:// or https://.
SPARSEVOL_INDEX_ROOT = os.environ.get(
    "SPARSEVOL_INDEX_ROOT",
    "file://" + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sparsevol_index")),
)

SPARSEVOL_DATASOURCES = {
    ### We are currently not planning to build an index for the FlyWire public segmentation,
    ### so this is commented out.
    # "flywire_public": {
    #     "description": "Sparse voxels for FlyWire FAFB public segmentation",
    #     # Only used to resolve a root ID to its supervoxels; no image data is
    #     # ever read through this. Confirm this matches the datastack before
    #     # serving -- it is whatever CAVE reports:
    #     #   CAVEclient("flywire_fafb_public").info.segmentation_source()
    #     "graphene": "graphene://https://prod.flywire-daf.com/segmentation/1.0/flywire_public",
    #     "index": SPARSEVOL_INDEX_ROOT + "/flywire_public",
    #     "scales": [0, 1, 2, 3, 4],  # mips the index has been built for
    #     "voxel_size": [4, 4, 40],
    #     "max_supervoxels": 200_000,
    # },
    "test_index": {
        "description": "Synthetic index built by the test suite",
        "index": "file://"
        + os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "sparsevol_test_index")
        ),
        "scales": [0],
        "voxel_size": [4, 4, 40],
        # No graphene source, so root IDs cannot be resolved and the chunk bit
        # layout is stated here instead of read from the volume's metadata.
        "chunk_layout": {"spatial_bits": 8, "layer_bits": 8},
    },
}

DATASOURCES = {
    "test": {
        "description": "Test volume",  # Description of data
        "type": "zarr",  # Which datatype?
        "scales": [7],  # List of mip levels available
        "voxel_size": [4, 4, 40],  # Base resolution (mip 0)
        "services": ["transform"],  # Is this for the 'transform' or 'query' service?
        "dtype": "float32",  # What datatype is stored?
        "width": 2,  # How many elements are stored? (e.g., dx,dy for transforms)
        "tsinfo": {  # Details for the tensorstore library to open the data
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": "test.zarr",
            },
        },
    },
    "test_segmentation": {
        "description": "Synthetic watershed volume built by the test suite",
        "type": "neuroglancer_precomputed",
        "scales": [0],
        "voxel_size": [16, 16, 45],
        "services": ["query", "sparsevol"],
        # Never opened: the tests substitute a stand-in graph. Real datasets
        # read their chunk layout from graphene metadata instead of stating it.
        "graphene": "graphene://example.invalid/not-opened-in-tests",
        "chunk_layout": {"spatial_bits": 8, "layer_bits": 8},
        "dtype": "uint64",
        "width": 1,
        "tsinfo": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "file",
                "path": "sparsevol_test_volume",
            },
        },
    },
    "zheng_ca3_v2": {
        "description": "super voxel segmentation of Zhihao's CA3 dataset [v0.2-8nm-updown3x-m0.01-no-inverse_18-18-45_20240720175632]",
        "type": "neuroglancer_precomputed",
        "scales": [
            1
        ],  # the dataset only has mip 0 and 1, and I only downloaded mip 1 so far
        "voxel_size": [18, 18, 45],
        "services": ["query", "sparsevol"],
        # The chunkedgraph sitting on top of this watershed volume. Used only
        # to resolve roots and to locate chunks -- voxels are always read from
        # the local copy above.
        "graphene": "graphene://https://minnie.microns-daf.com/segmentation/table/zheng_ca3",
        "dtype": "uint64",
        "width": 1,
        "tsinfo": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "file",
                "path": "segmentation/v0.2-8nm-updown3x-m0.01-no-inverse_18-18-45_20240720175632",
            },
        },
    },
    "wclee_aedes_brain": {
        "description": "super voxel segmentation of the mosquito whole brain [wclee_aedes_brain]",
        "type": "neuroglancer_precomputed",
        "scales": [
            0
        ],  # the dataset has mip 0 and 1 but I only downloaded mip 0 so far
        "voxel_size": [16, 16, 45],
        "services": ["query", "sparsevol"],
        "graphene": "graphene://https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain",
        "dtype": "uint64",
        "width": 1,
        "tsinfo": {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "file",
                "path": "segmentation/wclee_aedes_brain",
            },
        },
    },
}
