import os

# Number of cores used to parallel fetching of locations
MaxWorkers = 16

# Max number of locations per query
MaxLocations = 10e9

# Number of chunks for each worker to read
# Each chunk dimension is multiplied by this.
# e.g. 4 will lead to 64 (4*4*4) chunks per worker
CHUNK_MULTIPLIER = 1

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
    "zheng_ca3_v2": {
        "description": "super voxel segmentation of Zhihao's CA3 dataset [v0.2-8nm-updown3x-m0.01-no-inverse_18-18-45_20240720175632]",
        "type": "neuroglancer_precomputed",
        "scales": [
            1
        ],  # the dataset only has mip 0 and 1, and I only downloaded mip 1 so far
        "voxel_size": [18, 18, 45],
        "services": ["query"],
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
        "services": ["query"],
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
