#!/usr/bin/env python3
import gzip
import io
import os

import numpy as np

from enum import Enum
from typing import List, Tuple, Dict
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi import FastAPI, HTTPException, Response, Request

from pydantic import BaseModel
from msgpack_asgi import MessagePackMiddleware

from . import config
from . import sparsevol
from . import sparsevol_live
from .rle import decode_rle
from .query import map_points, query_points
from .annotations import (
    get_flywire_segmentation_properties,
    get_aedes_segmentation_properties,
    get_fanc_segmentation_properties,
    get_zhengCA3_segmentation_properties,
    get_banc_segmentation_properties,
)


API_DESCRIPTION = """
This service takes a set of points and looks up the values at those points in a dataset. The dataset can be a segmentation volume, a displacement vector field, or any other data that can be indexed by a set of 3D coordinates.

Query units should be in *pixels* at full resolution (e.g. mip=0), which generally maps to the coordinates shown in CATMAID or Neuroglancer.

Depending on the dataset, the return values are either the segmentation ID at the given location or the displacement vector at that location.

The selection of scale (mip) selects the granularity of the field being used, but will not change the units.

Error values are returned as `null`, not `NaN` as done with the previous iteration of this service. The most likely cause of an error is being out-of-bounds of the underlying array.

_Note on using [msgpack](https://msgpack.org/)_: Use `Content-type: application/x-msgpack` and `Accept: application/x-msgpack` to use msgpack instead of JSON. There is currently data size limit of *64KB* when using msgpack.
(<a href="https://github.com/florimondmanca/msgpack-asgi/issues/11">GitHub issue</a>).

"""

TEMPLATES = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

TAGS_METADATA = [
    {
        "name": "transform",
        "description": "Services to transform a set of points using a precomputed displacement field (e.g., FAFBv14 to FlyWire)",
    },
    {
        "name": "query",
        "description": "Retrieve the values stored at a set of points (e.g. segmentation values at a set of voxels)",
    },
    {
        "name": "mapping",
        "description": "Retrieve the mapping between two datasets",
    },
    {
        "name": "annotations",
        "description": "Retrieve segmentation annotations for FlyWire neurons",
    },
    {
        "name": "sparsevol",
        "description": (
            "Sparse voxels for a segment, as run-length encoded runs, served "
            "from a precomputed supervoxel index. No dense image data is read."
        ),
    },
    {"name": "other"},
    {"name": "deprecated"},
]


app = FastAPI(
    default_response_class=ORJSONResponse,
    title="Transformation Service",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    # we need to set a root path b/c we're running this behind a nginx proxy which
    # adds "/transform-service" as prefix to all routes
    root_path="/transform-service",
    debug=False,  # turn on for debugging
)

# MessagePackMiddleware does not currently support large request (`more_body`) so we'll do our own...
app.add_middleware(MessagePackMiddleware)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Main page of the service
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.get("/info/", tags=["other"])
async def dataset_info():
    """Retrieve a list of available datasources."""
    cleaned_datasets = {}
    for k, info in config.DATASOURCES.items():
        cleaned_datasets[k] = {}
        for field in ("scales", "voxel_size", "description", "dtype"):
            cleaned_datasets[k][field] = info.get(field, None)

    return cleaned_datasets


# Datasets to be displayed in UI if they are part of an enum...
# This is a hack to populate the values.
DataSetName = Enum(
    "DataSetName", dict(zip(config.DATASOURCES.keys(), config.DATASOURCES.keys()))
)


# Single point vector field query
class PointResponse(BaseModel):
    x: float
    y: float
    z: float
    dx: float
    dy: float


class PointList(BaseModel):
    locations: List[Tuple[float, float, float]]


@app.post(
    "/transform/dataset/{dataset}/s/{scale}/values",
    response_model=List[PointResponse],
    tags=["transform"],
)
async def transform_values(dataset: DataSetName, scale: int, data: PointList):
    """Return dx, dy and new coordinates for an input set of locations."""

    locs = np.array(data.locations).astype(np.float32)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(
            status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations),
        )

    # scale & adjust locations
    transformed = await run_in_threadpool(map_points, dataset.value, scale, locs)

    # Apply results
    results = []
    for i in range(transformed.shape[0]):
        row = transformed[i]
        results.append(
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "dx": float(row["dx"]),
                "dy": float(row["dy"]),
            }
        )

    return results


@app.post(
    "/query/dataset/{dataset}/s/{scale}/cloud_volume_server",
    response_model=List[str],
    tags=["query"],
)
async def query_values_cloud_volume_server(
    dataset: DataSetName, scale: int, data: PointList
):
    """
    Implements the [CloudVolumeServer](https://github.com/flyconnectome/CloudVolumeServer) API.
    """

    locs = np.array(data.locations).astype(np.float32)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(
            status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations),
        )

    data = await run_in_threadpool(query_points, dataset.value, scale, locs)
    data = data.flatten()

    return data.tolist()


class ColumnPointList(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]


class ColumnPointListResponse(BaseModel):
    x: List[float]
    y: List[float]
    z: List[float]
    dx: List[float]
    dy: List[float]


class QueryColumnPointListResponse(BaseModel):
    values: List[List[float]]


@app.post(
    "/query/dataset/{dataset}/s/{scale}/values_array",
    response_model=QueryColumnPointListResponse,
    tags=["query"],
)
async def query_values_array(dataset: DataSetName, scale: int, locs: ColumnPointList):
    """Return segment IDs at given locations.
    Note: This function returns float(s). For segments, use values_array_string_response.
    """

    # Get a Nx3 array of points
    locs = np.array([locs.x, locs.y, locs.z]).astype(np.float32).swapaxes(0, 1)

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(
            status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations),
        )

    data = await run_in_threadpool(query_points, dataset.value, scale, locs)
    # Nx1 to 1xN
    data = data.swapaxes(0, 1)

    # Set results
    results = {"values": data.tolist()}

    return results


class ColumnPointListStringResponse(BaseModel):
    values: List[List[str]]


@app.post(
    "/query/dataset/{dataset}/s/{scale}/values_array_string_response",
    response_model=ColumnPointListStringResponse,
    tags=["query"],
)
async def query_values_array_string(
    dataset: DataSetName, scale: int, locs: ColumnPointList
):
    """Return segment IDs at given locations.
    Like *query_values_array*, but result array contains strings for easier parsing in R.
    """

    results = await query_values_array(dataset, scale, locs)

    results = {"values": [[str(j) for j in i] for i in results["values"]]}

    return results


class BinaryFormats(str, Enum):
    array_3xN = "array_float_3xN"
    array_Nx3 = "array_float_Nx3"


@app.post(
    "/transform/dataset/{dataset}/s/{scale}/values_binary/format/{format}",
    response_model=None,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Binary encoding of output array.",
        }
    },
    tags=["transform"],
)
async def transform_values_binary(
    dataset: DataSetName, scale: int, format: BinaryFormats, request: Request
):
    """Raw binary version of the API. Data will consist of 1 uint 32.
    Currently acceptable formats consist of a single uint32 with the number of records,
    All values must be little-endian floating point nubers.

    The response will _only_ contain `dx` and `dy`, stored as either 2xN or Nx2 (depending on format chosen)
    """

    body = await request.body()
    points = len(body) // (3 * 4)  # 3 x float
    if format == BinaryFormats.array_3xN:
        locs = np.frombuffer(body, dtype=np.float32).reshape(3, points).swapaxes(0, 1)
    elif format == BinaryFormats.array_Nx3:
        locs = np.frombuffer(body, dtype=np.float32).reshape(points, 3)
    else:
        raise Exception("Unexpected format: {}".format(format))

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(
            status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations),
        )

    # scale & adjust locations
    transformed = await run_in_threadpool(map_points, dataset.value, scale, locs)

    data = np.zeros(dtype=np.float32, shape=(2, points), order="C")
    data[0, :] = transformed["dx"]
    data[1, :] = transformed["dy"]
    if format == BinaryFormats.array_Nx3:
        data = data.swapaxes(0, 1)

    return Response(content=data.tobytes(), media_type="application/octet-stream")


@app.post(
    "/query/dataset/{dataset}/s/{scale}/values_binary/format/{format}",
    response_model=None,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "Binary encoding of output array.",
        }
    },
    tags=["query"],
)
async def query_values_binary(
    dataset: DataSetName, scale: int, format: BinaryFormats, request: Request
):
    """Query a dataset for values at a point(s)

    The response will _only_ contain the value(s) at the coordinates requested.
    The datatype returned will be of the type referenced in */info/*.
    """

    body = await request.body()
    points = len(body) // (3 * 4)  # 3 x float
    if format == BinaryFormats.array_3xN:
        locs = np.frombuffer(body, dtype=np.float32).reshape(3, points).swapaxes(0, 1)
    elif format == BinaryFormats.array_Nx3:
        locs = np.frombuffer(body, dtype=np.float32).reshape(points, 3)
    else:
        raise Exception("Unexpected format: {}".format(format))

    if locs.shape[0] > config.MaxLocations:
        raise HTTPException(
            status_code=400,
            detail="Max number of locations ({}) exceeded".format(config.MaxLocations),
        )

    data = await run_in_threadpool(query_points, dataset.value, scale, locs)

    if format == BinaryFormats.array_Nx3:
        data = data.swapaxes(0, 1)

    return Response(content=data.tobytes(), media_type="application/octet-stream")


# Datasets we currently allow to be mapped between
ALLOWED_DATASETS = ["flywire", "aedes", "zhengCA3", "fanc", "banc"]
DatasetName = Enum("DatasetName", dict(zip(ALLOWED_DATASETS, ALLOWED_DATASETS)))
DATASET_FUNCS = {
    "flywire": get_flywire_segmentation_properties,
    "aedes": get_aedes_segmentation_properties,
    "zhengCA3": get_zhengCA3_segmentation_properties,
    "fanc": get_fanc_segmentation_properties,
    "banc": get_banc_segmentation_properties,
}

@app.get(
    "/segmentation_annotations/datasets",
    response_model=List[str],
)
async def segmentation_datasets():
    """Return list of available segmentation annotation datasets."""
    return ALLOWED_DATASETS


@app.get(
    "/segmentation_annotations/{dataset}/{version}/{labels}/info",
    tags=["annotations"],
    response_model=Dict,
)
@app.get(
    "/segmentation_annotations/{dataset}/{version}/{labels}/{tags}/info",
    tags=["annotations"],
    response_model=Dict,
)
async def segmentation_annotations(
    dataset: DatasetName,
    version: str,
    labels: str,
    request: Request,
    tags: str | None = None,
):
    """Generate segmentation properties from FlyTable."""
    if dataset.value not in DATASET_FUNCS:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset} not found. Available datasets: {ALLOWED_DATASETS}",
        )
    dataset_func = DATASET_FUNCS[dataset.value]
    try:
        return dataset_func(mat_version=version, labels=labels, tags=tags)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ---------------------------------------------------------------------------
# Sparse voxels (supervoxel -> RLE index)
#
# A segment's voxels are served by unioning precomputed per-supervoxel RLE
# fragments, never by reading the segmentation. The dense read happens once,
# offline, when the index is built.
# ---------------------------------------------------------------------------

# Two backends answer the same questions. "live" reads the locally-stored
# segmentation and sparsifies per request; "index" reads precomputed fragments.
# Which one a dataset uses is a deployment detail, so it is not in the URL --
# the response is identical either way.
SPARSEVOL_LIVE_DATASETS = [
    name
    for name, info in config.DATASOURCES.items()
    if "sparsevol" in info.get("services", [])
]
_collisions = set(config.SPARSEVOL_DATASOURCES) & set(SPARSEVOL_LIVE_DATASETS)
if _collisions:
    # Otherwise one silently shadows the other and requests go to a backend
    # the operator did not choose.
    raise RuntimeError(
        "Dataset name(s) {} are configured both as a sparsevol index and as a "
        "live sparsevol source. Names must be unique across the two.".format(
            sorted(_collisions)
        )
    )

SPARSEVOL_DATASETS = list(config.SPARSEVOL_DATASOURCES) + SPARSEVOL_LIVE_DATASETS

SparseVolDataSet = Enum(
    "SparseVolDataSet", dict(zip(SPARSEVOL_DATASETS, SPARSEVOL_DATASETS))
)


def sparsevol_backend(dataset):
    """The module answering for this dataset. Both expose the same two calls."""
    return sparsevol_live if dataset in SPARSEVOL_LIVE_DATASETS else sparsevol


class SparseVolFormat(str, Enum):
    rle = "rle"  # binary (M, 4) int32: x, y, z, length
    npy = "npy"  # the same array as a .npy file
    json = "json"  # runs plus stats as JSON
    coords = "coords"  # binary (N, 3) int32, one row per voxel


class SuperVoxelList(BaseModel):
    # Strings as well as ints: supervoxel IDs are uint64 and lose precision in
    # a JavaScript client the moment they exceed 2^53.
    supervoxels: List[int | str]


SPARSEVOL_RESPONSES = {
    200: {
        "content": {"application/octet-stream": {}},
        "description": (
            "Run-length encoded voxels. Response headers carry what the "
            "request cost: X-Sparsevol-Runs, -Voxels, -Supervoxels, "
            "-Fragments, -Missing, -Chunks, -Reads, -Bytes, -Seconds."
        ),
    }
}


def stats_headers(stats):
    """Turn a backend's stats into headers.

    Derived from the stats object rather than listed here, because the two
    backends account for different things -- one counts bytes fetched from a
    store, the other counts voxels read and discarded -- and the response
    should report whichever actually happened.
    """
    headers = {}
    for key, value in stats.as_dict().items():
        # n_range_reads -> X-Sparsevol-Range-Reads
        name = key[2:] if key.startswith("n_") else key
        name = "-".join(part.capitalize() for part in name.split("_"))
        headers["X-Sparsevol-" + name] = str(value)
    return headers


def sparsevol_response(runs, stats, fmt, request):
    """Serialize runs, preferring formats that keep the response small."""
    headers = stats_headers(stats)

    if fmt == SparseVolFormat.json:
        return ORJSONResponse(
            {"runs": runs.tolist(), "stats": stats.as_dict()}, headers=headers
        )

    if fmt == SparseVolFormat.coords:
        # Every voxel spelled out. Offered because some clients want it, but it
        # is the largest thing this service can return -- prefer 'rle'.
        array = decode_rle(runs).astype("<i4")
    else:
        array = runs.astype("<i4")

    if fmt == SparseVolFormat.npy:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        content = buffer.getvalue()
    else:
        content = np.ascontiguousarray(array).tobytes()

    # Runs are sorted by (z, y, x), so three of the four columns barely change
    # down the array and gzip does well on them. Worth the CPU on a response
    # that can run to megabytes.
    if "gzip" in request.headers.get("accept-encoding", ""):
        content = gzip.compress(content, compresslevel=6)
        headers["Content-Encoding"] = "gzip"

    return Response(
        content=content, media_type="application/octet-stream", headers=headers
    )


@app.get("/sparsevol/datasets", tags=["sparsevol"], response_model=Dict)
async def sparsevol_datasets():
    """Datasets that can return sparse voxels, and the scales they cover.

    `mode` is how the answer is produced: `live` sparsifies the locally-stored
    segmentation per request, `index` reads precomputed fragments. It does not
    change the request or the response.
    """
    datasets = {}
    for name in SPARSEVOL_DATASETS:
        live = name in SPARSEVOL_LIVE_DATASETS
        info = config.DATASOURCES[name] if live else config.SPARSEVOL_DATASOURCES[name]
        datasets[name] = {
            "mode": "live" if live else "index",
            **{
                field: info.get(field)
                for field in ("description", "scales", "voxel_size")
            },
        }
    return datasets


@app.get(
    "/sparsevol/dataset/{dataset}/s/{scale}/root/{root_id}",
    response_model=None,
    responses=SPARSEVOL_RESPONSES,
    tags=["sparsevol"],
)
async def sparsevol_root(
    dataset: SparseVolDataSet,
    scale: int,
    root_id: int,
    request: Request,
    fmt: SparseVolFormat = SparseVolFormat.rle,
):
    """Sparse voxels for a root ID.

    Asks the chunkedgraph which chunks the neuron occupies and which
    supervoxels belong to it, then either reads and masks those chunks from the
    local segmentation or unions precomputed fragments, depending on the
    dataset. Either way the segmentation itself never crosses the wire.

    Coordinates are voxels at the requested scale; multiply by the dataset's
    voxel size (scaled by 2^scale in x and y) for nanometres.

    Root IDs are immutable -- an edit mints a new one -- so a result may be
    cached indefinitely against `(root_id, scale)`.
    """
    runs, stats = await run_in_threadpool(
        sparsevol_backend(dataset.value).root_to_runs, dataset.value, scale, root_id
    )
    return sparsevol_response(runs, stats, fmt, request)


@app.post(
    "/sparsevol/dataset/{dataset}/s/{scale}/supervoxels",
    response_model=None,
    responses=SPARSEVOL_RESPONSES,
    tags=["sparsevol"],
)
async def sparsevol_supervoxels(
    dataset: SparseVolDataSet,
    scale: int,
    data: SuperVoxelList,
    request: Request,
    fmt: SparseVolFormat = SparseVolFormat.rle,
):
    """Sparse voxels for an explicit list of supervoxel IDs.

    The chunk a supervoxel lives in is packed into its ID, so this path never
    calls the chunkedgraph at all -- it is the root endpoint minus the manifest
    lookup. Use it when you already hold a manifest, want part of a neuron, or
    are asking about supervoxels the graph would not group together.

    Supervoxel IDs that resolve to nothing are skipped and counted rather than
    failing the request.
    """
    try:
        sv_ids = [int(sv) for sv in data.supervoxels]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Supervoxel IDs must be integers")

    # Anything outside uint64 is not a label. Caught here because numpy would
    # otherwise either raise deep in the request or wrap a negative ID round to
    # a plausible-looking one and report it as merely missing.
    if any(sv < 0 or sv >= 2**64 for sv in sv_ids):
        raise HTTPException(
            status_code=400, detail="Supervoxel IDs must fit in an unsigned 64-bit int"
        )

    runs, stats = await run_in_threadpool(
        sparsevol_backend(dataset.value).supervoxels_to_runs, dataset.value, scale, sv_ids
    )
    return sparsevol_response(runs, stats, fmt, request)


# Catch all for all other paths for debugging
# @app.api_route("/{path_name:path}", methods=["GET"])
# async def catch_all(request: Request, path_name: str):
#     return {"request_method": request.method, "path_name": path_name}
