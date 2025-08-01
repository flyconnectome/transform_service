#!/usr/bin/env python3
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
from .query import map_points, query_points
from .annotations import (
    get_flywire_segmentation_properties,
    get_aedes_segmentation_properties,
    get_zhengCA3_segmentation_properties
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
        for field in ("scales", "voxel_size", "description"):
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
ALLOWED_DATASETS = ["flywire", "aedes", "zhengCA3"]
DatasetName = Enum("DatasetName", dict(zip(ALLOWED_DATASETS, ALLOWED_DATASETS)))


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
    if dataset == DatasetName.flywire:
        try:
            return get_flywire_segmentation_properties(
                mat_version=version, labels=labels, tags=tags
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
    elif dataset == DatasetName.aedes:
        try:
            return get_aedes_segmentation_properties(
                mat_version=version, labels=labels, tags=tags
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
    elif dataset == DatasetName.zhengCA3:
        try:
            return get_zhengCA3_segmentation_properties(
                mat_version=version, labels=labels, tags=tags
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset} not found. Available datasets: {ALLOWED_DATASETS}",
        )



# Catch all for all other paths for debugging
# @app.api_route("/{path_name:path}", methods=["GET"])
# async def catch_all(request: Request, path_name: str):
#     return {"request_method": request.method, "path_name": path_name}
