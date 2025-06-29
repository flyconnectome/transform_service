import numpy as np
from fastapi import HTTPException

from . import config
from . import process
from . import datasource


def query_points(dataset, scale, locs):
    """Query a dataset for the points.
    Input:  [n,3] numpy array representing n (x,y,z) points
    Output: [n,5] numpy array representing n (new_x, new_y, new_z, new_dx, new_dy)
    """
    info = datasource.get_datasource_info(dataset)
    n5 = datasource.get_datastore(dataset, scale)
    downsample = datasource.get_datastore_downsample(dataset, scale)

    # TODO: There is probably a better way to get this from tensorstore?
    if info["type"] == "neuroglancer_precomputed":
        blocksize = (
            np.asarray(n5.spec().to_json()["scale_metadata"]["chunk_size"])
            * config.CHUNK_MULTIPLIER
        )
    elif info["type"] in ["zarr", "zarr-nested"]:
        blocksize = (
            np.array(n5.spec().to_json()["metadata"]["chunks"])[0:3]
            * config.CHUNK_MULTIPLIER
        )

    query_points = np.empty_like(locs)
    query_points[:, 0] = locs[:, 0] // downsample[0]
    query_points[:, 1] = locs[:, 1] // downsample[1]
    query_points[:, 2] = locs[:, 2] // downsample[2]

    bad_points = (
        (query_points < n5.domain.inclusive_min[0:3])
        | (query_points > n5.domain.inclusive_max[0:3])
    ).any(axis=1)
    query_points[bad_points] = np.NaN

    error_value = np.NaN
    if np.issubdtype(np.dtype(info["dtype"]), np.integer):
        # Return 0 for integers [otherwise, np.NaN maps to MAX_VALUE
        error_value = 0

    if bad_points.all():
        # No valid points. The binning code will otherwise fail.
        field = np.full(
            (query_points.shape[0], info["width"]), error_value, dtype=info["dtype"]
        )
    else:
        field = process.get_multiple_ids(
            query_points,
            n5,
            max_workers=config.MaxWorkers,
            blocksize=blocksize,
            error_value=error_value,
            dtype=info["dtype"],
        )
    return field


def map_points(dataset, scale, locs):
    """Do the work for mapping data.
    Input:  [n,3] numpy array representing n (x,y,z) points
    Output: [n,5] numpy array representing n (new_x, new_y, new_z, new_dx, new_dy)
    """
    info = datasource.get_datasource_info(dataset)
    if "transform" not in info["services"]:
        raise HTTPException(
            status_code=400, detail="This dataset does not provide transform services."
        )
    field = query_points(dataset, scale, locs)
    results = np.zeros(
        locs.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("dx", "<f4"), ("dy", "<f4")],
    )

    # From Tommy Macrina:
    #   We store the vectors as fixed-point int16 with two bits for the decimal.
    #   Even if a vector is stored at a different MIP level (e.g. these are stored at MIP2),
    #   the vectors represent MIP0 displacements, so there's no further scaling required.

    results["dx"] = field[:, 1] / 4.0
    results["dy"] = field[:, 0] / 4.0
    results["x"] = locs[:, 0] + results["dx"]
    results["y"] = locs[:, 1] + results["dy"]
    results["z"] = locs[:, 2]

    return results
