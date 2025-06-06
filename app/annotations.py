import re
import os
import threading

import pandas as pd
import nglscenes as ngl
import seaserpent as ss

FLYWIRE_MAT_VERSION_TO_COL = {"630": "root_630", "783": "root_783", "live": "root_id"}
BAD_STATUS = ("duplicate", "bad_nucleus")

TABLES = {}


def get_flywire_segmentation_properties(mat_version, labels, tags):
    """Compile Neuroglancer segment properties for FlyWire neurons.

    Args:
        mat_version (str): The version of the FlyWire segmentation data.
        labels (str): A string determining which labels to include and how to format.
        tags (str): A string determining which tags to include and how to format.

    Returns:
        dict: A dict of dictionaries containing segment properties.

    """
    if mat_version not in FLYWIRE_MAT_VERSION_TO_COL:
        raise ValueError(
            f"Invalid mat_version: {mat_version}. Must be one of {FLYWIRE_MAT_VERSION_TO_COL}."
        )

    # Get the tables
    info, optic = get_tables()

    # Parse labels into columns
    available_columns = info.columns
    cols_to_fetch = {FLYWIRE_MAT_VERSION_TO_COL[mat_version]}
    if "{" in labels:
        for label in re.findall(r"\{(.*?)\}", labels):
            cols_to_fetch.add(label.strip())
    else:
        cols_to_fetch.add(labels)
        labels = f"{labels}"

    if tags:
        for tag in tags.split(","):
            cols_to_fetch.add(tag.strip())

    for col in cols_to_fetch:
        if col not in available_columns:
            raise ValueError(
                f"Invalid label column: {col}. Available columns: {available_columns}"
            )

    # Fetch the data
    data = pd.concat(
        [
            info.loc[~info.status.isin(BAD_STATUS), list(cols_to_fetch)],
            optic.loc[~optic.status.isin(BAD_STATUS), list(cols_to_fetch)],
        ],
        axis=0,
    ).rename(columns={FLYWIRE_MAT_VERSION_TO_COL[mat_version]: "root_id"})

    # Some clean-ups
    data = data.drop_duplicates(subset="root_id")

    # Compile data into labels
    labels_compiled = data.apply(labels.format_map, axis=1)
    labels_compiled.name = "labels"

    # Generate the properties
    props = ngl.SegmentProperties(data.root_id.values)

    props.add_property(labels_compiled, type="label")

    if tags:
        props.add_property(
            # Expects a list of list
            data[[t.strip() for t in tags.split(",")]].values.tolist(),
            type="tags",
            name="tags",
        )

    return props.as_dict()


def get_tables():
    """Return seaserpent.Tables object connected to the info and optic lobes table.

    The tables are cached, so this function will return the same object if called again
    from the same thread with the same arguments.
    """
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        info, optic = TABLES[(thread_id, pid)]
    except KeyError:
        info = ss.Table("info", "main")
        optic = ss.Table("optic", "optic_lobes")

        TABLES[(thread_id, pid)] = (info, optic)

    return info, optic
