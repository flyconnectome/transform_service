import re
import os
import threading

import pandas as pd
import nglscenes as ngl
import seaserpent as ss

FLYWIRE_MAT_VERSION_TO_COL = {"630": "root_630", "783": "root_783", "live": "root_id"}
AEDES_MAT_VERSION_TO_COL = {"live": "root_id"}
ZHENGCA3_MAT_VERSION_TO_COL = {"live": "root_id"}
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
    info, optic = get_flywire_tables()

    return _get_segmentation_properties(
        tables=(info, optic),
        labels=labels,
        tags=tags,
        id_col=FLYWIRE_MAT_VERSION_TO_COL[mat_version],
    )


def get_aedes_segmentation_properties(mat_version, labels, tags):
    """Compile Neuroglancer segment properties for Aedes neurons.

    Args:
        mat_version (str): The version of the Aedes segmentation data.
        labels (str): A string determining which labels to include and how to format.
        tags (str): A string determining which tags to include and how to format.

    Returns:
        dict: A dict of dictionaries containing segment properties.

    """
    if mat_version not in AEDES_MAT_VERSION_TO_COL:
        raise ValueError(
            f"Invalid mat_version: {mat_version}. Must be one of {AEDES_MAT_VERSION_TO_COL}."
        )

    # Get the table
    aedes = get_aedes_table()

    return _get_segmentation_properties(
        tables=(aedes,),
        labels=labels,
        tags=tags,
        id_col=AEDES_MAT_VERSION_TO_COL[mat_version],
    )

    
def get_zhengCA3_segmentation_properties(mat_version, labels, tags):
    """Compile Neuroglancer segment properties for ZhengCA3 neurons.

    Args:
        mat_version (str): The version of the ZhengCA3 segmentation data.
        labels (str): A string determining which labels to include and how to format.
        tags (str): A string determining which tags to include and how to format.

    Returns:
        dict: A dict of dictionaries containing segment properties.

    """
    if mat_version not in ZHENGCA3_MAT_VERSION_TO_COL:
        raise ValueError(
            f"Invalid mat_version: {mat_version}. Must be one of {ZHENGCA3_MAT_VERSION_TO_COL}."
        )

    # Get the tables
    zhengCA3 = get_zhengCA3_table()

    return _get_segmentation_properties(
        tables=(zhengCA3),
        labels=labels,
        tags=tags,
        id_col=ZHENGCA3_MAT_VERSION_TO_COL[mat_version],
    )


def _get_segmentation_properties(tables, labels, tags, id_col):
    """Compile Neuroglancer segment properties from the provided tables.

    Args:
        tables (tuple): A tuple containing one or more tables to concatenate.
        labels (str): A string determining which labels to include and how to format.
        tags (str): A string determining which tags to include and how to format.
        id_col (str): The column name to use as the root ID.

    Returns:
        dict: A dict of dictionaries containing segment properties.

    """
    if isinstance(tables, ss.Table):
        # If a single table is provided, convert it to a tuple
        tables = (tables,)

    # Parse labels into columns
    available_columns = tables[0].columns
    cols_to_fetch = {id_col}
    backfills = []
    if "{" in labels:
        for label in re.findall(r"\{(.*?)\}", labels):
            if "<" in label:
                backfills.append(label)
                for la in label.split("<"):
                    cols_to_fetch.add(la.strip())
            else:
                cols_to_fetch.add(label.strip())
    else:
        if "<" in labels:
            backfills.append(labels)
            for la in labels.split("<"):
                cols_to_fetch.add(la.strip())
        else:
            cols_to_fetch.add(labels)
        labels = "{" + labels + "}"

    if tags:
        for tag in tags.split(","):
            cols_to_fetch.add(tag.strip())

    for col in cols_to_fetch:
        if col not in available_columns:
            raise ValueError(
                f"Invalid label column: '{col}' does not exist in table(s). Available columns: {available_columns}"
            )

    # Fetch the data
    if "status" in available_columns:
        data = pd.concat(
            [t.loc[~t.status.isin(BAD_STATUS), list(cols_to_fetch)] for t in tables],
            axis=0,
        ).rename(columns={id_col: "root_id"})
    else:
        data = pd.concat(
            [t[list(cols_to_fetch)] for t in tables], axis=0
        ).rename(columns={id_col: "root_id"})

    # Some clean-ups
    data = data.drop_duplicates(subset="root_id")

    # Generate backfills if needed
    for bf in backfills:
        data[bf] = pd.Series(None, dtype="string")
        for col in bf.split("<"):
            col = col.strip()
            data[bf] = data[bf].fillna(data[col])

    # Compile data into labels
    labels_compiled = data.apply(labels.format_map, axis=1)
    labels_compiled.name = "labels"

    # Generate the properties
    props = ngl.SegmentProperties(data.root_id.values)

    props.add_property(labels_compiled, type="label")

    if tags:
        props.add_property(
            # Expects a list of list
            data[[t.strip() for t in tags.split(",")]].values.astype(str).tolist(),
            type="tags",
            name="tags",
        )

    return props.as_dict()


def get_flywire_tables():
    """Return seaserpent.Tables object connected to the FlyWire (info + optic lobes) tables.

    The tables are cached, so this function will return the same object if called again
    from the same thread with the same arguments.
    """
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        info, optic = TABLES[("flywire", thread_id, pid)]
    except KeyError:
        info = ss.Table("info", "main")
        optic = ss.Table("optic", "optic_lobes")

        TABLES[("flywire", thread_id, pid)] = (info, optic)

    return info, optic


def get_aedes_table():
    """Return seaserpent.Tables object connected to the Aedes table.

    The tables are cached, so this function will return the same object if called again
    from the same thread with the same arguments.
    """
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        aedes = TABLES[("aedes", thread_id, pid)]
    except KeyError:
        aedes = ss.Table("aedes_main", "aedes")
        TABLES[("aedes", thread_id, pid)] = aedes

    return aedes

def get_zhengCA3_table():
    """Return seaserpent.Tables object connected to the ZhengCA3 table.

    The tables are cached, so this function will return the same object if called again
    from the same thread with the same arguments.
    """
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        zhengCA3 = TABLES[("zhengCA3", thread_id, pid)]
    except KeyError:
        zhengCA3 = ss.Table("annotations", "zheng_ca3")
        TABLES[("zhengCA3", thread_id, pid)] = zhengCA3

    return zhengCA3
