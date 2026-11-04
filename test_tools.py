"""Tests for the repacking tools in tools/: the voxels must survive unchanged."""

import gzip
import json
import os
import sys
import threading

import numpy as np
import pytest
import tensorstore as ts
import tqdm
import urllib3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

import fetch_sharded  # noqa: E402
import to_sharded  # noqa: E402

# Tensorstore aborts the whole process if it forks after tensorstore has been
# used, and these tests read volumes with it before and after running a tool.
# The tools never use tensorstore themselves, so a real run forks freely; here
# they run with one worker, which stays in-process, and tqdm -- which
# CloudVolume's shard builder uses for progress -- gets a thread lock instead of
# the multiprocessing one it would otherwise start a helper process for.
tqdm.tqdm.set_lock(threading.RLock())

# Uneven on every axis and offset off the chunk grid, like CA3: the clipped
# chunks on the far edges are the ones most likely to go wrong.
SHAPE = (150, 140, 70)
OFFSET = [17, 29, 5]
CHUNK = [64, 64, 32]


def scale(key, resolution, size, offset):
    return {
        "key": key, "size": list(size), "resolution": resolution, "voxel_offset": offset,
        "chunk_size": CHUNK, "encoding": "compressed_segmentation",
        "compressed_segmentation_block_size": [8, 8, 8],
    }


def open_scale(path, index):
    return ts.open({
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": str(path)},
        "scale_index": index,
    }).result()


@pytest.fixture
def volume(tmp_path):
    """An unsharded two-scale segmentation; scale 1 holds blocky labels."""
    rng = np.random.default_rng(0)
    coarse = rng.integers(1, 2**40, size=(38, 35, 18), dtype=np.uint64)
    data = np.repeat(np.repeat(np.repeat(coarse, 4, 0), 4, 1), 4, 2)
    data = np.ascontiguousarray(data[: SHAPE[0], : SHAPE[1], : SHAPE[2]])

    path = tmp_path / "raw"
    ts.open({
        "driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": str(path)},
        "multiscale_metadata": {"type": "segmentation", "data_type": "uint64", "num_channels": 1},
        "scale_metadata": scale("s0", [8, 8, 45], [300, 280, 70], [34, 58, 5]), "create": True,
    }).result()
    ts.open({
        "driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": str(path)},
        "scale_metadata": scale("s1", [16, 16, 45], SHAPE, OFFSET), "create": True,
    }).result()
    info = json.loads((path / "info").read_text())
    info["chunks_start_at_voxel_offset"] = True
    (path / "info").write_text(json.dumps(info))

    store = open_scale(path, 1)
    store[..., 0] = data
    return path, data


def test_repacked_scale_reads_back_identically(volume, tmp_path):
    src, data = volume
    dst = tmp_path / "sharded"
    to_sharded.main([str(src), str(dst), "s1", "--workers", "1"])

    out = open_scale(dst, 1)
    assert out.domain == open_scale(src, 1).domain
    # The chunk size is what the service tiles its reads by, so it must not move.
    assert out.chunk_layout.read_chunk.shape == (*CHUNK, 1)
    assert np.array_equal(out.read().result()[..., 0], data)

    files = os.listdir(dst / "s1")
    assert files and all(f.endswith(".shard") for f in files)


def test_other_scales_keep_their_place_in_the_info(volume, tmp_path):
    src, _ = volume
    dst = tmp_path / "sharded"
    to_sharded.main([str(src), str(dst), "s1", "--workers", "1"])

    before = json.loads((src / "info").read_text())
    after = json.loads((dst / "info").read_text())
    # The service opens a scale by position, so order and the untouched scale
    # must be exactly as they were.
    assert [s["key"] for s in after["scales"]] == [s["key"] for s in before["scales"]]
    assert after["scales"][0] == before["scales"][0]
    assert after["scales"][1]["sharding"]["data_encoding"] == "gzip"


def test_a_rerun_skips_finished_shards(volume, tmp_path, capsys):
    src, data = volume
    dst = tmp_path / "sharded"
    to_sharded.main([str(src), str(dst), "s1", "--workers", "1"])
    capsys.readouterr()

    to_sharded.main([str(src), str(dst), "s1", "--workers", "1"])
    assert ", 0 to build" in capsys.readouterr().out
    assert np.array_equal(open_scale(dst, 1).read().result()[..., 0], data)


def test_a_scale_sharded_earlier_keeps_its_settings(volume, tmp_path):
    src, _ = volume
    dst = tmp_path / "sharded"
    dst.mkdir()
    info = json.loads((src / "info").read_text())
    earlier = dict(to_sharded.SHARDING, shard_bits=1)
    info["scales"][0]["sharding"] = earlier
    (dst / "info").write_text(json.dumps(info))

    to_sharded.main([str(src), str(dst), "s1", "--workers", "1"])

    after = json.loads((dst / "info").read_text())
    assert after["scales"][0]["sharding"] == earlier
    assert "sharding" in after["scales"][1]


def test_the_source_is_never_the_destination(volume):
    src, _ = volume
    with pytest.raises(SystemExit):
        to_sharded.main([str(src), str(src), "s1"])


class FakeBucket:
    """Serves the raw volume's chunk files the way GCS does with Accept-Encoding: gzip.

    Drops the connection on the first read of one chunk, and has no copy of
    another, so both the retry and the missing-chunk path are exercised.
    """

    def __init__(self, scale_dir, drop, absent):
        self.scale_dir = scale_dir
        self.drop, self.absent = drop, absent
        self.dropped = False

    def get(self, url, **kwargs):
        name = url.rsplit("/", 1)[1]
        bucket = self

        class Response:
            headers = {}

            class raw:
                @staticmethod
                def read(decode_content=True):
                    if name == bucket.drop and not bucket.dropped:
                        bucket.dropped = True
                        raise urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")
                    return Response.body

        if name == self.absent or not os.path.exists(os.path.join(self.scale_dir, name)):
            Response.status_code = 404
            return Response
        with open(os.path.join(self.scale_dir, name), "rb") as f:
            Response.body = gzip.compress(f.read(), mtime=0)
        Response.status_code = 200
        Response.headers = {"Content-Encoding": "gzip", "Content-Length": str(len(Response.body))}
        return Response


def test_fetching_from_the_bucket_matches_the_volume(volume, tmp_path, monkeypatch):
    src, data = volume
    dst = tmp_path / "fetched"
    dst.mkdir()
    (dst / "info").write_text((src / "info").read_text())

    names = sorted(os.listdir(src / "s1"))
    drop, absent = names[0], names[-1]
    bucket = FakeBucket(str(src / "s1"), drop=drop, absent=absent)
    monkeypatch.setattr(fetch_sharded, "_session", bucket)
    monkeypatch.setattr(fetch_sharded.time, "sleep", lambda seconds: None)

    # Built in-process rather than through main()'s process pool, so the fake
    # bucket is the one every download sees.
    info = json.loads((dst / "info").read_text())
    sharding, groups, n_chunks = fetch_sharded.plan(info, "s1")
    assert n_chunks == len(names)
    out_dir = dst / "s1"
    out_dir.mkdir()
    fetched = missing = 0
    for members in groups.values():
        c, m, _ = fetch_sharded.build_one(("https://bucket/s1/", str(out_dir), sharding, members, 2))
        fetched, missing = fetched + c, missing + m
    fetch_sharded.write_info(str(dst / "info"), info, "s1", sharding)

    assert bucket.dropped, "the dropped connection was never exercised"
    assert (fetched, missing) == (len(names) - 1, 1)

    # The absent chunk reads as empty; every other voxel is the original.
    want = data.copy()
    lo = [int(v) for v in absent.replace("_", "-").split("-")[::2]]
    hi = [int(v) for v in absent.replace("_", "-").split("-")[1::2]]
    want[tuple(slice(a - o, b - o) for a, b, o in zip(lo, hi, OFFSET))] = 0
    assert np.array_equal(open_scale(dst, 1).read().result()[..., 0], want)
