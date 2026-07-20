"""A bounded cache of layer-2 node -> RLE runs.

The live sparsevol service reads and masks real chunks on every request. That
is affordable but not cheap -- a large aedes neuron is ~180 chunks and ~40
seconds -- and almost all of that work is repeated verbatim the next time
anyone asks about the same neuron, or about any neuron sharing its L2 nodes.

**Why layer 2 is the key.** A supervoxel ID tells you its chunk but not which
L2 node owns it, and a root ID changes every time somebody proofreads. L2 node
IDs sit in between: stable enough to be worth caching, fine-grained enough that
an edit invalidates almost nothing. A merge spanning two chunks adds an edge
*above* layer 2 and mints no new L2 node at all; only within-chunk merges and
splits recompute one, and only for the chunks the cut touches. So entries are
written once and read forever, and the handful that do go stale are simply
never asked for again.

**Why it fails loudly when full.** Rather than evicting, the cache stops
writing and starts refusing every request. An LRU would quietly thrash at the
ceiling and look, from the outside, exactly like a cache that is working. The
thing worth noticing here is the cache reaching its ceiling at all -- that is a
capacity decision for a person, not something to paper over. See
``config.L2CacheMaxBytes`` for how to recover.

Storage is SQLite because the writers are several gunicorn worker processes
appending concurrently, which rules out the append-only fragment files in
:mod:`app.sparsevol` -- those assume a single offline builder. WAL mode gives
concurrent readers alongside a writer, and the counters that enforce the cap
update in the same transaction as the rows they account for.
"""

import os
import sqlite3
import threading

import numpy as np
from fastapi import HTTPException

from . import config
from .rle import pack_runs, rle_voxel_count, unpack_runs

# SQLite's parameter limit is 999 on older builds. Lookups are chunked to stay
# well inside it rather than depending on the version we happen to be linked to.
_MAX_PARAMS = 500

# Charged on top of each blob so the accounted size tracks the file on disk
# rather than just the payload: a row carries a 24-byte key, its primary-key
# index entry and per-page slack. Approximate on purpose -- `file_bytes` in
# stats() reports what actually landed on disk.
_ROW_OVERHEAD = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fragments (
    dataset  TEXT    NOT NULL,
    scale    INTEGER NOT NULL,
    l2_id    INTEGER NOT NULL,
    n_runs   INTEGER NOT NULL,
    n_voxels INTEGER NOT NULL,
    blob     BLOB    NOT NULL,
    PRIMARY KEY (dataset, scale, l2_id)
);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""


def to_signed(value):
    """Reinterpret a uint64 label as the int64 SQLite stores.

    SQLite has no unsigned integer type. Graphene labels for layer 2 stay
    inside int64 today, but the layer lives in the top bits and a high enough
    layer would overflow, so the bit pattern is preserved rather than the value.
    """
    value = int(value)
    if value < 0 or value >= 2**64:
        raise ValueError("label {} is not a uint64".format(value))
    return value - 2**64 if value >= 2**63 else value


def to_unsigned(value):
    """Inverse of :func:`to_signed`."""
    value = int(value)
    return value + 2**64 if value < 0 else value


class CacheFull(Exception):
    """Raised by the offline builder; the service raises HTTP 503 instead."""


class L2Cache:
    """L2 node -> packed runs, keyed by ``(dataset, scale, l2_id)``.

    Entries are immutable: the same L2 node at the same scale always has the
    same voxels, so a write conflict between two workers computing it at once
    is a no-op rather than something to resolve.
    """

    def __init__(self, path, max_bytes=None, max_keys=None, timeout=30.0):
        self.path = path
        self.max_bytes = int(config.L2CacheMaxBytes if max_bytes is None else max_bytes)
        self.max_keys = int(config.L2CacheMaxKeys if max_keys is None else max_keys)
        self.timeout = timeout
        # One connection per thread: sqlite3 objects are not safe to share, and
        # the service answers from a threadpool.
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    # --- plumbing ---------------------------------------------------------

    def _ensure_schema(self):
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            directory = os.path.dirname(os.path.abspath(self.path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=self.timeout)
            try:
                # Set here rather than per connection: switching journal mode
                # wants a lock no other connection is holding, so doing it on
                # every open turns a burst of concurrent requests into
                # "database is locked". It is a property of the file and
                # persists, so setting it once at creation is enough.
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError:
                    # Another process got there first and is already using it.
                    pass
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()
            self._schema_ready = True

    def connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        self._ensure_schema()
        # isolation_level=None means we say where transactions begin and end,
        # which matters for put_many: the rows and the counters accounting for
        # them have to land together or the cap drifts.
        conn = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout={}".format(int(self.timeout * 1000)))
        self._local.conn = conn
        return conn

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # --- accounting -------------------------------------------------------

    def counters(self):
        rows = self.connection().execute("SELECT name, value FROM counters").fetchall()
        counts = dict(rows)
        return int(counts.get("n_keys", 0)), int(counts.get("n_bytes", 0))

    def stats(self):
        n_keys, n_bytes = self.counters()
        try:
            file_bytes = os.path.getsize(self.path)
        except OSError:
            file_bytes = 0
        return {
            "path": self.path,
            "n_keys": n_keys,
            "n_bytes": n_bytes,
            "file_bytes": file_bytes,
            "max_keys": self.max_keys,
            "max_bytes": self.max_bytes,
            "fraction_keys": round(n_keys / self.max_keys, 4) if self.max_keys else 0.0,
            "fraction_bytes": (
                round(n_bytes / self.max_bytes, 4) if self.max_bytes else 0.0
            ),
            "full": self.is_full(),
        }

    def is_full(self):
        n_keys, n_bytes = self.counters()
        return (self.max_keys and n_keys >= self.max_keys) or (
            self.max_bytes and n_bytes >= self.max_bytes
        )

    def check_health(self):
        """Refuse the request outright if the cache has hit its ceiling.

        Called at the top of every sparse volume request, including ones that
        would have been served entirely from cache. That is the point: a full
        cache should be impossible to keep using without noticing.
        """
        n_keys, n_bytes = self.counters()
        if not self.is_full():
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "The layer-2 sparse volume cache is full ({:,} keys / {:.1f} GB, "
                "limits {:,} keys / {:.1f} GB) and is refusing requests so this "
                "is not missed. Recover by raising config.L2CacheMaxBytes or "
                "L2CacheMaxKeys, clearing it with "
                "`python -m app.l2cache_warm --clear`, or setting "
                "config.L2CacheEnabled = False to compute every request live."
            ).format(
                n_keys,
                n_bytes / 1024**3,
                self.max_keys,
                self.max_bytes / 1024**3,
            ),
        )

    # --- reads and writes -------------------------------------------------

    def get_many(self, dataset, scale, l2_ids):
        """Cached runs for whatever subset of ``l2_ids`` is present.

        Returns ``{uint64 l2_id: (M, 4) runs}``. Absent keys are simply absent;
        the caller computes those and hands them back to :meth:`put_many`.
        """
        ids = [int(i) for i in np.asarray(l2_ids, dtype=np.uint64).ravel().tolist()]
        if not ids:
            return {}

        conn = self.connection()
        found = {}
        for start in range(0, len(ids), _MAX_PARAMS):
            batch = ids[start : start + _MAX_PARAMS]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                "SELECT l2_id, blob FROM fragments "
                "WHERE dataset = ? AND scale = ? AND l2_id IN ({})".format(
                    placeholders
                ),
                [dataset, int(scale)] + [to_signed(i) for i in batch],
            ).fetchall()
            for l2_id, blob in rows:
                found[to_unsigned(l2_id)] = unpack_runs(bytes(blob))
        return found

    def have(self, dataset, scale, l2_ids):
        """Which of ``l2_ids`` are already stored, without reading the blobs.

        The warm-up asks this about millions of keys, where :meth:`get_many`
        would decompress every one of them only to throw the runs away.
        """
        ids = [int(i) for i in np.asarray(l2_ids, dtype=np.uint64).ravel().tolist()]
        conn = self.connection()
        present = set()
        for start in range(0, len(ids), _MAX_PARAMS):
            batch = ids[start : start + _MAX_PARAMS]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                "SELECT l2_id FROM fragments "
                "WHERE dataset = ? AND scale = ? AND l2_id IN ({})".format(placeholders),
                [dataset, int(scale)] + [to_signed(i) for i in batch],
            ).fetchall()
            present.update(to_unsigned(row[0]) for row in rows)
        return present

    def put_many(self, dataset, scale, fragments):
        """Store ``{l2_id: runs}``, skipping keys another worker already wrote.

        Empty fragments are stored too. An L2 node with no voxels in the local
        volume is a real answer -- usually a chunk that was clipped away at the
        volume edge -- and not recording it would mean paying the dense read
        for it on every subsequent request.

        The caps are checked *before* this writes, so a single request can
        carry the cache a little past its ceiling. Bounded by one request's
        worth of fragments, which is megabytes against a limit in gigabytes;
        the alternative is a partial write whose failure point depends on
        dictionary order.
        """
        items = [
            (to_signed(l2_id), np.asarray(runs)) for l2_id, runs in dict(fragments).items()
        ]
        if not items:
            return 0

        rows = []
        for key, runs in items:
            blob = pack_runs(runs)
            rows.append((dataset, int(scale), key, len(runs), rle_voxel_count(runs), blob))

        conn = self.connection()
        # IMMEDIATE so the counter read and the inserts cannot interleave with
        # another process's transaction and lose an update.
        conn.execute("BEGIN IMMEDIATE")
        try:
            written = 0
            added = 0
            for row in rows:
                cursor = conn.execute(
                    "INSERT INTO fragments "
                    "(dataset, scale, l2_id, n_runs, n_voxels, blob) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (dataset, scale, l2_id) DO NOTHING",
                    row,
                )
                # Entries are immutable, so a conflict means a concurrent
                # worker computed the same node. Not counted, not an error.
                if cursor.rowcount > 0:
                    written += 1
                    added += len(row[5]) + _ROW_OVERHEAD

            if written:
                self._bump(conn, "n_keys", written)
                self._bump(conn, "n_bytes", added)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return written

    @staticmethod
    def _bump(conn, name, delta):
        conn.execute(
            "INSERT INTO counters (name, value) VALUES (?, ?) "
            "ON CONFLICT (name) DO UPDATE SET value = value + ?",
            (name, delta, delta),
        )

    # --- administration ---------------------------------------------------

    def clear(self, dataset=None, scale=None):
        """Drop entries and re-derive the counters from what is left.

        Recounted rather than decremented: this is the path back from a full
        cache, and it should not depend on the counters having stayed correct.
        """
        where, params = [], []
        if dataset is not None:
            where.append("dataset = ?")
            params.append(dataset)
        if scale is not None:
            where.append("scale = ?")
            params.append(int(scale))
        clause = (" WHERE " + " AND ".join(where)) if where else ""

        conn = self.connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            deleted = conn.execute("DELETE FROM fragments" + clause, params).rowcount
            n_keys, n_bytes = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(blob) + ?), 0) FROM fragments",
                (_ROW_OVERHEAD,),
            ).fetchone()
            conn.execute("DELETE FROM counters")
            self._bump(conn, "n_keys", int(n_keys))
            self._bump(conn, "n_bytes", int(n_bytes))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Deleting rows does not shrink the file; without this the disk stays
        # occupied and clearing a full cache looks like it did nothing.
        conn.execute("VACUUM")
        return deleted


_cache = None
_cache_lock = threading.Lock()


def get_cache():
    """The process-wide cache, or ``None`` when caching is switched off.

    ``None`` is a supported state, not a degraded one: every caller falls back
    to computing the request live, which is what the service did before this
    module existed.
    """
    if not config.L2CacheEnabled:
        return None

    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = L2Cache(config.L2_CACHE_PATH)
    return _cache


def reset_cache():
    """Drop the process-wide handle. For tests and for reopening a moved file."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.close()
        _cache = None
