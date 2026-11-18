"""Review verdicts from Coda, kept for SEAM's calibration.

Readers of Coda's Review Suggestions card judge SEAM's merge suggestions --
accept, reject, ends, unsure -- and, if they opt in, each verdict is posted here, with
``auto-accepted`` for a piece Coda's Auto-extend accepted and Submit merged unjudged. SEAM
reads them back through the export to refit how often a suggestion at a given
score is right. Coda's side and the decisions behind it are in Coda's
``docs/seam-verdict-log-plan.md``.

**Who may post.** Coda never sends a reader's CAVE token. Instead the service
asks a question only someone the chunkedgraph lets in can answer: *what was
supervoxel S's root at moment T?* The service picks both: T is "now", and S
comes from the service's own copy of the segmentation, so the reader cannot pick
an easy one. The service checks the answer with its own credentials. That proves
the poster can look up this dataset's roots -- or holds a recent copy of its
root table, which answers for every supervoxel not edited since. It does not
prove who they are. The name and email on a record are what the reader typed.

**Nothing is held between the two steps.** A challenge is signed, not stored,
so whichever gunicorn worker receives the answer can check it. Only its nonce is
written down, so it cannot be answered twice. A passed challenge buys a session
token, signed the same way, good for one datastack until it expires. A record's
datastack is the session's, stamped here, never the client's to say.

**Records are append-only.** A changed mind is a newer record. Each carries an
id minted by the client, so a retried post is not counted twice.
"""

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import random
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import List, Literal, Optional

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config, datasource, sparsevol_live

router = APIRouter(prefix="/verdicts", tags=["verdicts"])
log = logging.getLogger(__name__)

# A root, supervoxel or candidate id: text, as Coda and SEAM keep them. Through a
# float an 18-digit id names a different segment.
_ID = r"^[0-9]{1,20}$"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    id          TEXT PRIMARY KEY,
    datastack   TEXT NOT NULL,
    session     TEXT NOT NULL,
    received_at TEXT NOT NULL,
    source      TEXT NOT NULL,
    neuron      TEXT NOT NULL,
    piece       TEXT NOT NULL,
    end_uid     INTEGER NOT NULL,
    kind        TEXT,
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
    candidate   TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    q REAL, r REAL, p_tip REAL, conf REAL,
    shown       INTEGER,
    organelle   INTEGER,
    verdict     TEXT NOT NULL,
    at          TEXT NOT NULL,
    searched_at TEXT,
    coda        TEXT,
    name        TEXT,
    email       TEXT
);
CREATE INDEX IF NOT EXISTS verdicts_received ON verdicts (datastack, received_at);
CREATE TABLE IF NOT EXISTS used_nonces (
    nonce   TEXT PRIMARY KEY,
    expires REAL NOT NULL
);
"""

# Rows per step of the export's stream.
_EXPORT_BATCH = 1000


# --- storage ----------------------------------------------------------------

_schema_lock = threading.Lock()
_schema_ready = set()


def _connect(across_threads=False):
    """A connection to the verdict store, its schema made on first use.

    A connection per call rather than one per thread: posts are rare next to
    the sparsevol traffic, and a short-lived connection cannot be left holding a
    lock by a thread that went on to something else. WAL is set once, when the
    file is made -- see ``l2cache`` for why not per connection.

    ``across_threads`` is for the export's stream: Starlette runs each step of
    a plain generator as its own hop to the threadpool, so one connection is
    used, one step at a time, from several threads.
    """
    path = config.VERDICTS_DB_PATH
    with _schema_lock:
        if path not in _schema_ready:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            with closing(sqlite3.connect(path, timeout=30.0)) as conn:
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.OperationalError:
                    pass
                conn.executescript(_SCHEMA)
                conn.commit()
            _schema_ready.add(path)
    return closing(sqlite3.connect(path, timeout=30.0, check_same_thread=not across_threads))


def _spend_nonce(nonce, expires):
    """Write a challenge's nonce down; ``False`` if it was already spent.

    Spent ones that have expired are swept on the way, since an expired
    challenge is refused on its own date and its nonce has nothing left to guard.
    """
    with _connect() as conn, conn:
        conn.execute("DELETE FROM used_nonces WHERE expires < ?", (time.time(),))
        try:
            conn.execute(
                "INSERT INTO used_nonces (nonce, expires) VALUES (?, ?)",
                (nonce, expires),
            )
        except sqlite3.IntegrityError:
            return False
    return True


# --- signed tokens ----------------------------------------------------------


def _secret():
    secret = config.VERDICT_SECRET
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="This service is not collecting verdicts: VERDICT_SECRET is not set.",
        )
    return secret.encode()


def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload):
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    mac = hmac.new(_secret(), body.encode(), hashlib.sha256).digest()
    return "{}.{}".format(body, _b64(mac))


def _open(token, kind):
    """The payload of a token this service signed, of the kind expected and in date.

    The kind is checked so a challenge can never be presented as a session.
    """
    secret = _secret()
    try:
        body, mac = token.split(".")
        good = hmac.new(secret, body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(good, _unb64(mac)):
            raise ValueError
        payload = json.loads(_unb64(body))
    except Exception:
        raise HTTPException(status_code=401, detail="That {} is not one this service issued.".format(kind))
    if payload.get("kind") != kind:
        raise HTTPException(status_code=401, detail="That token is not a {}.".format(kind))
    if payload.get("exp", 0) < time.time():
        again = "ask for a new challenge" if kind == "challenge" else "answer a new challenge"
        raise HTTPException(status_code=401, detail="The {} has expired; {}.".format(kind, again))
    return payload


def _bearer(authorization):
    """The token of an ``Authorization: Bearer`` header, or ``""``. The scheme is any case."""
    scheme, _, token = (authorization or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _iso(seconds):
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(value):
    """One fixed spelling, microseconds always present, so stored moments sort as text.

    A moment without a time zone is taken as UTC.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --- datastacks and the supervoxel pool --------------------------------------


def _collected():
    """``{datastack: dataset key}`` for every dataset offering the verdict service.

    A dataset offering it without naming its datastack, or two naming the same
    one, is a mistake in the config, and said as one rather than half-working.
    """
    out = {}
    for key, info in config.DATASOURCES.items():
        if "verdicts" not in info.get("services", []):
            continue
        datastack = info.get("datastack")
        if not datastack or datastack in out:
            raise HTTPException(
                status_code=500,
                detail="The service's config is wrong: dataset {} offers verdicts {}.".format(
                    key, "but names no datastack" if not datastack else "for a datastack another dataset has"
                ),
            )
        out[datastack] = key
    return out


def _dataset(datastack):
    """The dataset collecting a CAVE datastack's verdicts, or a 404 naming the ones there are."""
    found = _collected().get(datastack)
    if found is None:
        known = ", ".join(sorted(_collected())) or "none"
        raise HTTPException(
            status_code=404,
            detail="Verdicts are not collected for {}. Collected for: {}.".format(datastack, known),
        )
    return found


def sample_supervoxels(store, rng, chunk=None, tries=8):
    """Distinct non-zero supervoxels from one random chunk of a segmentation.

    One whole chunk, on the grid: a sharded, gzipped volume decompresses a whole
    chunk per read, so a chunk costs what one point would and gives hundreds of
    supervoxels. A box off the grid would decompress up to eight. A chunk can
    fall outside the tissue, so a few are tried before giving up. Any axes past
    the third (a channel) are read whole; ``np.unique`` flattens them away.
    """
    lo = np.asarray(store.domain.inclusive_min[:3], dtype=np.int64)
    hi = np.asarray(store.domain.exclusive_max[:3], dtype=np.int64)
    if chunk is None:
        chunk = store.chunk_layout.read_chunk.shape
        chunk = chunk[:3] if chunk is not None and None not in chunk[:3] else (256, 256, 64)
    chunk = np.asarray(chunk, dtype=np.int64)
    counts = -(-(hi - lo) // chunk)
    for _ in range(tries):
        start = lo + np.array([rng.randrange(int(n)) for n in counts]) * chunk
        stop = np.minimum(start + chunk, hi)
        block = store[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]]
        ids = np.unique(block.read().result())
        ids = ids[ids != 0]
        if ids.size:
            return [str(int(i)) for i in ids]
    return []


class _Pool:
    """Supervoxels to ask about, per datastack, refilled from a random chunk when empty."""

    def __init__(self):
        self._lock = threading.Lock()
        self._held = {}
        self._rng = random.SystemRandom()

    def take(self, datastack):
        held = self._take(datastack)
        if held is not None:
            return held
        # Filled outside the lock: a refill reads and decompresses a chunk, and
        # may wait for a read slot, which must not hold up any other challenge.
        found = self._fill(datastack)
        with self._lock:
            self._held.setdefault(datastack, []).extend(found)
        held = self._take(datastack)
        if held is None:
            raise HTTPException(
                status_code=503,
                detail="Could not find a supervoxel to ask about just now; try again.",
            )
        return held

    def _take(self, datastack):
        with self._lock:
            held = self._held.get(datastack)
            return held.pop(self._rng.randrange(len(held))) if held else None

    def _fill(self, datastack):
        dataset = _dataset(datastack)
        scale = config.DATASOURCES[dataset]["verdict_scale"]
        store = datasource.get_datastore(dataset, scale)
        # The read shares the sparsevol read slots: it decompresses a chunk,
        # which is what those slots are rationing.
        with sparsevol_live.read_slot(sparsevol_live.LiveStats()):
            return sample_supervoxels(store, self._rng)


pool = _Pool()


class _Checks:
    """How many answers this process may check against the chunkedgraph, per minute.

    Asking for a challenge and answering it need no credential, but checking an
    answer is a call to a shared chunkedgraph under the service's own. A reader
    answers once per session; a script could answer in a loop. A token bucket,
    per worker process: past it, an answer is turned away before its challenge
    is spent.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tokens = None
        self._at = 0.0

    def admit(self):
        per_minute = config.VerdictChecksPerMinute
        with self._lock:
            now = time.monotonic()
            if self._tokens is None:
                self._tokens = float(per_minute)
            self._tokens = min(per_minute, self._tokens + (now - self._at) * per_minute / 60.0)
            self._at = now
            if self._tokens < 1.0:
                raise HTTPException(
                    status_code=429,
                    detail="Too many answers to check just now; try again in a minute.",
                    headers={"Retry-After": "60"},
                )
            self._tokens -= 1.0


checks = _Checks()


def root_at(datastack, supervoxel, at):
    """The supervoxel's root at ``at`` (epoch seconds), asked with the service's own credentials.

    Any failure is a 502 that says to ask again -- the challenge is spent by
    now -- and the cause is logged, not sent: it can name the service's own
    endpoints, and anyone may trigger it.
    """
    when = datetime.fromtimestamp(at, tz=timezone.utc)
    try:
        graph = sparsevol_live.open_graph(_dataset(datastack))
        roots = sparsevol_live.call_graph(graph.get_roots, [int(supervoxel)], timestamp=when)
    except Exception:
        log.exception("verdicts: checking supervoxel %s on %s failed", supervoxel, datastack)
        raise HTTPException(
            status_code=502,
            detail="Could not check the answer against {}'s chunkedgraph just now. Ask for a new challenge.".format(
                datastack
            ),
        )
    return str(int(roots[0]))


# --- the challenge ------------------------------------------------------------


@router.get("/datastacks")
def list_datastacks():
    """The datastacks verdicts are collected for, and whether collecting is on at all."""
    return {"collecting": bool(config.VERDICT_SECRET), "datastacks": sorted(_collected())}


@router.get("/challenge")
def challenge(datastack: str = Query(...)):
    """A supervoxel and a moment: answer with its root then, through your own CAVE access.

    Free to ask: the chunkedgraph is consulted only when an answer comes back.
    """
    _secret()
    _dataset(datastack)
    supervoxel = pool.take(datastack)
    # Whole seconds, so both sides ask the chunkedgraph about the same instant
    # with no rounding between them; a second back, so it is never ahead of
    # the chunkedgraph's own clock.
    at = math.floor(time.time()) - 1
    expires = time.time() + config.VerdictChallengeSeconds
    token = _sign(
        {
            "kind": "challenge",
            "ds": datastack,
            "sv": supervoxel,
            "at": at,
            "nonce": secrets.token_hex(16),
            "exp": expires,
        }
    )
    return {
        "challenge": token,
        "datastack": datastack,
        "supervoxel": supervoxel,
        "at": _iso(at),
        "expires": _iso(expires),
    }


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    challenge: str = Field(max_length=2000)
    root: str = Field(pattern=_ID)


@router.post("/session")
def session(answer: Answer):
    """Check an answer; a right one buys a session for that datastack.

    The nonce is spent before the answer is checked, so each challenge gets one
    answer, right or wrong.
    """
    payload = _open(answer.challenge, "challenge")
    checks.admit()
    if not _spend_nonce(payload["nonce"], payload["exp"]):
        raise HTTPException(
            status_code=401, detail="That challenge has been answered already; ask for a new one."
        )
    expected = root_at(payload["ds"], payload["sv"], payload["at"])
    if expected == "0" or answer.root != expected:
        raise HTTPException(
            status_code=403,
            detail="That is not supervoxel {}'s root at {}. Ask for a new challenge.".format(
                payload["sv"], _iso(payload["at"])
            ),
        )
    expires = time.time() + config.VerdictSessionSeconds
    token = _sign(
        {"kind": "session", "ds": payload["ds"], "sid": secrets.token_hex(8), "exp": expires}
    )
    return {"session": token, "datastack": payload["ds"], "expires": _iso(expires)}


# --- records ------------------------------------------------------------------


class Record(BaseModel):
    """One record on one candidate at one tip: a reader's verdict, or ``auto-accepted``.

    ``auto-accepted`` is not a reader's verdict (see ``verdict`` below): a fit that counts human
    labels filters on the word.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(max_length=36)
    # Which q, cutoff and r these numbers are: a published set, or a live
    # run's model and configuration. Never mixed in a fit.
    source: str = Field(pattern=r"^(set|model):\S{1,200}$")
    neuron: str = Field(pattern=_ID)
    piece: str = Field(pattern=_ID)
    end_uid: int = Field(ge=0)
    kind: Optional[str] = Field(default=None, max_length=16)
    x: float
    y: float
    z: float
    candidate: str = Field(pattern=_ID)
    rank: int = Field(ge=1, le=100_000)
    q: Optional[float] = None
    r: Optional[float] = None
    p_tip: Optional[float] = None
    conf: Optional[float] = None
    shown: Optional[bool] = None
    organelle: Optional[bool] = None
    # `reject` is a wrong continuation, `ends` a candidate rejected because the neuron ends at
    # this tip, `cleared` a verdict taken back. `auto-accepted` is not a reader's verdict: Coda's
    # Auto-extend accepted the candidate on SEAM's own cutoff, and the reader let Submit merge it
    # without judging it. Never a human label -- a fit must not count it as one.
    verdict: Literal["accept", "reject", "ends", "unsure", "cleared", "auto-accepted"]
    at: datetime
    searched_at: Optional[datetime] = None
    coda: Optional[str] = Field(default=None, max_length=64)
    name: Optional[str] = Field(default=None, max_length=200)
    email: Optional[str] = Field(default=None, max_length=320)

    @field_validator("id")
    @classmethod
    def _uuid(cls, value):
        # One spelling per UUID, so a retry in other letters is still the same record.
        try:
            return str(uuid.UUID(value))
        except ValueError:
            raise ValueError("must be a UUID")

    @field_validator("x", "y", "z")
    @classmethod
    def _finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    @field_validator("q", "r", "p_tip", "conf")
    @classmethod
    def _probability(cls, value):
        if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError("must be between 0 and 1")
        return value

    @field_validator("at", "searched_at")
    @classmethod
    def _zoned(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("must name its time zone (UTC: end with Z)")
        return value


# The stored columns: what the service stamps, then the record's own fields.
_COLUMNS = ("datastack", "session", "received_at", *Record.model_fields)
# Kept as 0/1 by SQLite, given back as booleans.
_FLAGS = tuple(name for name, field in Record.model_fields.items() if field.annotation == Optional[bool])


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[Record] = Field(min_length=1, max_length=config.VerdictMaxBatch)


@router.post("")
def post_verdicts(batch: Batch, authorization: Optional[str] = Header(default=None)):
    """Append a batch of records under a session. A record already held is skipped."""
    token = _bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="No session: answer a challenge first.")
    payload = _open(token, "session")
    sql = "INSERT OR IGNORE INTO verdicts ({}) VALUES ({})".format(
        ", ".join(_COLUMNS), ", ".join("?" * len(_COLUMNS))
    )
    with _connect() as conn, conn:
        # The write lock first, then the stamp: posts are stored in the order
        # they are stamped, so an export resumed from the last `received_at`
        # it saw never misses a post that was stamped earlier but stored later.
        conn.execute("BEGIN IMMEDIATE")
        stamped = {
            "datastack": payload["ds"],
            "session": payload["sid"],
            "received_at": _utc(datetime.now(timezone.utc)),
        }
        rows = []
        for record in batch.records:
            row = {**record.model_dump(), **stamped}
            row.update(at=_utc(record.at), searched_at=_utc(record.searched_at))
            rows.append(tuple(row[column] for column in _COLUMNS))
        before = conn.total_changes
        conn.executemany(sql, rows)
        new = conn.total_changes - before
    return {"received": len(rows), "new": new}


@router.get("/export")
def export(
    datastack: Optional[str] = None,
    since: Optional[datetime] = Query(
        default=None, description="Only records received at or after this ISO 8601 moment (UTC if no zone)."
    ),
    authorization: Optional[str] = Header(default=None),
):
    """Every record, as JSON lines in the order received. Behind the admin key.

    ``name`` and ``email`` are what the reader typed; ``session`` is the one
    thing checked -- that whoever posted could read the datastack then.
    """
    key = config.VERDICT_ADMIN_KEY
    if not key:
        raise HTTPException(status_code=503, detail="No VERDICT_ADMIN_KEY is set, so there is no export.")
    if not hmac.compare_digest(_bearer(authorization).encode(), key.encode()):
        raise HTTPException(status_code=401, detail="The export needs the admin key.")

    clauses, params = [], []
    if datastack:
        clauses.append("datastack = ?")
        params.append(datastack)
    if since:
        clauses.append("received_at >= ?")
        params.append(_utc(since))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = "SELECT {} FROM verdicts{} ORDER BY received_at, rowid".format(", ".join(_COLUMNS), where)

    def line(row):
        item = dict(zip(_COLUMNS, row))
        for flag in _FLAGS:
            if item[flag] is not None:
                item[flag] = bool(item[flag])
        return json.dumps(item, separators=(",", ":")) + "\n"

    def lines():
        # In batches: each step of a plain generator is a hop to the threadpool.
        with _connect(across_threads=True) as conn:
            cursor = conn.execute(sql, params)
            while rows := cursor.fetchmany(_EXPORT_BATCH):
                yield "".join(line(row) for row in rows)

    return StreamingResponse(lines(), media_type="application/x-ndjson")
