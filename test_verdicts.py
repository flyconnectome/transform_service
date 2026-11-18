"""The verdict log: the challenge, sessions, records and the export.

The chunkedgraph and the supervoxel pool are stood in for: what is under test
is what the service accepts and refuses, not CAVE. The pool's sampler is tested
on its own against an in-memory volume.
"""

import json
import random
import uuid

import numpy as np
import pytest
import tensorstore as ts
from fastapi.testclient import TestClient

from app import config, verdicts
from app.main import app

client = TestClient(app)

# The real lookup, before the fixture stands in for it.
REAL_ROOT_AT = verdicts.root_at

STACK = "test_stack"
OTHER = "other_stack"
SV = "81624458144052736"
ROOT = "720575940626838909"


@pytest.fixture(autouse=True)
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VERDICTS_DB_PATH", str(tmp_path / "verdicts.sqlite"))
    monkeypatch.setattr(config, "VERDICT_SECRET", "test-secret")
    monkeypatch.setattr(config, "VERDICT_ADMIN_KEY", "admin-key")
    # Only these datasets, so a real one offering verdicts is not listed here.
    collecting = {"services": ["verdicts"], "graphene": "graphene://example.invalid", "verdict_scale": 0}
    monkeypatch.setattr(
        config,
        "DATASOURCES",
        {
            "seg_a": {**collecting, "datastack": STACK},
            "seg_b": {**collecting, "datastack": OTHER},
            "seg_c": {"services": ["query"], "datastack": "not_offered"},
        },
    )
    monkeypatch.setattr(verdicts.pool, "take", lambda name: SV)
    monkeypatch.setattr(verdicts, "checks", verdicts._Checks())
    asked = []

    def root_at(name, supervoxel, at):
        asked.append((name, supervoxel, at))
        return ROOT

    monkeypatch.setattr(verdicts, "root_at", root_at)
    return asked


def challenge(datastack=STACK):
    response = client.get("/verdicts/challenge", params={"datastack": datastack})
    assert response.status_code == 200, response.text
    return response.json()


def answer(token, root=ROOT):
    return client.post("/verdicts/session", json={"challenge": token, "root": root})


def session(datastack=STACK):
    response = answer(challenge(datastack)["challenge"])
    assert response.status_code == 200, response.text
    return response.json()["session"]


def record(**changes):
    out = {
        "id": str(uuid.uuid4()),
        "source": "set:rv4-ad75ed161bb6",
        "neuron": "720575940600000001",
        "piece": "720575940600000001",
        "end_uid": 17,
        "kind": "tip",
        "x": 1000.0,
        "y": 2000.0,
        "z": 3000.0,
        "candidate": "720575940600000002",
        "rank": 1,
        "q": 0.41,
        "r": 0.93,
        "p_tip": 0.8,
        "conf": 0.51,
        "shown": True,
        "organelle": False,
        "verdict": "accept",
        "at": "2026-11-18T14:09:05.250Z",
        "searched_at": "2026-11-18T12:00:00Z",
        "coda": "76d68e7b",
        "name": "A Reader",
        "email": "reader@example.org",
    }
    out.update(changes)
    return out


def post(records, token):
    return client.post(
        "/verdicts", json={"records": records}, headers={"Authorization": "Bearer " + token}
    )


def export_response(**params):
    return client.get("/verdicts/export", params=params, headers={"Authorization": "Bearer admin-key"})


def export(**params):
    response = export_response(**params)
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines()]


# --- the challenge ------------------------------------------------------------


def test_datastacks_lists_the_datasets_offering_verdicts():
    assert client.get("/verdicts/datastacks").json() == {
        "collecting": True,
        "datastacks": [OTHER, STACK],
    }


def test_without_a_secret_nothing_is_collected(monkeypatch):
    monkeypatch.setattr(config, "VERDICT_SECRET", None)
    assert client.get("/verdicts/datastacks").json()["collecting"] is False
    response = client.get("/verdicts/challenge", params={"datastack": STACK})
    assert response.status_code == 503
    assert "VERDICT_SECRET" in response.json()["detail"]


def test_a_datastack_not_collected_is_named():
    response = client.get("/verdicts/challenge", params={"datastack": "flywire"})
    assert response.status_code == 404
    assert STACK in response.json()["detail"]


def test_a_challenge_names_a_supervoxel_and_a_whole_second():
    asked = challenge()
    assert asked["supervoxel"] == SV
    assert asked["datastack"] == STACK
    assert asked["at"].endswith("Z") and "." not in asked["at"]


def test_a_right_answer_buys_a_session_and_is_checked_at_the_named_moment(service):
    asked = challenge()
    response = answer(asked["challenge"])
    assert response.status_code == 200
    assert response.json()["datastack"] == STACK
    ((name, supervoxel, at),) = service
    assert (name, supervoxel) == (STACK, SV)
    assert verdicts._iso(at) == asked["at"]


def test_a_wrong_answer_is_refused():
    asked = challenge()
    response = answer(asked["challenge"], "720575940600000999")
    assert response.status_code == 403


def test_a_supervoxel_the_graph_does_not_know_is_never_passed(monkeypatch):
    monkeypatch.setattr(verdicts, "root_at", lambda *args: "0")
    asked = challenge()
    response = answer(asked["challenge"], "0")
    assert response.status_code == 403


def test_a_challenge_is_answered_once_even_when_wrong():
    asked = challenge()
    wrong = answer(asked["challenge"], "1")
    assert wrong.status_code == 403
    again = answer(asked["challenge"])
    assert again.status_code == 401
    assert "already" in again.json()["detail"]


def test_an_expired_challenge_is_refused(monkeypatch):
    monkeypatch.setattr(config, "VerdictChallengeSeconds", -1)
    asked = challenge()
    response = answer(asked["challenge"])
    assert response.status_code == 401
    assert "expired" in response.json()["detail"]


def test_a_tampered_challenge_is_refused():
    asked = challenge()
    body, mac = asked["challenge"].split(".")
    forged = verdicts._b64(
        json.dumps({**json.loads(verdicts._unb64(body)), "sv": "1"}, separators=(",", ":")).encode()
    )
    response = answer(forged + "." + mac)
    assert response.status_code == 401


def test_a_challenge_signed_with_another_secret_is_refused(monkeypatch):
    asked = challenge()
    monkeypatch.setattr(config, "VERDICT_SECRET", "rotated")
    response = answer(asked["challenge"])
    assert response.status_code == 401


def test_answers_past_the_rate_are_turned_away_without_spending_the_challenge(monkeypatch):
    monkeypatch.setattr(config, "VerdictChecksPerMinute", 1)
    assert answer(challenge()["challenge"]).status_code == 200
    asked = challenge()
    refused = answer(asked["challenge"])
    assert refused.status_code == 429
    monkeypatch.setattr(verdicts, "checks", verdicts._Checks())
    assert answer(asked["challenge"]).status_code == 200


def test_a_failed_check_says_to_ask_again_and_keeps_the_cause_to_itself(monkeypatch):
    def broken(dataset):
        raise RuntimeError("https://internal.example/secret-path")

    monkeypatch.setattr(verdicts, "root_at", REAL_ROOT_AT)
    monkeypatch.setattr(verdicts.sparsevol_live, "open_graph", broken)
    response = answer(challenge()["challenge"])
    assert response.status_code == 502
    assert "Ask for a new challenge" in response.json()["detail"]
    assert "secret-path" not in response.text


def test_a_dataset_offering_verdicts_without_a_datastack_is_a_config_error(monkeypatch):
    monkeypatch.setitem(config.DATASOURCES, "seg_d", {"services": ["verdicts"]})
    response = client.get("/verdicts/datastacks")
    assert response.status_code == 500
    assert "seg_d" in response.json()["detail"]


def test_two_datasets_offering_one_datastack_is_a_config_error(monkeypatch):
    monkeypatch.setitem(config.DATASOURCES, "seg_d", {"services": ["verdicts"], "datastack": STACK})
    assert client.get("/verdicts/datastacks").status_code == 500


# --- records ------------------------------------------------------------------


def test_records_are_kept_and_exported_as_given():
    token = session()
    first, second = record(), record(verdict="reject", q=None, r=None, shown=None)
    response = post([first, second], token)
    assert response.status_code == 200, response.text
    assert response.json() == {"received": 2, "new": 2}

    rows = export()
    assert [row["id"] for row in rows] == [first["id"], second["id"]]
    assert rows[0]["neuron"] == first["neuron"]  # text, never a float
    assert rows[0]["shown"] is True and rows[0]["organelle"] is False
    assert rows[0]["at"] == "2026-11-18T14:09:05.250000Z"
    assert rows[1]["q"] is None and rows[1]["shown"] is None
    assert rows[0]["session"] == rows[1]["session"]
    assert rows[0]["name"] == "A Reader"


def test_a_retried_record_is_not_counted_twice():
    token = session()
    one = record()
    assert post([one], token).json() == {"received": 1, "new": 1}
    assert post([one, record()], token).json() == {"received": 2, "new": 1}
    assert len(export()) == 2


def test_a_retry_spelled_in_capitals_is_the_same_record():
    token = session()
    one = record()
    post([one], token)
    assert post([{**one, "id": one["id"].upper()}], token).json() == {"received": 1, "new": 0}


def test_the_scheme_of_the_header_is_any_case():
    response = client.post(
        "/verdicts", json={"records": [record()]}, headers={"Authorization": "bearer " + session()}
    )
    assert response.status_code == 200


def test_name_and_email_are_optional():
    token = session()
    one = record()
    del one["name"], one["email"]
    assert post([one], token).status_code == 200
    assert export()[0]["email"] is None


def test_a_post_needs_a_session():
    response = client.post("/verdicts", json={"records": [record()]})
    assert response.status_code == 401


def test_a_challenge_is_not_a_session():
    response = post([record()], challenge()["challenge"])
    assert response.status_code == 401


def test_an_expired_session_is_refused(monkeypatch):
    monkeypatch.setattr(config, "VerdictSessionSeconds", -1)
    token = session()
    response = post([record()], token)
    assert response.status_code == 401
    assert "new challenge" in response.json()["detail"]


def test_a_record_is_filed_under_its_session_s_datastack():
    post([record()], session(OTHER))
    assert [row["datastack"] for row in export()] == [OTHER]


def test_a_record_cannot_name_its_own_datastack():
    response = post([record(datastack=STACK)], session(OTHER))
    assert response.status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"q": 1.5},
        {"conf": -0.1},
        {"neuron": "7.2e17"},
        {"verdict": "maybe"},
        {"source": "rv4"},
        {"at": "2026-11-18T14:09:05"},
        {"rank": 0},
        {"id": "-" * 36},
        {"surprise": 1},
    ],
)
def test_one_bad_record_refuses_the_batch_and_says_which(change):
    token = session()
    response = post([record(), record(**change)], token)
    assert response.status_code == 422
    assert any(error["loc"][:3] == ["body", "records", 1] for error in response.json()["detail"])
    assert export() == []


def test_a_neuron_ending_here_is_a_verdict_of_its_own():
    token = session()
    assert post([record(verdict="ends")], token).status_code == 200
    assert export()[0]["verdict"] == "ends"


def test_a_piece_auto_extend_accepted_is_a_record_of_its_own():
    token = session()
    assert post([record(verdict="auto-accepted")], token).status_code == 200
    assert export()[0]["verdict"] == "auto-accepted"


def test_a_cleared_verdict_is_a_record():
    token = session()
    assert post([record(verdict="cleared")], token).status_code == 200
    assert export()[0]["verdict"] == "cleared"


# --- the export ---------------------------------------------------------------


def test_an_export_streamed_a_row_at_a_time_is_whole(monkeypatch):
    monkeypatch.setattr(verdicts, "_EXPORT_BATCH", 1)
    token = session()
    sent = [record() for _ in range(25)]
    post(sent, token)
    assert [row["id"] for row in export()] == [one["id"] for one in sent]


def test_the_export_needs_the_admin_key():
    assert client.get("/verdicts/export").status_code == 401
    response = client.get("/verdicts/export", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_without_an_admin_key_there_is_no_export(monkeypatch):
    monkeypatch.setattr(config, "VERDICT_ADMIN_KEY", None)
    response = client.get("/verdicts/export", headers={"Authorization": "Bearer admin-key"})
    assert response.status_code == 503


def test_the_export_filters_by_datastack_and_moment():
    post([record()], session())
    post([record()], session(OTHER))
    assert [row["datastack"] for row in export(datastack=OTHER)] == [OTHER]
    assert len(export(since="2000-01-01T00:00:00Z")) == 2
    assert export(since="2999-01-01T00:00:00Z") == []
    assert export_response(since="yesterday").status_code == 422


# --- the supervoxel pool --------------------------------------------------------


def test_sampling_returns_the_distinct_supervoxels_of_one_chunk():
    data = np.zeros((64, 64, 16, 1), dtype=np.uint64)
    data[:, :32] = 81624458144052736
    data[:, 32:] = 81624458144052737
    data[:, :, 8:] = 5  # in other chunks along z only
    found = verdicts.sample_supervoxels(ts.array(data), random.Random(0), chunk=(16, 64, 8))
    assert sorted(found) in (["81624458144052736", "81624458144052737"], ["5"])


def test_sampling_reads_a_volume_with_no_channel_axis():
    data = np.full((32, 32, 8), 7, dtype=np.uint64)
    assert verdicts.sample_supervoxels(ts.array(data), random.Random(0)) == ["7"]


def test_sampling_outside_the_tissue_finds_nothing():
    data = np.zeros((64, 64, 16, 1), dtype=np.uint64)
    assert verdicts.sample_supervoxels(ts.array(data), random.Random(0), tries=3) == []


def test_a_pool_hands_out_what_it_filled(monkeypatch):
    pool = verdicts._Pool()
    monkeypatch.setattr(pool, "_fill", lambda name: ["1", "2"])
    assert {pool.take(STACK), pool.take(STACK)} == {"1", "2"}


def test_an_empty_pool_says_so(monkeypatch):
    pool = verdicts._Pool()
    monkeypatch.setattr(pool, "_fill", lambda name: [])
    with pytest.raises(verdicts.HTTPException) as refused:
        pool.take(STACK)
    assert refused.value.status_code == 503
