# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (c) 2022 ScyllaDB

import pytest

from sdcm.stress_thread import CassandraStressThread, get_timeout_from_stress_cmd
from sdcm.utils.common import time_period_str_to_seconds


@pytest.mark.parametrize(
    "duration,seconds",
    (
        ("1h1m20s", 3680),
        ("1m20s", 80),
        ("1h20s", 3620),
        ("25m", 1500),
        ("10h", 36000),
        ("25s", 25),
    ),
)
def test_duration_str_to_seconds_function(duration, seconds):
    assert time_period_str_to_seconds(duration) == seconds


@pytest.mark.parametrize(
    "stress_cmd, timeout",
    (
        (
            "cassandra-stress counter_write cl=QUORUM duration=20m"
            " -schema 'replication(strategy=NetworkTopologyStrategy,replication_factor=3)' no-warmup",
            1200 + 900,
        ),
        ("scylla-bench -workload=uniform -concurrency 64 -duration 1h -validate-data", 3600 + 900),
        ("scylla-bench -partition-count=20000 -duration=250s", 250 + 900),
        ("gemini -d --duration 10m --warmup 10s -c 5 -m write", 610 + 900),
        ("latte run --duration 10m --sampling 5s", 600 + 900),
        # Gemini commands from the issue - test case with 24h duration
        (
            "--duration 24h --warmup 10m --concurrency 200 --mode mixed --max-mutation-retries-backoff 10s",
            86400 + 600 + 900,
        ),
        # Gemini with equals sign format
        ("--duration=3h --warmup=30m --concurrency=50 --mode=mixed", 10800 + 1800 + 900),
        # Gemini command without warmup
        ("--duration 1h --concurrency 100 --mode write", 3600 + 900),
        # Critical case: YAML multiline format with newlines (the actual issue scenario)
        ("--duration 24h\n--warmup 10m\n--concurrency 200", 86400 + 600 + 900),
    ),
)
def test_get_timeout_from_stress_cmd(stress_cmd, timeout):
    assert get_timeout_from_stress_cmd(stress_cmd) == timeout


# ---------------------------------------------------------------------------
# set_hdr_tags — user-profile ops() naming convention
# ---------------------------------------------------------------------------
# Helper: build a minimal user-profile cassandra-stress command string that
# contains an ops() clause so set_hdr_tags() can parse it.  The command never
# runs; we only test tag derivation, which happens purely from the string.


def _user_profile_cmd(ops_clause: str, throttled: bool = False) -> str:
    rate = "fixed=1000/s" if throttled else "threads=100"
    return (
        f"cassandra-stress user profile=/tmp/cs_lwt_perf_small.yaml "
        f"'ops({ops_clause})' no-warmup cl=QUORUM duration=10m "
        f"-mode cql3 native -rate '{rate}'"
    )


def _get_hdr_tags(ops_clause: str, throttled: bool = False) -> list[str]:
    """Instantiate a minimal CassandraStressThread and return its hdr_tags."""
    cmd = _user_profile_cmd(ops_clause, throttled=throttled)
    thread = object.__new__(CassandraStressThread)
    thread.hdr_tags = []
    thread.set_hdr_tags(cmd)
    return thread.hdr_tags


@pytest.mark.parametrize(
    "ops_clause, throttled, expected_tags",
    [
        # --- prefixed write_* ---
        pytest.param(
            "write_stmt-insert-if-not-exists=1",
            False,
            ["WRITE-st"],
            id="write_prefix_unthrottled",
        ),
        pytest.param(
            "write_stmt-update-if-cond=1",
            True,
            ["WRITE-rt"],
            id="write_prefix_throttled",
        ),
        pytest.param(
            "write_stmt-insert=1",
            False,
            ["WRITE-st"],
            id="write_prefix_plain_insert",
        ),
        # --- prefixed read_* ---
        pytest.param(
            "read_stmt-select=1",
            False,
            ["READ-st"],
            id="read_prefix_unthrottled",
        ),
        pytest.param(
            "read_stmt-select=1",
            True,
            ["READ-rt"],
            id="read_prefix_throttled",
        ),
        # --- mixed: write_* + read_* in same ops() ---
        pytest.param(
            "write_stmt-insert-if-not-exists=1,read_stmt-select=1",
            False,
            ["WRITE-st", "READ-st"],
            id="mixed_write_and_read_prefixes",
        ),
        pytest.param(
            "write_stmt-update-if-cond=1,read_stmt-select=1",
            True,
            ["WRITE-rt", "READ-rt"],
            id="mixed_write_and_read_prefixes_throttled",
        ),
        # --- mix_* prefix: single statement that declares itself as both ---
        pytest.param(
            "mix_lwt-insert-and-select=1",
            False,
            ["WRITE-st", "READ-st"],
            id="mix_prefix_emits_both_tags",
        ),
        pytest.param(
            "mix_lwt-insert-and-select=1",
            True,
            ["WRITE-rt", "READ-rt"],
            id="mix_prefix_throttled_emits_both_tags",
        ),
        # --- backward compat: old unprefixed ops(insert=1) ---
        pytest.param(
            "insert=1",
            False,
            ["WRITE-st"],
            id="backward_compat_insert",
        ),
        pytest.param(
            "insert=1",
            True,
            ["WRITE-rt"],
            id="backward_compat_insert_throttled",
        ),
        # --- backward compat: old unprefixed ops(read=1) ---
        pytest.param(
            "read=1",
            False,
            ["READ-st"],
            id="backward_compat_read",
        ),
    ],
)
def test_set_hdr_tags_user_profile(ops_clause, throttled, expected_tags):
    """set_hdr_tags correctly maps user-profile ops() naming prefixes to HDR tags."""
    assert _get_hdr_tags(ops_clause, throttled=throttled) == expected_tags


def test_set_hdr_tags_user_profile_unknown_ops_raises():
    """set_hdr_tags raises ValueError when ops() contains no recognised prefix."""
    with pytest.raises(ValueError):
        _get_hdr_tags("stmt-insert-if-not-exists=1")  # old unprefixed, not insert=/read=


# --- non-user-profile commands (existing behaviour, regression guard) ---


@pytest.mark.parametrize(
    "stress_cmd, expected_tags",
    [
        pytest.param(
            "cassandra-stress write cl=QUORUM duration=10m -rate threads=100",
            ["WRITE-st"],
            id="plain_write",
        ),
        pytest.param(
            "cassandra-stress read cl=QUORUM duration=10m -rate threads=100",
            ["READ-st"],
            id="plain_read",
        ),
        pytest.param(
            "cassandra-stress mixed cl=QUORUM duration=10m -rate threads=100",
            ["WRITE-st", "READ-st"],
            id="plain_mixed",
        ),
        pytest.param(
            "cassandra-stress write cl=QUORUM duration=10m -rate 'fixed=5000/s threads=100'",
            ["WRITE-rt"],
            id="throttled_write",
        ),
    ],
)
def test_set_hdr_tags_standard_commands(stress_cmd, expected_tags):
    """set_hdr_tags handles standard (non-user-profile) commands unchanged."""
    thread = object.__new__(CassandraStressThread)
    thread.hdr_tags = []
    thread.set_hdr_tags(stress_cmd)
    assert thread.hdr_tags == expected_tags
