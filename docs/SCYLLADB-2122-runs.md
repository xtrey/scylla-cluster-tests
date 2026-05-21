# SCYLLADB-2122 Reproduction Runs

## Issue Summary

**Problem:** Paxos table grows to 645 GB vs 226 GB user table (2.85:1 ratio) with LWT operations on tablets-enabled ScyllaDB, causing semaphore starvation and cascading timeouts.

**Customer:** Solitics cluster 44881, ScyllaDB 2025.4.5, 3x i8g.2xlarge (EBS gp3), tablets enabled.

**Root Cause (confirmed via SCYLLADB-2130):** Paxos entries are LIVE cells with baked-in TTL (`paxos_grace_seconds`). Compaction cannot remove them before expiry.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `data_dir/cs_paxos_growth_lwt.yaml` | Stress profile — schema, 3 non-key columns, INSERT IF NOT EXISTS + SERIAL read queries |
| `longevity_lwt_paxos_growth_test.py` | Test class — `test_paxos_growth()`: INSERT (400 threads) + SERIAL reads (200 threads), no DELETE, no repair |
| `test-cases/longevity/longevity-lwt-paxos-growth-tablets.yaml` | Cluster/infra config — tablets enabled, 4h stress |
| `test-cases/longevity/longevity-lwt-paxos-growth-vnodes.yaml` | Same as tablets but `enable_tablets: false` |
| `jenkins-pipelines/oss/longevity/longevity-lwt-paxos-growth-tablets.jenkinsfile` | CI pipeline — tablets |
| `jenkins-pipelines/oss/longevity/longevity-lwt-paxos-growth-vnodes.jenkinsfile` | CI pipeline — vnodes |

### Current Schema (`cs_paxos_growth_lwt.yaml`)

```sql
CREATE TABLE event_history (
    brand text,
    subscriber_id text,
    creation_date timestamp,
    unique_key text,
    event_type text,
    event_name text,
    username text,
    PRIMARY KEY ((brand, subscriber_id), creation_date, unique_key)
) WITH CLUSTERING ORDER BY (creation_date DESC, unique_key DESC)
  AND default_time_to_live = 604800
  AND gc_grace_seconds = 864000
  AND paxos_grace_seconds = 864000
  AND tombstone_gc = {'mode': 'repair', 'propagation_delay_in_seconds': '3600'}
  AND compaction = {'class': 'IncrementalCompactionStrategy'}
  AND compression = {'sstable_compression': 'LZ4Compressor'}
```

---

## Changelog

### 2026-05-28 — Consolidation and simplification

- Deleted `cs_paxos_growth_insert_only.yaml` — single profile now covers all test variants
- Trimmed `cs_paxos_growth_lwt.yaml` from 9 non-key columns to 3 (`event_type`, `event_name`, `username`) — keeps user rows small so paxos metadata dominates
- Removed `write_delete-if-exists` query from profile — INSERT + SERIAL read only (matches customer's primary pattern)
- Fixed `LZ4WithDictsCompressor` → `LZ4Compressor` — experimental codec caused vnodes cluster start failure
- Simplified `test_paxos_growth()` — removed DELETE phase and `test_paxos_growth_insert_only()` method
- Both tablets and vnodes test YAMLs now identical except `enable_tablets: false` on vnodes
- Both Jenkinsfiles updated to call `test_paxos_growth`
- Removed `simulated_racks: 0` from tablets YAML — AWS default (3 racks) matches customer topology; kept `rf_rack_valid_keyspaces: false`

---

## Run History

### Run 8

| Field | Value |
|-------|-------|
| **Test ID** | `6d1b5b78` |
| **Commit** | `7babd4444` |
| **Date** | 2026-05-24 |
| **ScyllaDB Version** | 2025.4.5 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Cluster** | 3 DB nodes, 3 loaders, on-demand |
| **Duration** | 60 min stress |
| **Schema** | Flat PK: `PRIMARY KEY ((unique_key, unique_value, context))` |
| **Approach** | 100K keys, continuous INSERT IF NOT EXISTS + SERIAL reads, 400 write + 200 read threads |
| **Paxos:User Ratio** | ~1.14-1.21:1 |
| **Timeouts** | ReadTimeoutException + OVERSIZED_ALLOCATION reproduced |
| **Outcome** | Proved timeouts/contention under LWT+EBS load. Ratio didn't grow significantly. |

---

### Run 9

| Field | Value |
|-------|-------|
| **Test ID** | `4e9407b0` |
| **Commit** | `7babd4444` |
| **Date** | 2026-05-24 |
| **ScyllaDB Version** | 2025.4.5 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Cluster** | 3 DB nodes, 3 loaders, on-demand |
| **Duration** | 120 min stress |
| **Schema** | Flat PK: `PRIMARY KEY ((unique_key, unique_value, context))` |
| **Approach** | 5M key space, continuous INSERT IF NOT EXISTS + SERIAL reads for 2h |
| **Paxos:User Ratio** | Peaked 1.43:1 early, converged to ~1.0 (keys overwritten too often) |
| **Timeouts** | Reproduced |
| **Throughput** | Write: mean 28.8ms, P99 113.9ms, max 1985ms; Read: mean 26.2ms, P99 102ms, max 5016ms |
| **Outcome** | Longer run but ratio converged. Keys get overwritten → paxos entries replaced in-place. |

---

### Run 10

| Field | Value |
|-------|-------|
| **Test ID** | `78537b37` |
| **Commit** | `7babd4444` |
| **Date** | 2026-05-25 |
| **ScyllaDB Version** | 2025.4.5 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Cluster** | 3 DB nodes, 3 loaders, on-demand |
| **Duration** | 120 min stress (184 cycles) |
| **Schema** | Flat PK: `PRIMARY KEY ((unique_key, unique_value, context))` |
| **Approach** | Cyclic write-100K / delete-90K, with repair at 90 min |
| **Paxos:User Ratio** | Stuck at ~1.14-1.15:1 throughout (paxos ~40MB, user ~35MB) |
| **Timeouts** | Reproduced (DELETE IF EXISTS ~24ms mean, P99 38.5ms) |
| **Throughput** | ~39 sec/cycle (18 sec write + 18 sec delete + overhead) |
| **Outcome** | Cyclic approach failed — same 100K keys being overwritten means paxos entries replaced in-place. Ratio never diverges. |

---

### Run 11 (INVALID)

| Field | Value |
|-------|-------|
| **Test ID** | `cc638efa` |
| **Commit** | `46dca1359` |
| **Date** | 2026-05-25 |
| **ScyllaDB Version** | 2025.4.5 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Duration** | 60 min stress |
| **Schema** | Wide partitions: `PRIMARY KEY ((brand, subscriber_id), creation_date, unique_key)` — all 15 columns including `event_payload` (200-2000 bytes) |
| **Approach** | Concurrent INSERT IF NOT EXISTS (400 threads) + DELETE IF EXISTS (400 threads, 5 min delayed) + SERIAL reads (200 threads). No repair. |
| **Paxos:User Ratio** | 0.40 → 0.58 → 0.65:1 (inverted — user rows have heavy payload) |
| **Timeouts** | 2.2M WriteTimeoutExceptions — "2 replica required but only 1 acknowledged" |
| **Outcome** | **INVALID** — one node unresponsive from start. Ratio inverted because `event_payload` made user rows much larger than paxos entries. |

---

### Run 12

| Field | Value |
|-------|-------|
| **Test ID** | `54085143-4e1b-4d30-88dd-c4e8519abb72` |
| **Date** | 2026-05-26 |
| **ScyllaDB Version** | 2025.4.5 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Duration** | ~71 min total |
| **Schema** | Wide partitions with `event_payload` (200-2000 bytes) |
| **Approach** | Sequential phases: INSERT → DELETE → SERIAL reads. 50M key space. |
| **Paxos:User Ratio** | **1.01:1** (3.33 MB paxos vs 3.28 MB user) |
| **Timeouts** | ~2,240,658 WriteTimeoutExceptions |
| **Outcome** | Paxos:user ratio ~1:1 because user rows had large payload. Timeouts confirmed paxos coordination overhead. |

---

### Run 13 (FAILED — vnodes startup error)

| Field | Value |
|-------|-------|
| **Date** | 2026-05-27 |
| **Variant** | Vnodes |
| **Failure** | `unable to find class 'org.apache.cassandra.io.compress.LZ4WithDictsCompressor'` |
| **Outcome** | **FAILED** — `LZ4WithDictsCompressor` is an experimental ScyllaDB codec not available by default. Fixed by changing to `LZ4Compressor`. |

---

### Run 14 (FAILED — tablets wrong schema)

| Field | Value |
|-------|-------|
| **Date** | 2026-05-27 |
| **Variant** | Tablets |
| **Failure** | `Unknown identifier branch` — table created with PK-only schema but stress used full-schema queries |
| **Outcome** | **FAILED** — `prepare_write_cmd` pointed to wrong profile. Fixed by aligning to `cs_paxos_growth_lwt.yaml`. |

---

### Run 15 (FAILED — vnodes wrong test method)

| Field | Value |
|-------|-------|
| **Date** | 2026-05-28 |
| **Variant** | Vnodes |
| **Failure** | `not found: test_paxos_growth_insert_only` — method was removed |
| **Outcome** | **FAILED** — Jenkins ran stale Jenkinsfile. Fixed: both Jenkinsfiles now call `test_paxos_growth`. Needs re-trigger. |

---

### Run 16 — Tablets

| Field | Value |
|-------|-------|
| **Test ID** | `b655f57e` |
| **Date** | 2026-05-27 → 2026-05-28 |
| **Variant** | Tablets |
| **ScyllaDB Version** | 2025.4.5-0.20260301.b65523b50f46 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Cluster** | 3 DB nodes, 3 loaders (c7i.xlarge) |
| **Duration** | 240 min stress (4h) |
| **Schema** | `PRIMARY KEY ((brand, subscriber_id), creation_date, unique_key)` + 3 non-key columns |
| **Key Space** | 50M (`seq=1..50000000`) |
| **Threads** | 400 write + 200 SERIAL read |
| **Tablet Count** | 128 tablets for `event_history`, paxos co-located |
| **Paxos:User Ratio** | 0.90 → 1.02 → settled at **0.94:1** (final: 9.0 MB paxos / 9.5 MB user) |
| **SSTables** | ~700 per measurement (high tablet parallelism); final 629 paxos, 652 user |
| **Compression** | Paxos: 0.5 ratio, User: 0.8 ratio |
| **Throughput (write)** | 1,885 op/s, mean 185.7 ms, P99 883.9 ms, 27.1M total partitions |
| **Throughput (read)** | 875 op/s, mean 209.5 ms, P99 903.9 ms, 12.6M total partitions |
| **Timeouts** | 4.05M WriteTimeoutException (CAS) + 1,286 server-side paxos timeouts |
| **cassandra-stress errors** | **0** (all timeouts retried successfully) |
| **Outcome** | Ratio stable ~0.95:1 for 4h. Paxos never diverges from user table. No unbounded growth. |

---

### Run 16 — Vnodes

| Field | Value |
|-------|-------|
| **Test ID** | `0e698882` |
| **Date** | 2026-05-29 |
| **Variant** | Vnodes |
| **ScyllaDB Version** | 2025.4.5 |
| **Instance Type** | r5b.2xlarge + gp2 300GB EBS |
| **Cluster** | 3 DB nodes, 3 loaders (c7i.xlarge) |
| **Duration** | 120 min stress (2h) — **NOTE: should have been 240 min** |
| **Schema** | `PRIMARY KEY ((brand, subscriber_id), creation_date, unique_key)` + 3 non-key columns |
| **Key Space** | 50M (`seq=1..50000000`) |
| **Threads** | 400 write + 200 SERIAL read |
| **Paxos:User Ratio** | 0.41 → 0.68 → settled at **0.47:1** (final: 21.1 MB paxos / 45.1 MB user) |
| **SSTables** | 7 per node (21 cluster-wide) — compaction aggressive |
| **Compression** | Ratio 1.0 (no compression effect on vnodes) |
| **Throughput (write)** | 1,330 op/s, mean 288.9 ms, P99 851.4 ms, 9.6M total partitions |
| **Throughput (read)** | ~346 op/s per loader (3 loaders = ~1,037 op/s), mean 339.8 ms, P99 982 ms |
| **Timeouts** | 2.38M ReadTimeoutException (LOCAL_SERIAL) + 207K WriteTimeoutException |
| **cassandra-stress errors** | **0** (all timeouts retried successfully) |
| **Outcome** | User table 2x paxos size (inverted ratio). Vnodes show lower throughput and higher latency than tablets. |

---

### Run 16 — Comparative Analysis

| Metric | Tablets | Vnodes | Delta |
|--------|---------|--------|-------|
| Paxos:User ratio | 0.94:1 | 0.47:1 | Tablets paxos ~equal to user; vnodes paxos is half |
| Write throughput | 1,885 op/s | 1,330 op/s | Tablets 1.4x faster |
| Write latency (mean) | 185.7 ms | 288.9 ms | Tablets 1.6x lower |
| Read latency (mean) | 209.5 ms | 339.8 ms | Tablets 1.6x lower |
| Read latency (P99) | 903.9 ms | 982 ms | Similar |
| Timeout type | WriteTimeout (CAS) | ReadTimeout (LOCAL_SERIAL) | Different failure mode |
| SSTables (cluster) | ~650 | 21 | Tablets: 30x more SSTables |
| Compression effective | Yes (0.5-0.8) | No (1.0) | Vnodes not compressing |
| Stress duration | 240 min | 120 min | **Vnodes ran half the time** |

**Key findings:**

1. **Paxos ratio inverted on vnodes** — user table grows to 2x paxos size on vnodes. On tablets, they stay equal. This is the opposite of the customer's problem (paxos > user).

2. **Neither variant reproduces 2.85:1** — the customer's ratio requires paxos entries to accumulate faster than user data. Our test overwrites existing keys too often (54% key collision rate on tablets over 4h with 27M writes into 50M key space).

3. **Vnodes compression ineffective** — SSTable compression ratio is 1.0 on vnodes vs 0.5 on tablets, making vnodes user table appear much larger on disk.

4. **Tablets are faster** — 1.4x write throughput, 1.6x lower latency. Tablet-level parallelism benefits LWT coordination.

---

## Key Learnings

1. **Paxos has ONE row per full primary key** — repeated LWT on the same PK updates in-place. Small key spaces can't grow paxos unbounded.

2. **User row size matters** — large payloads invert the ratio. 3 small non-key columns keep user rows ~50-100 bytes vs paxos ~300-500 bytes.

3. **Timeouts reproduced consistently** — every valid run produced WriteTimeoutException / ReadTimeoutException under high LWT throughput on EBS.

4. **`LZ4WithDictsCompressor` is experimental** — replaced with `LZ4Compressor`; no effect on paxos growth behavior.

5. **SCYLLADB-2130 root cause confirmed** — paxos entries are LIVE cells with baked-in TTL; compaction cannot remove them.

6. **Single profile, two test YAMLs** — tablets and vnodes share everything; only difference is `enable_tablets: false`.

7. **Key collision prevents growth** — with 50M key space and ~27M writes over 4h, 54% of keys are touched at least twice. Each re-touch replaces the paxos entry in-place rather than accumulating a new one.

8. **Vnodes compression anomaly** — vnodes show SSTable compression ratio 1.0 (no benefit) while tablets achieve 0.5. This inflates vnodes user table relative to paxos.

---

## Next Run (READY)

| Field | Value |
|-------|-------|
| **Test** | `test_paxos_growth` |
| **Profile** | `data_dir/cs_paxos_growth_lwt.yaml` |
| **Variants** | Tablets and Vnodes |
| **Duration** | 4 hours (240 min stress) — **fix vnodes YAML to match** |
| **Schema** | `PRIMARY KEY ((brand, subscriber_id), creation_date, unique_key)` + 3 non-key columns |
| **Key Space** | 500M unique keys (`seq=1..500000000`) — increased 10x to minimize collisions |
| **Threads** | 400 write + 200 SERIAL read |
| **Operations** | INSERT IF NOT EXISTS + SERIAL SELECT |
| **Expected** | paxos:user ratio > 1.0 and growing monotonically |
| **Rationale** | At ~1,800 writes/sec over 4h = ~26M unique keys written. With 500M key space, collision rate drops to ~5% vs 54% with 50M. Each write creates a new paxos entry that persists for `paxos_grace_seconds` (10 days). |
