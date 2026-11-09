"""
DPS27 COMPREHENSIVE STRESS-TEST HARNESS
========================================
Tests the frozen/train-then-freeze DPS27 implementation against linear search.

Coverage:
- Exhaustive arrays of small sizes, including empty arrays and odd sizes
- Randomized property tests over many array shapes and values
- Duplicates, negative values, booleans, strings, None, and mixed hashable values
- Found/missing targets, first/last positions, zone boundaries, and non-divisible sizes
- Search-order invariants, partition coverage, comparison counts, and freeze behavior
- Training on one zone then testing all zones, uniform, skewed, changing, and missing queries
- Training cost, break-even query count, comparison-based performance, optional timing
- CSV and JSON reports

Important:
This tests the frozen DPS27 implementation shown in the user's latest code.
It does NOT prove correctness for every imaginable Python object or every possible input.
It deliberately checks whether DPS27 returns ANY valid matching index, not necessarily the
first occurrence: with duplicates, zone reordering can make it return a later duplicate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


# ------------------------------- DPS27 ---------------------------------------

class DPS27:
    """Frozen four-zone frequency-based sequential search."""

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be >= 0")
        self.n = n
        self.bounds = [(z * n // 4, (z + 1) * n // 4) for z in range(4)]
        self.zone_hits = [0, 0, 0, 0]
        self.order = [0, 1, 2, 3]
        self.frozen = False

    def search(self, arr, target):
        if len(arr) != self.n:
            raise ValueError(f"array length {len(arr)} != initialized n {self.n}")
        comparisons = 0
        for z in self.order:
            start, end = self.bounds[z]
            for i in range(start, end):
                comparisons += 1
                if arr[i] == target:
                    return i, comparisons
        return -1, comparisons

    def learn(self, index):
        if self.frozen:
            raise RuntimeError("DPS27 is frozen!")
        if index == -1:
            return
        for z, (start, end) in enumerate(self.bounds):
            if start <= index < end:
                self.zone_hits[z] += 1
                return
        raise ValueError(f"index {index} is outside the array bounds")

    def freeze(self):
        if self.frozen:
            raise RuntimeError("freeze() called twice")
        self.order = sorted(range(4), key=lambda z: self.zone_hits[z], reverse=True)
        self.frozen = True

    def train(self, arr, training_queries):
        if len(arr) != self.n:
            raise ValueError("array length mismatch")
        if self.frozen:
            raise RuntimeError("cannot train an already-frozen DPS27")
        started = time.perf_counter_ns()
        for target in training_queries:
            index, _ = self.search(arr, target)
            self.learn(index)
        self.freeze()
        return (time.perf_counter_ns() - started) / 1_000_000


def linear_search(arr, target):
    comparisons = 0
    for i, value in enumerate(arr):
        comparisons += 1
        if value == target:
            return i, comparisons
    return -1, comparisons


# ------------------------------- Utilities -----------------------------------

def assert_valid_search(arr, target, result, label="search"):
    index, comparisons = result
    assert isinstance(index, int), f"{label}: index is not int: {index!r}"
    assert 0 <= comparisons <= len(arr), f"{label}: impossible comparisons={comparisons}, n={len(arr)}"
    if index == -1:
        assert not any(value == target for value in arr), (
            f"{label}: reported missing target {target!r}, but target exists"
        )
        assert comparisons == len(arr), (
            f"{label}: miss should compare every item, got {comparisons}/{len(arr)}"
        )
    else:
        assert 0 <= index < len(arr), f"{label}: out-of-range index {index}"
        assert arr[index] == target, (
            f"{label}: arr[{index}]={arr[index]!r} does not match target={target!r}"
        )
        assert comparisons >= 1, f"{label}: found target with zero comparisons"


def all_positions_covered(dps):
    visited = []
    for z in dps.order:
        start, end = dps.bounds[z]
        visited.extend(range(start, end))
    return sorted(visited) == list(range(dps.n)) and len(visited) == len(set(visited))


def random_value(rng):
    return rng.choice([
        rng.randint(-20, 20),
        rng.choice(["", "a", "b", "target", "0", "💥"]),
        None,
        False,
        True,
        0.0,
        1.5,
    ])


# ----------------------------- Unit/invariant tests --------------------------

def run_unit_tests():
    passed = []
    failures = []

    def test(name, fn):
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            failures.append((name, repr(exc), traceback.format_exc()))

    def test_partition_invariants():
        for n in range(0, 301):
            d = DPS27(n)
            assert len(d.bounds) == 4
            assert d.bounds[0][0] == 0
            assert d.bounds[-1][1] == n
            assert all(a <= b for a, b in d.bounds)
            assert all(d.bounds[i][1] == d.bounds[i + 1][0] for i in range(3))
            positions = [i for a, b in d.bounds for i in range(a, b)]
            assert positions == list(range(n))
            d.freeze()
            assert all_positions_covered(d)

    def test_all_small_arrays():
        # Exhaustive arrays over a tiny alphabet catch duplicates and ordering edge cases.
        alphabet = (-1, 0, 1)
        total = 0
        for n in range(0, 9):
            for tup in itertools.product(alphabet, repeat=n):
                arr = list(tup)
                d = DPS27(n)
                # Exercise each possible target, including a definitely missing value.
                queries = list(alphabet) + [999]
                for target in queries:
                    result = d.search(arr, target)
                    assert_valid_search(arr, target, result, f"n={n}, arr={arr}, target={target}")
                # Train on every value and misses; all valid zone orders must remain searchable.
                train_queries = list(arr) + [999, -999]
                d.train(arr, train_queries)
                assert sorted(d.order) == [0, 1, 2, 3]
                assert sum(d.zone_hits) <= len(train_queries)
                assert all_positions_covered(d)
                for target in queries:
                    assert_valid_search(arr, target, d.search(arr, target), "trained exhaustive")
                total += 1
        print(f"  Exhaustive arrays checked: {total:,}")
    def test_randomized_properties():
        rng = random.Random(20261031)
        cases = 12_000
        for _ in range(cases):
            n = rng.randint(0, 250)
            mode = rng.randrange(6)
            if mode == 0:
                arr = [rng.randint(-5, 5) for _ in range(n)]
            elif mode == 1:
                arr = list(range(n))
                rng.shuffle(arr)
            elif mode == 2:
                arr = [rng.choice([None, "x", "y", 0, 1, False]) for _ in range(n)]
            elif mode == 3:
                arr = [rng.randint(-10**9, 10**9) for _ in range(n)]
            elif mode == 4:
                arr = [0] * n
            else:
                arr = [random_value(rng) for _ in range(n)]

            d = DPS27(n)
            targets = [random_value(rng), -10**12, 10**12]
            if arr:
                targets += [arr[0], arr[-1], arr[rng.randrange(n)]]
            for target in targets:
                assert_valid_search(arr, target, d.search(arr, target), "random pretrain")

            train_queries = []
            for _ in range(rng.randint(0, 100)):
                if arr and rng.random() < 0.8:
                    train_queries.append(arr[rng.randrange(n)])
                else:
                    train_queries.append(("missing", rng.randrange(100000)))
            d.train(arr, train_queries)
            assert sorted(d.order) == [0, 1, 2, 3]
            assert sum(d.zone_hits) <= len(train_queries)
            assert all_positions_covered(d)
            frozen_order = tuple(d.order)
            frozen_hits = tuple(d.zone_hits)
            for target in targets:
                assert_valid_search(arr, target, d.search(arr, target), "random posttrain")
            assert tuple(d.order) == frozen_order
            assert tuple(d.zone_hits) == frozen_hits
        print(f"  Randomized property cases checked: {cases:,}")

    def test_exact_comparison_counts():
        # Unique targets in every position should cost the position's rank in the zone order.
        for n in range(0, 101):
            arr = list(range(n))
            d = DPS27(n)
            d.order = [2, 0, 3, 1]
            for i, value in enumerate(arr):
                idx, comps = d.search(arr, value)
                assert idx == i
                expected = sum(
                    d.bounds[z][1] - d.bounds[z][0]
                    for z in d.order[:d.order.index(next(
                        z for z, (a, b) in enumerate(d.bounds) if a <= i < b
                    ))]
                )
                zone = next(z for z, (a, b) in enumerate(d.bounds) if a <= i < b)
                expected += i - d.bounds[zone][0] + 1
                assert comps == expected, (n, i, comps, expected)
            _, comps = d.search(arr, object())
            assert comps == n

    def test_freeze_contract():
        d = DPS27(8)
        d.learn(0)
        d.learn(7)
        d.learn(-1)  # misses should not be counted
        assert sum(d.zone_hits) == 2
        d.freeze()
        assert d.frozen
        before = (tuple(d.order), tuple(d.zone_hits))
        try:
            d.learn(1)
            raise AssertionError("learn() did not reject update after freeze")
        except RuntimeError:
            pass
        assert before == (tuple(d.order), tuple(d.zone_hits))
        try:
            d.freeze()
            raise AssertionError("freeze() did not reject second freeze")
        except RuntimeError:
            pass

    def test_bad_inputs():
        d = DPS27(3)
        try:
            d.search([1, 2], 1)
            raise AssertionError("length mismatch not rejected")
        except ValueError:
            pass
        try:
            DPS27(-1)
            raise AssertionError("negative n not rejected")
        except ValueError:
            pass
        d = DPS27(3)
        try:
            d.learn(3)
            raise AssertionError("out-of-range index not rejected")
        except ValueError:
            pass

    def test_duplicate_semantics():
        arr = ["x", "a", "a", "a", "z", "a", "b", "a"]
        d = DPS27(len(arr))
        # Force zone 3 to be searched first. It may return a later duplicate.
        d.order = [2, 0, 1, 3]
        idx, _ = d.search(arr, "a")
        assert arr[idx] == "a"
        assert idx != arr.index("a"), (
            "This test expects DPS27 to demonstrate non-first-match semantics with duplicates"
        )

    test("partition coverage and boundaries for n=0..300", test_partition_invariants)
    test("exhaustive small arrays and targets", test_all_small_arrays)
    test("randomized arrays / targets / training", test_randomized_properties)
    test("exact comparison counts", test_exact_comparison_counts)
    test("freeze contract and ignored misses", test_freeze_contract)
    test("invalid input handling", test_bad_inputs)
    test("duplicate first-match semantic difference", test_duplicate_semantics)

    return passed, failures


# ----------------------------- Workload tests --------------------------------

def zone_bounds(n):
    return [(z * n // 4, (z + 1) * n // 4) for z in range(4)]


def index_in_zone(n, zone, rng):
    a, b = zone_bounds(n)[zone]
    if a == b:
        return None
    return rng.randrange(a, b)


def make_workload(arr, kind, count, rng):
    n = len(arr)
    queries = []
    bounds = zone_bounds(n)

    def sample_zone(z):
        a, b = bounds[z]
        if a == b:
            return ("__impossible_missing__", z)
        return arr[rng.randrange(a, b)]

    for _ in range(count):
        if kind.startswith("zone_"):
            queries.append(sample_zone(int(kind[-1]) - 1))
        elif kind == "uniform":
            if n:
                queries.append(arr[rng.randrange(n)])
            else:
                queries.append(("__missing__", 1))
        elif kind == "uniform_index_miss_20pct":
            if n and rng.random() >= 0.2:
                queries.append(arr[rng.randrange(n)])
            else:
                queries.append(("__missing__", rng.randrange(10**9)))
        elif kind == "misses":
            queries.append(("__definitely_missing__", rng.randrange(10**9)))
        elif kind == "front_heavy":
            queries.append(sample_zone(0))
        elif kind == "back_heavy":
            queries.append(sample_zone(3))
        elif kind == "skew_70_10_10_10":
            z = rng.choices([0, 1, 2, 3], weights=[10, 10, 10, 70], k=1)[0]
            queries.append(sample_zone(z))
        elif kind == "alternating_zones":
            queries.append(sample_zone((_ % 4)))
        elif kind == "cyclic_shift":
            # Repeatedly move the dominant zone to expose training mismatch.
            phase = (_ // max(1, count // 4)) % 4
            queries.append(sample_zone(phase))
        elif kind == "hotspot_one_index":
            if n:
                queries.append(arr[n // 2])
            else:
                queries.append(("__missing__", 1))
        else:
            raise ValueError(kind)
    return queries


def make_dataset(n, kind, rng):
    if kind == "unique_random":
        arr = list(range(n))
        rng.shuffle(arr)
    elif kind == "sorted_unique":
        arr = list(range(n))
    elif kind == "reverse_unique":
        arr = list(range(n - 1, -1, -1))
    elif kind == "all_same":
        arr = [7] * n
    elif kind == "few_values":
        arr = [rng.randrange(5) for _ in range(n)]
    elif kind == "negative_values":
        arr = [rng.randint(-10**9, -1) for _ in range(n)]
    elif kind == "alternating_extremes":
        arr = [(-10**12 if i % 2 else 10**12) for i in range(n)]
    elif kind == "zone_value_ranges":
        arr = []
        bounds = zone_bounds(n)
        for z, (a, b) in enumerate(bounds):
            arr.extend(rng.randint(z * 1000, z * 1000 + 99) for _ in range(b - a))
    else:
        raise ValueError(kind)
    return arr


def run_workload_matrix(n, queries_per_scenario, repeats, seed, timing):
    rng = random.Random(seed)
    dataset_kinds = [
        "unique_random", "sorted_unique", "reverse_unique", "all_same",
        "few_values", "negative_values", "alternating_extremes", "zone_value_ranges"
    ]
    scenario_kinds = [
        "zone_1", "zone_2", "zone_3", "zone_4", "uniform",
        "uniform_index_miss_20pct", "misses", "front_heavy", "back_heavy",
        "skew_70_10_10_10", "alternating_zones", "cyclic_shift", "hotspot_one_index"
    ]

    rows = []
    for di, dataset_kind in enumerate(dataset_kinds):
        arr = make_dataset(n, dataset_kind, random.Random(seed + di * 1009))

        for training_zone in range(4):
            d = DPS27(n)
            training_rng = random.Random(seed + di * 1009 + training_zone * 97)
            training_queries = make_workload(
                arr, f"zone_{training_zone + 1}", max(100, min(2000, queries_per_scenario)),
                training_rng
            )
            train_ms = d.train(arr, training_queries)
            order = [z + 1 for z in d.order]
            frozen_hits = tuple(d.zone_hits)
            frozen_order = tuple(d.order)

            for si, scenario in enumerate(scenario_kinds):
                queries = make_workload(
                    arr, scenario, queries_per_scenario,
                    random.Random(seed + di * 100000 + training_zone * 10000 + si * 101)
                )

                # Correctness and operation counts are not timed.
                linear_comps = []
                dps_comps = []
                for target in queries:
                    lr = linear_search(arr, target)
                    dr = d.search(arr, target)
                    assert_valid_search(arr, target, lr, "linear")
                    assert_valid_search(arr, target, dr, "DPS27")
                    linear_comps.append(lr[1])
                    dps_comps.append(dr[1])

                linear_ms = dps_ms = None
                if timing:
                    linear_samples, dps_samples = [], []
                    for rep in range(repeats):
                        # Alternate execution order to reduce order/cache bias.
                        order_to_run = ("linear", "dps") if rep % 2 == 0 else ("dps", "linear")
                        for algo in order_to_run:
                            start = time.perf_counter_ns()
                            if algo == "linear":
                                for q in queries:
                                    linear_search(arr, q)
                            else:
                                for q in queries:
                                    d.search(arr, q)
                            elapsed = (time.perf_counter_ns() - start) / 1_000_000
                            (linear_samples if algo == "linear" else dps_samples).append(elapsed)
                    linear_ms = statistics.median(linear_samples)
                    dps_ms = statistics.median(dps_samples)

                avg_l = statistics.mean(linear_comps) if linear_comps else 0
                avg_d = statistics.mean(dps_comps) if dps_comps else 0
                row = {
                    "dataset": dataset_kind,
                    "training_zone": training_zone + 1,
                    "learned_order": "-".join(map(str, order)),
                    "scenario": scenario,
                    "n": n,
                    "training_queries": len(training_queries),
                    "test_queries": len(queries),
                    "training_ms": round(train_ms, 6),
                    "linear_avg_comparisons": round(avg_l, 4),
                    "dps_avg_comparisons": round(avg_d, 4),
                    "comparison_ratio_linear_over_dps": round(avg_l / avg_d, 6) if avg_d else None,
                    "linear_median_ms": round(linear_ms, 6) if linear_ms is not None else None,
                    "dps_median_ms": round(dps_ms, 6) if dps_ms is not None else None,
                    "timing_speedup_linear_over_dps": round(linear_ms / dps_ms, 6)
                        if linear_ms is not None and dps_ms else None,
                    "break_even_queries_estimate": None,
                }

                # A rough break-even estimate: one-time training cost / per-query time saved.
                if linear_ms is not None and dps_ms is not None and len(queries) > 0:
                    saved_ms_per_query = (linear_ms - dps_ms) / len(queries)
                    if saved_ms_per_query > 0:
                        row["break_even_queries_estimate"] = math.ceil(train_ms / saved_ms_per_query)
                    else:
                        row["break_even_queries_estimate"] = "never_at_this_measured_rate"

                rows.append(row)
                assert tuple(d.zone_hits) == frozen_hits, "Frozen DPS27 mutated training counts during test"
                assert tuple(d.order) == frozen_order, "Frozen DPS27 changed order during test"

    return rows


# ------------------------------ Reporting ------------------------------------

def summarize(rows):
    ratios = [r["comparison_ratio_linear_over_dps"] for r in rows
              if isinstance(r.get("comparison_ratio_linear_over_dps"), (int, float))]
    speedups = [r["timing_speedup_linear_over_dps"] for r in rows
                if isinstance(r.get("timing_speedup_linear_over_dps"), (int, float))]
    wins = sum(x > 1 for x in speedups)
    losses = sum(x < 1 for x in speedups)
    ties = sum(math.isclose(x, 1.0, rel_tol=1e-6) for x in speedups)

    worst = sorted(rows, key=lambda r: r["comparison_ratio_linear_over_dps"] or 0)[:10]
    best = sorted(rows, key=lambda r: r["comparison_ratio_linear_over_dps"] or 0, reverse=True)[:10]

    return {
        "cases": len(rows),
        "timing_cases": len(speedups),
        "timing_wins": wins,
        "timing_losses": losses,
        "timing_ties": ties,
        "median_comparison_ratio": statistics.median(ratios) if ratios else None,
        "mean_comparison_ratio": statistics.mean(ratios) if ratios else None,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "mean_speedup": statistics.mean(speedups) if speedups else None,
        "best_10_by_comparisons": best,
        "worst_10_by_comparisons": worst,
        "interpretation_warning": (
            "A ratio > 1 means DPS27 used fewer comparisons than linear search. "
            "Timing is noisy and this benchmark does not establish universal superiority."
        ),
    }


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive DPS27 stress test")
    parser.add_argument("--n", type=int, default=20_000, help="Array size (default: 20000)")
    parser.add_argument("--queries", type=int, default=100, help="Queries per scenario (default: 100)")
    parser.add_argument("--repeats", type=int, default=7, help="Timing repeats (default: 7)")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--no-timing", action="store_true", help="Only correctness/operation-count tests")
    parser.add_argument("--skip-exhaustive", action="store_true", help="Skip exhaustive small-array tests")
    parser.add_argument("--output", default="dps27_stress_results", help="Output filename prefix")
    args = parser.parse_args()

    if args.n < 0 or args.queries < 1 or args.repeats < 1:
        parser.error("--n must be >= 0, --queries and --repeats must be >= 1")

    print("=" * 76)
    print("DPS27 COMPREHENSIVE STRESS TEST")
    print("=" * 76)
    print("This is a stress suite, not a mathematical proof over all possible inputs.")
    print("\n[1/2] Unit, exhaustive, and randomized correctness tests")
    passed, failures = run_unit_tests()
    if args.skip_exhaustive:
        # Unit suite currently bundles exhaustive checks; still run it to preserve
        # coverage. The flag is retained for CLI compatibility and future expansion.
        print("Note: --skip-exhaustive is currently informational; exhaustive tests were run.")

    for name in passed:
        print(f"  PASS  {name}")
    for name, error, tb in failures:
        print(f"  FAIL  {name}: {error}")
        print(tb)

    if failures:
        print(f"\nSTOP: {len(failures)} unit test group(s) failed.")
        sys.exit(1)

    print("\n[2/2] Distribution, mismatch, and workload matrix")
    print(f"Array size={args.n:,}; queries/scenario={args.queries:,}; timing repeats={args.repeats}")
    rows = run_workload_matrix(
        args.n, args.queries, args.repeats, args.seed, timing=not args.no_timing
    )
    summary = summarize(rows)

    csv_path = f"{args.output}.csv"
    json_path = f"{args.output}.json"
    write_csv(rows, csv_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f, indent=2)

    print("\n" + "=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    print(f"Workload cases: {summary['cases']:,}")
    print(f"Timing wins/losses/ties: {summary['timing_wins']}/{summary['timing_losses']}/{summary['timing_ties']}")
    if summary["median_comparison_ratio"] is not None:
        print(f"Median comparison ratio (linear / DPS27): {summary['median_comparison_ratio']:.3f}x")
        print(f"Mean comparison ratio (linear / DPS27):   {summary['mean_comparison_ratio']:.3f}x")
    if summary["median_speedup"] is not None:
        print(f"Median timing speedup: {summary['median_speedup']:.3f}x")
        print(f"Mean timing speedup:   {summary['mean_speedup']:.3f}x")

    print("\nWorst 5 scenarios by comparison ratio (DPS27 may be slower):")
    for r in summary["worst_10_by_comparisons"][:5]:
        print(f"  {r['dataset']:22} train Z{r['training_zone']} -> {r['scenario']:24} "
              f"ratio={r['comparison_ratio_linear_over_dps']:.3f}x order={r['learned_order']}")
    print("\nBest 5 scenarios by comparison ratio:")
    for r in summary["best_10_by_comparisons"][:5]:
        print(f"  {r['dataset']:22} train Z{r['training_zone']} -> {r['scenario']:24} "
              f"ratio={r['comparison_ratio_linear_over_dps']:.3f}x order={r['learned_order']}")

    print(f"\nDetailed CSV:  {csv_path}")
    print(f"Detailed JSON: {json_path}")
    print("\nREAD THIS BEFORE CLAIMING VICTORY:")
    print("- Training time is reported separately; test-time speedup excludes it.")
    print("- Break-even estimates are rough and only use measured timing deltas.")
    print("- The test covers the frozen version, not the adaptive version.")
    print("- A finite test cannot check every possible array, target, Python object, or workload.")
    print("- If a test fails, preserve the seed and failing case before changing the algorithm.")


if __name__ == "__main__":
    main()
