"""Compare Adaptive Region Ordering Search with ordinary linear search.

The timed adaptive run includes learning/reordering after every successful
search. Dataset and query generation are excluded from measured time.
"""
from __future__ import annotations
import argparse, csv, json, random, statistics, sys, time, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaptive_region_ordering_search import AdaptiveRegionOrderingSearch

def get_bounds(n: int, regions: int = 4):
    return [(i * n // regions, (i + 1) * n // regions) for i in range(regions)]

def linear_search(values, target):
    comparisons = 0
    for index, value in enumerate(values):
        comparisons += 1
        if value == target: return index, comparisons
    return None, comparisons

def make_queries(values, scenario, count, seed):
    rng = random.Random(seed); bounds = get_bounds(len(values)); queries = []
    for q in range(count):
        if scenario == "zone_1": zone = 0
        elif scenario == "zone_4": zone = 3
        elif scenario == "uniform": zone = rng.randrange(4)
        elif scenario == "shift_zone_2_to_zone_4": zone = 1 if q < count // 2 else 3
        elif scenario == "misses":
            queries.append(len(values) + q + 1); continue
        else: raise ValueError(f"Unknown scenario: {scenario}")
        start, end = bounds[zone]
        queries.append(values[rng.randrange(start, end)] if start < end else len(values) + q + 1)
    return queries

def measure(values, queries, method, repeats):
    totals_ms, comparison_counts = [], []
    for _ in range(repeats):
        adaptive = AdaptiveRegionOrderingSearch(values) if method == "adaptive" else None
        comparisons = 0; start = time.perf_counter_ns()
        for target in queries:
            index, count = linear_search(values, target) if adaptive is None else adaptive.search(target)
            comparisons += count
        totals_ms.append((time.perf_counter_ns() - start) / 1e6)
        comparison_counts.append(comparisons / max(1, len(queries)))
    return statistics.median(totals_ms), statistics.median(comparison_counts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", default=str(ROOT / "results" / "latest_benchmark.csv"))
    args = parser.parse_args()
    if args.n < 1 or args.queries < 1 or args.repeats < 1: parser.error("n, queries, and repeats must be positive")
    rng = random.Random(args.seed); values = list(range(args.n)); rng.shuffle(values)
    scenarios = ("zone_1", "zone_4", "uniform", "shift_zone_2_to_zone_4", "misses")
    rows = []
    for sidx, scenario in enumerate(scenarios):
        queries = make_queries(values, scenario, args.queries, args.seed + sidx + 1)
        # Correctness validation is deliberately outside the timed region.
        present = set(values)
        check = AdaptiveRegionOrderingSearch(values)
        for target in queries:
            expected_present = target in present
            linear_index, _ = linear_search(values, target)
            adaptive_index, _ = check.search(target)
            assert (linear_index is not None) == expected_present
            assert (adaptive_index is not None) == expected_present
            if adaptive_index is not None:
                assert values[adaptive_index] == target
        linear_ms, linear_cmp = measure(values, queries, "linear", args.repeats)
        adaptive_ms, adaptive_cmp = measure(values, queries, "adaptive", args.repeats)
        rows.append({"scenario": scenario, "n": args.n, "queries": args.queries, "repeats": args.repeats,
            "linear_median_ms": round(linear_ms, 6), "adaptive_median_ms": round(adaptive_ms, 6),
            "linear_avg_comparisons": round(linear_cmp, 3), "adaptive_avg_comparisons": round(adaptive_cmp, 3),
            "timing_speedup_linear_over_adaptive": round(linear_ms / adaptive_ms, 4) if adaptive_ms else None,
            "comparison_ratio_linear_over_adaptive": round(linear_cmp / adaptive_cmp, 4) if adaptive_cmp else None})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    json_path = output.with_suffix(".json")
    payload = {
        "project": "Adaptive Region Ordering Search",
        "source_revision": "DPS28A (latest archived source located; 2026-10-31)",
        "configuration": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "methodology": "Same array and query sequence for both methods; query generation and correctness validation are outside the timed section; adaptive timing includes online learning and order selection; median of complete runs is reported.",
        "results": rows,
        "limitations": [
            "Single local environment and exploratory run; not publication-grade evidence.",
            "Synthetic four-region workloads only.",
            "Timing and comparison savings are workload-dependent.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2)); print(f"CSV: {output}\nJSON: {json_path}")

if __name__ == "__main__": main()
