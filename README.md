# Adaptive Region Ordering Search

**A query-distribution-aware sequential search prototype for unsorted arrays.**

The project investigates whether observing where successful queries occur can improve the expected number of comparisons in an unsorted array by changing the order in which contiguous regions are scanned.



## Team

- **Project Lead &  Developer:** Motaz Billah Jamal Abu Al-Joloud
- **Contributors &  Developer:** Ali Abu Najm; Ahmad Radwan

See [`AUTHORS.md`](AUTHORS.md).

## Core idea

The input is an **unsorted array**. The implementation divides its index range into four contiguous regions and scans those regions in a configurable order. It learns from the region containing each successful query and estimates the probability that future successful queries will land in each region.

The adaptive implementation combines:

1. **Decayed history:** older successful-query observations gradually lose weight.
2. **Recent window:** a bounded queue tracks the most recent successful-query regions.
3. **Trend component:** compares the current normalized history with the previous history estimate.
4. **Expected-cost ordering:** evaluates all `4! = 24` region permutations and selects the order with the lowest estimated comparison cost.

The array itself is not sorted or rearranged. Only the order in which its regions are visited changes.

## Search behavior

For each query, the algorithm:

1. visits the four regions in the current learned order;
2. scans positions sequentially within each visited region;
3. returns the index of a matching value, or `None` if the value is absent;
4. after a successful search, updates the region statistics and may choose a new region order.

**Duplicate semantics:** The result is a valid index containing the target. It is not guaranteed to be the first occurrence in original array order, because the learned region order can visit a later region first.

**Miss behavior:** A miss requires checking every array element. Misses do not update the region-frequency model in this version.

## Expected-cost model

Let the array be partitioned into regions `R_1, ..., R_4`. Let `p_i` be the estimated probability that a successful query belongs to region `i`, and let `s_i` be the number of elements in that region. For a candidate region order `π`, the estimated comparisons are:

\[
C(\pi) = \sum_{j=1}^{4} p_{\pi_j}\left(\sum_{t<j}s_{\pi_t} + \frac{s_{\pi_j}+1}{2}\right)
\]

The selected order minimizes this estimated cost among the 24 possible permutations. The model assumes a roughly uniform target position within each region; actual query locations may violate that assumption.

## Complexity and limitations

- **Worst-case search:** `O(N)` comparisons, including unsuccessful queries.
- **Learning/reordering:** after a successful query, the recent window is scanned in `O(W)` and all 24 four-region permutations are evaluated.
- **Model state:** `O(4 + W)` where `W` is the recent-window length; the input array is external to the model.
- **Input contract:** keep the input array unchanged while the search object is in use; the object stores a reference to it and precomputes region boundaries.
- **No universal speed claim:** fewer comparisons do not necessarily mean lower wall-clock time. Learning, probability updates, permutation evaluation, Python overhead, and query-distribution shifts can offset comparison savings.
- **Current scope:** four contiguous regions, static array contents, sequential scans within regions, and learning only from successful queries.
- **Not a replacement for:** binary search on sorted data, hash-based lookup, or a general-purpose index when their input requirements and workload fit.

## Repository layout

```text
adaptive-region-ordering-search/
├── src/
│   └── adaptive_region_ordering_search.py
├── tests/
│   └── test_search.py
├── benchmarks/
│   ├── benchmark.py
│   └── archive_dps27_stress_test.py
├── results/
│   ├── dps28a_historical_demo.csv
│   ├── dps28a_historical_demo.json
│   └── archive/
│       ├── dps27_test_results.csv
│       ├── dps27_hybrid_results.json
│       └── dps27_legacy_summary.txt
├── archive/
│   ├── dps27_hybrid_v1.py
│   └── dps28a_original.py
├── docs/
├── AUTHORS.md
└── README.md
```

## Requirements

- Python 3.10 or newer (standard library only)
- No third-party packages are required

## Run the correctness tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover successful lookups, misses, duplicate values, empty/single-element arrays, comparison-count bounds, normalized probability estimates, and valid region permutations.

## Run the demonstration

```bash
python src/adaptive_region_ordering_search.py --n 30000 --queries 6000 --seed 28
```

This runs the source's built-in checks and produces a comparison of the adaptive method with the historical DPS27 hybrid baseline under three generated workloads. It is a demonstration, not a publication-grade benchmark.

## Run the linear-search benchmark

```bash
python benchmarks/benchmark.py --n 20000 --queries 1000 --repeats 7
```

The benchmark compares linear search and the adaptive implementation on identical query sequences from five workloads: first region, fourth region, uniform regions, a zone-distribution shift, and misses. The adaptive measurement includes its online learning/reordering cost. Dataset/query generation and correctness checks are excluded from the timed section. CSV and JSON output are written under `results/`.

## Historical evidence and how to read it

- `results/dps28a_historical_demo.*` contains the archived DPS28A-versus-DPS27-Hybrid demo (`N=30,000`, 6,000 queries, three synthetic workloads). In those recorded runs, DPS28A had slightly more comparisons and higher elapsed time in all three scenarios. This is a useful negative result, not evidence of a speedup.
- `results/archive/dps27_test_results.csv` and `dps27_legacy_summary.txt` belong to DPS27, not the latest DPS28A implementation. The old report summarizes 112 experiments (84 timing wins and 28 losses), but those outcomes must not be attributed to the renamed current version.
- Historical measurements were obtained in a specific environment and should not be generalized to arbitrary arrays or hardware.

## Research assessment

This is a worthwhile **experimental algorithm and portfolio project** because it explores online query-distribution learning, expected comparison cost, adaptation under changing workloads, and the trade-off between comparison count and update overhead.

It is **not yet established as a new or generally faster search algorithm**. Before making a research-performance claim, run repeated, same-machine comparisons against linear search and relevant indexed baselines; separate search and learning costs; evaluate many seeds, array sizes, hit/miss rates and changing distributions; and report medians, tail latency, comparisons, and memory overhead.

## License

No open-source license has been assigned in this release. Public visibility on GitHub does not by itself grant permission to reuse or redistribute the code. Choose and add a license before inviting external reuse.

### Recorded exploratory linear-baseline run

A small same-query smoke benchmark (`N=8,000`, `Q=400`, five repetitions) is included in `results/benchmark_smoke_2026-11-09.*`. It showed **5.37× measured speedup** for a workload targeting the fourth region and **3.49×** for a distribution shift from region two to region four. The first-region workload was slower (`0.73×` linear/adaptive), the uniform workload was slightly slower (`0.94×`), and misses had identical comparison counts. These mixed results are important: the method can help under skewed query distributions, but is not a universal replacement for linear search. See [`docs/BENCHMARK_METHODOLOGY.md`](docs/BENCHMARK_METHODOLOGY.md) for details and limitations.
