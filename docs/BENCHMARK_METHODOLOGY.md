# Benchmark Methodology and Recorded Smoke Run

## Method

The included benchmark compares the adaptive method to a plain linear scan on the same randomly shuffled permutation array and identical generated query sequence for each scenario. Query generation and a correctness-validation pass are completed before timing. The adaptive timed path includes all online learning, probability updates, and region-order selection triggered by successful queries. Each repetition constructs a fresh adaptive model. The median elapsed time across complete repetitions is recorded.

Scenarios:

- `zone_1`: all successful queries target the first quarter.
- `zone_4`: all successful queries target the last quarter.
- `uniform`: target regions are selected uniformly.
- `shift_zone_2_to_zone_4`: the target region distribution shifts halfway through the query stream.
- `misses`: all targets are absent.

## Smoke run: 2026-11-09

Configuration: `N=8,000`, `Q=400`, five repetitions, deterministic seed `20261109`, one runtime environment. See `../results/benchmark_smoke_2026-11-09.json` for full environment metadata and `../results/benchmark_smoke_2026-11-09.csv` for the table.

| Workload | Linear comparisons/query | Adaptive comparisons/query | Linear / adaptive median time |
|---|---:|---:|---:|
| First region | 991.49 | 991.49 | 0.72× |
| Fourth region | 7,014.99 | 1,029.99 | 5.37× |
| Uniform regions | 3,878.44 | 3,978.44 | 1.02× |
| Shift zone 2 → zone 4 | 5,008.31 | 1,168.31 | 3.42× |
| Misses | 8,000.00 | 8,000.00 | 1.13× |

Interpretation:

- For a strong last-region skew and the tested shift workload, the adaptive region order substantially reduced comparisons and measured time.
- For the first-region workload, the adaptive version was slower because the default order already checks that region first and learning adds computation. For the uniform workload, comparison count was slightly higher while elapsed time was effectively similar in this small run; no robust advantage is established.
- Misses still require a full scan. The observed timing difference with identical comparison counts is not a structural improvement and should be treated as timing variation.
- This is a small exploratory run, not a statistically sufficient research evaluation. Larger query counts, many seeds, repeated process runs, hardware controls, and additional baselines are required before claiming performance superiority.
