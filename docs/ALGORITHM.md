# Algorithm Notes — Adaptive Region Ordering Search

## 1. State

For four contiguous regions, the model stores:

- `weights[i]`: exponentially decayed successful-query weight for region `i`;
- `recent`: a bounded queue of recently observed successful-query regions;
- `prev_hist`: normalized long-term weights before the latest update;
- `pred`: the current normalized estimate of future successful-query region probabilities;
- `order`: the current region visitation permutation.

## 2. Probability estimate

Let `H` be normalized decayed history and `H_prev` the previous normalized history. The trend estimate is:

\[
T_i = \operatorname{normalize}(H_i + \beta(H_i-H_{prev,i}))
\]

Let `R` be the normalized recent-window counts. The long-term blend is:

\[
L_i = \gamma H_i + (1-\gamma)T_i
\]

The predicted probability is:

\[
P_i = \operatorname{normalize}(\alpha R_i + (1-\alpha)L_i)
\]

The current defaults are `decay=0.95`, `window=128`, `alpha=0.5`, `gamma=0.5`, and `beta=0.25`.

## 3. Order selection

For each permutation of the four regions, estimate comparisons assuming successful targets are uniformly distributed inside their selected region. Choose the permutation with minimum expected cost. Since there are four regions, there are 24 permutations.

## 4. Update policy

A successful search identifies the region containing the returned index. The algorithm updates the old histogram, applies decay, increments the observed region, appends the region to the recent window, recalculates predicted probabilities, and selects the lowest-cost region order. A miss increments the miss counter but does not update the probability model.

## 5. Correctness contract

For an unchanged array, the method scans positions in the array and returns only after direct equality with the target. Therefore a returned index always contains the target. If a query is absent, all positions are checked and `None` is returned. With duplicates, the returned matching index depends on the learned region visitation order and is not necessarily the first physical occurrence.

## 6. Open research questions

- Can model-update cost be reduced without losing adaptation quality?
- Should misses contribute to a separate region/difficulty model?
- How should the algorithm react to abrupt query-distribution changes?
- Can confidence or a minimum-improvement threshold prevent costly order oscillations?
- Does the method beat linear search in end-to-end wall-clock time for sustained skewed workloads?
- Which workloads justify it over a hash index or another established structure?


## 7. Complexity detail

For four regions, order selection evaluates 24 permutations and takes constant work with respect to `N`. However, building recent-region counts scans the bounded recent window, so each successful-query learning step is `O(W + 24 * 4)`. A lookup itself may scan up to `N` array positions. The input array should not be mutated while an instance is in use.
