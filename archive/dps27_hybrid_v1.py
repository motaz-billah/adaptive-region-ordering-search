#!/usr/bin/env python3
"""DPS27 Hybrid v1.0: exponential-decay zone learning + expected-cost ordering.

Research prototype, not a claim of universal superiority. Python implementation.
Run: python dps27_hybrid_v1.py --n 20000 --queries 500 --repeats 7
"""
from __future__ import annotations
import argparse, itertools, json, math, random, statistics, time
from dataclasses import dataclass
from typing import Iterable

ZONES = 4


def make_bounds(n: int):
    return [(z*n//ZONES, (z+1)*n//ZONES) for z in range(ZONES)]


def zone_for_index(index: int, bounds):
    for z, (a, b) in enumerate(bounds):
        if a <= index < b:
            return z
    raise ValueError(f"index {index} outside zones")


def linear_search(arr, target):
    comparisons = 0
    for i, value in enumerate(arr):
        comparisons += 1
        if value == target:
            return i, comparisons
    return -1, comparisons


class ZoneSearchBase:
    name = "base"
    def __init__(self, n: int):
        self.n = n
        self.bounds = make_bounds(n)
        self.order = [0, 1, 2, 3]

    def search(self, arr, target):
        if len(arr) != self.n:
            raise ValueError("array length mismatch")
        comparisons = 0
        for z in self.order:
            a, b = self.bounds[z]
            for i in range(a, b):
                comparisons += 1
                if arr[i] == target:
                    return i, comparisons
        return -1, comparisons


class DPS27Frequency(ZoneSearchBase):
    name = "DPS27_Frequency"
    def __init__(self, n):
        super().__init__(n)
        self.hits = [0, 0, 0, 0]

    def learn_index(self, index):
        if index >= 0:
            self.hits[zone_for_index(index, self.bounds)] += 1

    def train(self, arr, queries):
        t0 = time.perf_counter_ns()
        for q in queries:
            idx, _ = self.search(arr, q)
            self.learn_index(idx)
        self.order = sorted(range(4), key=lambda z: (-self.hits[z], z))
        return (time.perf_counter_ns()-t0)/1e6


class DPS27ExpectedCost(ZoneSearchBase):
    name = "DPS27_ExpectedCost"
    def __init__(self, n):
        super().__init__(n)
        self.hits = [0, 0, 0, 0]

    def optimize(self, weights):
        total = sum(weights)
        if total <= 0:
            self.order = [0, 1, 2, 3]
            return
        probs = [w/total for w in weights]
        best_order, best_cost = None, float("inf")
        for perm in itertools.permutations(range(4)):
            preceding = 0
            cost = 0.0
            for z in perm:
                a, b = self.bounds[z]
                size = b-a
                cost += probs[z] * (preceding + (size+1)/2)
                preceding += size
            if cost < best_cost - 1e-12:
                best_cost, best_order = cost, perm
        self.order = list(best_order)

    def train(self, arr, queries):
        t0 = time.perf_counter_ns()
        for q in queries:
            idx, _ = self.search(arr, q)
            if idx >= 0:
                self.hits[zone_for_index(idx, self.bounds)] += 1
        self.optimize(self.hits)
        return (time.perf_counter_ns()-t0)/1e6


class DPS27Hybrid(ZoneSearchBase):
    name = "DPS27_Hybrid"
    def __init__(self, n, decay=0.95, min_improvement=0.0):
        super().__init__(n)
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be strictly between 0 and 1")
        if min_improvement < 0:
            raise ValueError("min_improvement must be >= 0")
        self.decay = decay
        self.min_improvement = min_improvement
        self.weights = [0.0]*4
        self.updates = 0
        self.adaptations = 0
        self.last_expected_cost = None

    def optimize(self):
        total = sum(self.weights)
        if total <= 0:
            self.order = [0, 1, 2, 3]
            self.last_expected_cost = None
            return
        probs = [w/total for w in self.weights]
        def cost(order):
            preceding, result = 0, 0.0
            for z in order:
                a, b = self.bounds[z]
                size = b-a
                result += probs[z]*(preceding+(size+1)/2)
                preceding += size
            return result
        current_cost = cost(self.order)
        best_order = min(itertools.permutations(range(4)), key=cost)
        best_cost = cost(best_order)
        improvement = (current_cost-best_cost)/current_cost if current_cost else 0.0
        self.last_expected_cost = best_cost
        if tuple(best_order) != tuple(self.order) and improvement > self.min_improvement:
            self.order = list(best_order)
            self.adaptations += 1

    def observe_index(self, index):
        # Decay every zone on each successful observation, then reward its zone.
        if index < 0:
            return
        self.weights = [w*self.decay for w in self.weights]
        self.weights[zone_for_index(index, self.bounds)] += 1.0
        self.updates += 1
        self.optimize()

    def train(self, arr, queries):
        t0 = time.perf_counter_ns()
        for q in queries:
            idx, _ = self.search(arr, q)
            self.observe_index(idx)
        return (time.perf_counter_ns()-t0)/1e6

    def search_and_learn(self, arr, target):
        idx, comps = self.search(arr, target)
        self.observe_index(idx)
        return idx, comps


def make_queries(arr, scenario, count, seed):
    rng = random.Random(seed)
    n = len(arr)
    bounds = make_bounds(n)
    def in_zone(z):
        a,b = bounds[z]
        if a == b:
            return ("__missing_zone__", z)
        return arr[rng.randrange(a,b)]
    out=[]
    for k in range(count):
        if scenario.startswith("zone_"):
            out.append(in_zone(int(scenario[-1])-1))
        elif scenario == "uniform":
            out.append(arr[rng.randrange(n)] if n else "__missing__")
        elif scenario == "misses":
            out.append(("__definitely_missing__", k))
        elif scenario == "uniform_20pct_misses":
            out.append(("__missing__", k) if rng.random()<0.2 else (arr[rng.randrange(n)] if n else "__missing__"))
        elif scenario == "alternating_zones":
            out.append(in_zone(k%4))
        elif scenario == "shift_1_to_4":
            out.append(in_zone(0 if k<count//2 else 3))
        elif scenario == "shift_4_to_1":
            out.append(in_zone(3 if k<count//2 else 0))
        else:
            raise ValueError(scenario)
    return out


def validate(arr, target, result):
    idx, comps = result
    assert 0 <= comps <= len(arr)
    if idx == -1:
        assert not any(v == target for v in arr), (target, idx)
        assert comps == len(arr)
    else:
        assert 0 <= idx < len(arr) and arr[idx] == target


def timed_batch(fn, arr, queries, repeats):
    samples=[]
    # Warm-up outside measurements.
    for q in queries[:min(20,len(queries))]:
        fn(arr,q)
    for _ in range(repeats):
        t0=time.perf_counter_ns()
        for q in queries:
            fn(arr,q)
        samples.append((time.perf_counter_ns()-t0)/1e6)
    return statistics.median(samples), samples


def run(args):
    rng=random.Random(args.seed)
    arr=list(range(args.n)); rng.shuffle(arr)
    scenarios=["zone_1","zone_2","zone_3","zone_4","uniform","uniform_20pct_misses","misses","alternating_zones","shift_1_to_4","shift_4_to_1"]
    training=make_queries(arr, f"zone_{args.train_zone}", args.train_queries, args.seed+1)
    models=[DPS27Frequency(args.n),DPS27ExpectedCost(args.n),DPS27Hybrid(args.n,args.decay,args.min_improvement)]
    train_cost={m.name:m.train(arr,training) for m in models}
    results=[]
    for si,scenario in enumerate(scenarios):
        queries=make_queries(arr,scenario,args.queries,args.seed+100+si)
        # Separate, identical fresh model per scenario so scenario order cannot contaminate results.
        freq=DPS27Frequency(args.n); freq.train(arr,training)
        exp=DPS27ExpectedCost(args.n); exp.train(arr,training)
        hyb=DPS27Hybrid(args.n,args.decay,args.min_improvement); hyb.train(arr,training)
        comp_sums={"Linear":0,"DPS27_Frequency":0,"DPS27_ExpectedCost":0,"DPS27_Hybrid":0}
        # Untimed correctness/comparison pass. Hybrid learns online after each successful query.
        for q in queries:
            lr=linear_search(arr,q); fr=freq.search(arr,q); er=exp.search(arr,q); hr=hyb.search_and_learn(arr,q)
            for r in (lr,fr,er,hr): validate(arr,q,r)
            comp_sums["Linear"]+=lr[1]; comp_sums[freq.name]+=fr[1]; comp_sums[exp.name]+=er[1]; comp_sums[hyb.name]+=hr[1]
        # Timings use independent fresh models; Hybrid timing includes its online update/optimizer cost.
        def fresh_frequency():
            m=DPS27Frequency(args.n); m.train(arr,training); return m
        def fresh_expected():
            m=DPS27ExpectedCost(args.n); m.train(arr,training); return m
        def fresh_hybrid():
            m=DPS27Hybrid(args.n,args.decay,args.min_improvement); m.train(arr,training); return m
        # Rebuild per timing pass to avoid warm-up mutating the adaptive model's state.
        timing={}
        for name in ["Linear","DPS27_Frequency","DPS27_ExpectedCost","DPS27_Hybrid"]:
            medians=[]
            for rep in range(args.repeats):
                if name=="Linear": fn=linear_search
                elif name=="DPS27_Frequency": fn=fresh_frequency().search
                elif name=="DPS27_ExpectedCost": fn=fresh_expected().search
                else: fn=fresh_hybrid().search_and_learn
                # warmup intentionally omitted here: adaptive warm-up would change state.
                t0=time.perf_counter_ns()
                for q in queries: fn(arr,q)
                medians.append((time.perf_counter_ns()-t0)/1e6)
            timing[name]=statistics.median(medians)
        row={"scenario":scenario,"queries":len(queries),"training_zone":args.train_zone,"training_queries":args.train_queries,
             "avg_comparisons":{k:round(v/len(queries),3) for k,v in comp_sums.items()},
             "median_batch_ms":{k:round(v,4) for k,v in timing.items()},
             "speedup_vs_linear":{k:round(timing["Linear"]/v,3) for k,v in timing.items()},
             "hybrid_final_order":[z+1 for z in hyb.order],"hybrid_final_weights":[round(w,4) for w in hyb.weights],
             "hybrid_adaptations":hyb.adaptations}
        results.append(row)
    summary={"training_ms":{k:round(v,4) for k,v in train_cost.items()},"scenarios":len(results),
             "warning":"Exploratory Python benchmark. Training excluded from test-phase speedups; Hybrid timed path includes online decay and re-optimization. Validate with larger runs and a compiled implementation before publication."}
    return {"config":vars(args),"summary":summary,"results":results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=20000)
    p.add_argument('--queries',type=int,default=500)
    p.add_argument('--train-queries',type=int,default=500)
    p.add_argument('--train-zone',type=int,choices=range(1,5),default=2)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--decay',type=float,default=0.95)
    p.add_argument('--min-improvement',type=float,default=0.0)
    p.add_argument('--seed',type=int,default=2701)
    p.add_argument('--output',default='dps27_hybrid_results.json')
    args=p.parse_args()
    if args.n<0 or args.queries<1 or args.train_queries<0 or args.repeats<1:
        p.error('n must be >=0; queries/repeats >=1; train-queries >=0')
    report=run(args)
    with open(args.output,'w',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
    print(f"\nSaved: {args.output}")

if __name__=='__main__': main()
