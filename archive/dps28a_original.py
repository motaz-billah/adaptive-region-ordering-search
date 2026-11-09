from pathlib import Path

from collections import deque
from itertools import permutations
import argparse, csv, json, random, statistics, time

def bounds(n,k=4): return [(i*n//k,(i+1)*n//k) for i in range(k)]
def norm(v):
    v=[max(0.0,float(x)) for x in v]; s=sum(v)
    return [x/s for x in v] if s else [1/len(v)]*len(v)
def cost(order,b,p):
    prefix=0; out=0.0
    for z in order:
        a,e=b[z]; size=e-a
        out += p[z]*(prefix+(size+1)/2) if size else 0
        prefix += size
    return out
def best_order(b,p): return min(permutations(range(len(b))),key=lambda o:cost(o,b,p))

class Base:
    def __init__(self,a,k=4):
        self.a=a; self.b=bounds(len(a),k); self.k=k
        self.order=tuple(range(k)); self.successes=0; self.misses=0; self.updates=0
    def _search(self,x):
        c=0
        for z in self.order:
            lo,hi=self.b[z]
            for i in range(lo,hi):
                c+=1
                if self.a[i]==x: return i,c,z
        return None,c,None
    def search(self,x):
        i,c,z=self._search(x)
        if i is None: self.misses+=1
        else: self.successes+=1; self.learn(z)
        return i,c
    def learn(self,z): raise NotImplementedError

class DPS27Hybrid(Base):
    def __init__(self,a,k=4,decay=.95):
        super().__init__(a,k); self.decay=decay
        self.weights=[0.0]*k; self.prob=[1/k]*k
    def learn(self,z):
        self.weights=[self.decay*w for w in self.weights]
        self.weights[z]+=1
        self.prob=norm(self.weights)
        o=best_order(self.b,self.prob)
        if o!=self.order: self.order=o; self.updates+=1

class DPS28A(Base):
    def __init__(self,a,k=4,decay=.95,window=128,alpha=.5,gamma=.5,beta=.25):
        super().__init__(a,k)
        if not 0<decay<1 or window<1 or not 0<=alpha<=1 or not 0<=gamma<=1 or beta<0:
            raise ValueError("Invalid DPS28-A parameter")
        self.decay,self.window,self.alpha,self.gamma,self.beta=decay,window,alpha,gamma,beta
        self.weights=[0.0]*k; self.recent=deque(maxlen=window)
        self.prev_hist=None; self.pred=[1/k]*k
    def predict(self):
        hist=norm(self.weights)
        if self.prev_hist is None: trend=hist
        else:
            raw=[hist[i]+self.beta*(hist[i]-self.prev_hist[i]) for i in range(self.k)]
            trend=norm(raw)
        counts=[0]*self.k
        for z in self.recent: counts[z]+=1
        recent=norm(counts)
        long=[self.gamma*hist[i]+(1-self.gamma)*trend[i] for i in range(self.k)]
        self.pred=norm([self.alpha*recent[i]+(1-self.alpha)*long[i] for i in range(self.k)])
        return self.pred
    def learn(self,z):
        self.prev_hist=norm(self.weights)
        self.weights=[self.decay*w for w in self.weights]
        self.weights[z]+=1
        self.recent.append(z)
        o=best_order(self.b,self.predict())
        if o!=self.order: self.order=o; self.updates+=1

def percentile(x,p):
    s=sorted(x); pos=(len(s)-1)*p/100; lo=int(pos); hi=min(lo+1,len(s)-1); f=pos-lo
    return s[lo]*(1-f)+s[hi]*f
def run(a,qs,cls):
    alg=cls(a); lat=[]; comps=0; t=time.perf_counter()
    for q in qs:
        t0=time.perf_counter(); _,c=alg.search(q); lat.append(time.perf_counter()-t0); comps+=c
    total=time.perf_counter()-t
    return dict(algorithm=cls.__name__,comparisons=comps/len(qs),total_seconds=total,
                median_us=statistics.median(lat)*1e6,p95_us=percentile(lat,95)*1e6,
                successes=alg.successes,misses=alg.misses,order_updates=alg.updates)
def gen_queries(a,n,p1,p2,shift,seed):
    r=random.Random(seed); b=bounds(len(a)); qs=[]
    for j in range(n):
        p=p1 if j<shift else p2
        z=r.choices(range(4),weights=p,k=1)[0]; lo,hi=b[z]
        qs.append(a[r.randrange(lo,hi)])
    return qs
def tests():
    r=random.Random(28)
    for n in range(1,35):
        a=list(range(n)); r.shuffle(a)
        for cls in (DPS27Hybrid,DPS28A):
            x=cls(a)
            for q in range(-2,n+2):
                i,c=x.search(q)
                assert (i is not None and a[i]==q) if q in a else i is None
                assert 0<=c<=n
    for _ in range(200):
        a=[r.randrange(-8,9) for _ in range(r.randrange(1,150))]
        for cls in (DPS27Hybrid,DPS28A):
            x=cls(a)
            for q in set(a)|{1000}:
                i,c=x.search(q)
                assert (i is not None and a[i]==q) if q in a else i is None
    x=DPS28A(list(range(100)))
    for _ in range(1000):
        x.learn(r.randrange(4))
        assert abs(sum(x.pred)-1)<1e-9 and min(x.pred)>=0
        assert set(x.order)=={0,1,2,3}
    return "PASS: exact search, misses, duplicates, probability normalization, valid order"
def main():
    p=argparse.ArgumentParser(); p.add_argument("--n",type=int,default=30000); p.add_argument("--queries",type=int,default=6000); p.add_argument("--seed",type=int,default=28); p.add_argument("--output",default="/mnt/data/dps28a_demo_results.json"); x=p.parse_args()
    status=tests(); r=random.Random(x.seed); a=list(range(x.n)); r.shuffle(a)
    cases={"stable_zone2":([.1,.7,.1,.1],[.1,.7,.1,.1]),
           "shift_zone2_to4":([.1,.7,.1,.1],[.1,.1,.1,.7]),
           "uniform":([.25]*4,[.25]*4)}
    rows=[]
    for name,(p1,p2) in cases.items():
        qs=gen_queries(a,x.queries,p1,p2,x.queries//2,x.seed+1)
        for cls in (DPS27Hybrid,DPS28A): rows.append({"scenario":name,**run(a,qs,cls)})
    payload={"correctness":status,"configuration":{"N":x.n,"queries":x.queries,"seed":x.seed,"regions":4,"lambda":.95,"window":128,"alpha":.5,"gamma":.5,"beta":.25,"note":"Demonstration only; not publication-grade evidence."},"results":rows}
    Path(x.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    csvpath=x.output.rsplit(".",1)[0]+".csv"
    with open(csvpath,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(status); print(json.dumps(payload["configuration"],indent=2))
    for z in rows: print(z)
    print("FILES:",x.output,csvpath)
if __name__=="__main__": main()
