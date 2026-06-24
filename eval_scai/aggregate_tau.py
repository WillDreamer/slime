import json, glob, os, re, math, collections, statistics

# ---------- tau1 ----------
print("="*70); print("TAU1 (training-aligned, temp0, mean over runs)"); print("="*70)
t1 = collections.defaultdict(lambda: collections.defaultdict(list)); ns = {}
for f in glob.glob("/home/ec2-user/slime/eval_scai/results/*/run*/tau1_*.json"):
    if "SMOKE" in f: continue
    model = f.split("results/")[1].split("/")[0]; env = f.split("tau1_")[1][:-5]
    d = json.load(open(f))["summary"]; t1[model][env].append(d["accuracy"]); ns[(model,env)] = d["n"]
order = ["qwen-8b-base","Qwen3-8B-Base-Math","Qwen3-8B-Base-Math-SeaSFT-Search","Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau"]
for m in order:
    for e in ["retail","airline"]:
        a = t1.get(m,{}).get(e)
        if not a: continue
        mn = statistics.mean(a)*100; sd = statistics.stdev(a)*100 if len(a)>1 else 0
        print(f"  {m:46s} {e:8s} {mn:5.1f}% ± {sd:4.1f}  (runs={len(a)}, n={ns[(m,e)]})")

# ---------- tau2 (official) ----------
print("="*70); print("TAU2 (official tau2-bench, pass^k); sims grouped per model/domain"); print("="*70)
def passhatk(by_task, k):
    # sierra pass^k: avg over tasks of C(c,k)/C(n,k), c=#pass, n=#trials
    vals=[]
    for trials in by_task.values():
        n=len(trials); c=sum(1 for r in trials if r>0)
        if n<k: continue
        vals.append(math.comb(c,k)/math.comb(n,k))
    return (sum(vals)/len(vals)) if vals else None

agg = collections.defaultdict(lambda: collections.defaultdict(list))  # (model,dom)-> task_id -> [rewards]
sims_done = collections.Counter()
for d in sorted(glob.glob("/home/ec2-user/tau2-bench/data/simulations/tau2off_*")):
    rj = os.path.join(d,"results.json")
    if not os.path.exists(rj): continue
    try: data = json.load(open(rj))
    except: continue
    m = re.match(r"tau2off_(.+)_(retail|airline|telecom)_run(\d)", os.path.basename(d))
    if not m: continue
    model,dom = m.group(1), m.group(2)
    for s in data.get("simulations",[]):
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None: continue
        agg[(model,dom)][s.get("task_id")].append(float(r))
        sims_done[(model,dom)] += 1
hdr = f"  {'model':46s} {'dom':8s} {'sims':6s} {'pass^1':7s} {'pass^2':7s} {'pass^4':7s}"
print(hdr)
for m in order:
    for dom in ["retail","airline","telecom"]:
        k=(m,dom)
        if k not in agg: continue
        bt=agg[k]
        p1=passhatk(bt,1); p2=passhatk(bt,2); p4=passhatk(bt,4)
        f=lambda x: f"{100*x:5.1f}%" if x is not None else "  -  "
        print(f"  {m:46s} {dom:8s} {sims_done[k]:6d} {f(p1):7s} {f(p2):7s} {f(p4):7s}")
