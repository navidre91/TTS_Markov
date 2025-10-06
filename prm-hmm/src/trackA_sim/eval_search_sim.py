# SPDX-License-Identifier: MIT
import os, json, numpy as np, torch
from tqdm import tqdm
from .sim_env import Problem, State, sample_problems, apply_action, ACTIONS, features
from .train_prm_sim import MLP

def load_prm(model_dir="models/prm/sim"):
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    model = MLP(d=meta["feat_dim"])
    model.load_state_dict(torch.load(os.path.join(model_dir, "prm_sim.pt"), map_location="cpu"))
    model.eval()
    return model

def prm_score(model, row):
    x = torch.tensor(features(row)).float().unsqueeze(0)
    with torch.no_grad():
        return float(model(x).item())

def hmm_score(hmm_dir, row):
    meta = json.load(open(os.path.join(hmm_dir, "meta.json")))
    h = np.array(meta["h"], dtype=np.float32)
    return float(h.mean()) if len(row["steps"]) == 0 else float(h.max())

def search(problem, scorer, beam=8, per_expand=4, max_steps=10):
    from heapq import heappush, heappop
    Node = lambda cur, steps, score: (-score, {"cur":cur, "steps":steps})
    pq = []
    root = {"start": problem.start, "target": problem.target, "cur": problem.start, "steps":[]}
    heappush(pq, Node(problem.start, [], scorer(root)))
    seen = 0
    while pq and seen < 10000:
        _, node = heappop(pq)
        seen += 1
        cur, steps = node["cur"], node["steps"]
        if cur == problem.target:
            return True, steps, seen
        if len(steps) >= max_steps:
            continue
        # expand children
        for a in np.random.choice(ACTIONS, size=min(per_expand, len(ACTIONS)), replace=False):
            nxt = apply_action(cur, a)
            child = {"start": problem.start, "target": problem.target, "cur": nxt, "steps": steps+[a]}
            sc = scorer(child)
            heappush(pq, Node(nxt, steps+[a], sc))
        # prune beam
        if len(pq) > beam * 20:
            pq = pq[:beam * 20]
    return False, [], seen

def main(prm_dir="models/prm/sim", hmm_dir="models/hmm/sim"):
    model = load_prm(prm_dir)
    def scorer_prm(row): return prm_score(model, row)
    def scorer_hmm(row): return hmm_score(hmm_dir, row)
    def scorer_null(row): return 0.5

    test_probs = sample_problems(200, seed=123)
    res = {}
    for name, scorer in [("PRM", scorer_prm), ("HMM", scorer_hmm), ("Baseline", scorer_null)]:
        ok, nodes = 0, 0
        for pr in tqdm(test_probs, desc=f"Search {name}"):
            succ, steps, seen = search(pr, scorer, beam=8, per_expand=3, max_steps=10)
            ok += int(succ); nodes += seen
        res[name] = {"solve_rate": ok/len(test_probs), "avg_nodes": nodes/len(test_probs)}
    os.makedirs("results/simulator", exist_ok=True)
    with open("results/simulator/summary.json", "w") as f: json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
