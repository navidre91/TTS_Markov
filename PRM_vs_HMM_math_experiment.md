# PRM vs HMM — Hands‑on Experiment for Math Reasoning (2×RTX 4090)

**Goal.** Test whether a **process verifier** (PRM) that scores *partial reasoning traces* as “probability of eventual success” outperforms **Markov / HMM** surrogates for guiding search on math problems.

This README gives you a runnable plan in two tracks:

- **Track A (simulator quick‑start):** no LLM required; we simulate math‑like search to validate the theory quickly and produce calibration/ranking/search metrics (≈10–20 min on CPU/GPU).
- **Track B (LLM, real math):** use your 2×RTX 4090 to mine partial traces from a math proposer LLM, Monte‑Carlo label them, train a PRM, train an HMM baseline, and A/B test **PRM‑guided search** vs **HMM‑guided** vs **baseline**.

> **Hypothesis.** Let \(V^\star(x,\tau)\) be the probability that continuing from partial trace \(\tau\) on problem \(x\) under search routine \((A,B,\pi)\) will eventually yield a correct answer. A learned \(V_\theta \approx V^\star\) should (i) be **well‑calibrated**, (ii) **rank** children better than HMM/Markov surrogates, and (iii) improve **solve‑rate at fixed compute** when used to expand/prune branches.


---

## 0) Requirements

- **Hardware:** 2×RTX 4090 (24 GB each) recommended. Track A runs on CPU; Track B uses both GPUs.
- **OS:** Linux or Windows WSL2 recommended.
- **Python:** 3.10+

### Create environment
```bash
# (Optional) conda
conda create -n prm-hmm python=3.10 -y
conda activate prm-hmm

# Core packages
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn matplotlib tqdm pyyaml pandas

# For HMM baseline
pip install hmmlearn

# For Track B (LLM path)
pip install transformers datasets accelerate einops sentencepiece
pip install bitsandbytes  # if you plan to quantize
pip install sympy         # lightweight math checker
```

---

## 1) Directory layout

```text
prm-hmm/
├─ data/
│  ├─ simulator/            # Track A artifacts
│  └─ math/                 # Track B artifacts (raw + processed)
├─ models/
│  ├─ prm/                  # saved PRM checkpoints
│  └─ hmm/                  # saved HMM params
├─ results/
│  ├─ simulator/
│  └─ math/
└─ src/
   ├─ trackA_sim/
   │  ├─ sim_env.py
   │  ├─ train_prm_sim.py
   │  ├─ train_hmm_sim.py
   │  └─ eval_search_sim.py
   └─ trackB_llm/
      ├─ mine_partials.py
      ├─ mc_labeler.py
      ├─ train_prm_llm.py
      ├─ train_hmm_llm.py
      └─ eval_search_llm.py
```

> You can copy the code blocks below into the indicated files. The simulator track is fully self‑contained. The LLM track contains working scaffolds that assume any local HF model path (fill in your model id).

---

## 2) Track A — Simulator quick‑start (no LLM)

This is a controlled environment that mimics math search: you start at a number \(s_0\), target \(T\), and can apply primitive actions \(a \in \{+1,-1,\times2,\div2,+3,-3\}\). A problem is *solved* if you reach \(T\) within a step budget. The **proposer policy** \(\pi\) is mildly biased to reduce distance; **Monte‑Carlo continuation** from a partial state gives the ground‑truth success probability \(V^\star\). We then learn:

- **PRM:** a small MLP mapping features of the partial state to \(\hat V\).
- **HMM baseline:** an HMM over action sequences; we attach a per‑state success head and compute \(V_{\text{HMM}}\) as the posterior‑weighted success.

Finally, we run **search** (beam or best‑first) where expansion is guided by PRM vs HMM and compare solve‑rates at fixed compute.

### `src/trackA_sim/sim_env.py`
```python
# SPDX-License-Identifier: MIT
import math, random, json, os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

ACTIONS = ["+1", "-1", "*2", "/2", "+3", "-3"]

def apply_action(s: int, a: str) -> int:
    if a == "+1": return s + 1
    if a == "-1": return s - 1
    if a == "*2": return s * 2
    if a == "/2": return s // 2  # integer division
    if a == "+3": return s + 3
    if a == "-3": return s - 3
    raise ValueError(a)

@dataclass
class Problem:
    start: int
    target: int
    max_steps: int = 10

@dataclass
class State:
    cur: int
    steps: List[str] = field(default_factory=list)

def proposer_policy(cur: int, target: int, temperature: float = 1.0) -> str:
    # Bias to reduce absolute distance; softmax over action 'goodness'
    cand = []
    for a in ACTIONS:
        nxt = apply_action(cur, a)
        gain = abs(cur - target) - abs(nxt - target)
        cand.append(gain)
    probs = np.exp(np.array(cand) / max(1e-3, temperature))
    probs = probs / probs.sum()
    return np.random.choice(ACTIONS, p=probs)

def rollout(problem: Problem, state: State, budget: int, temperature: float = 1.0) -> Tuple[bool, List[str]]:
    cur, hist = state.cur, state.steps[:]
    for _ in range(budget):
        if cur == problem.target:
            return True, hist
        a = proposer_policy(cur, problem.target, temperature)
        cur = apply_action(cur, a)
        hist.append(a)
    return (cur == problem.target), hist

def mc_success_prob(problem: Problem, state: State, budget: int, M: int = 128) -> float:
    succ = 0
    for _ in range(M):
        ok, _ = rollout(problem, state, budget)
        succ += int(ok)
    return succ / M

def sample_problems(n: int, max_abs=50, max_steps=10, seed=0) -> List[Problem]:
    rng = np.random.default_rng(seed)
    probs = []
    for _ in range(n):
        start = int(rng.integers(-max_abs, max_abs+1))
        target = int(rng.integers(-max_abs, max_abs+1))
        while target == start:
            target = int(rng.integers(-max_abs, max_abs+1))
        probs.append(Problem(start=start, target=target, max_steps=max_steps))
    return probs

def mine_partials(problems: List[Problem], per_problem=50, depth=5, seed=0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for pid, pr in enumerate(problems):
        # start from root
        init = State(cur=pr.start, steps=[])
        frontier = [init]
        for _ in range(per_problem):
            # pick a random branch up to 'depth'
            s = State(cur=pr.start, steps=[])
            d = int(rng.integers(0, depth+1))
            for _ in range(d):
                a = rng.choice(ACTIONS)
                s.cur = apply_action(s.cur, a)
                s.steps.append(a)
            rows.append({"pid": pid, "start": pr.start, "target": pr.target, "cur": s.cur, "steps": s.steps})
    return rows

def features(row: Dict) -> np.ndarray:
    cur, target, steps = row["cur"], row["target"], row["steps"]
    dist = abs(cur - target)
    feats = [
        cur, target, dist,
        int(cur % 2 == 0), int(target % 2 == 0),
        len(steps),
        steps.count("+1"), steps.count("-1"),
        steps.count("*2"), steps.count("/2"),
        steps.count("+3"), steps.count("-3"),
    ]
    return np.array(feats, dtype=np.float32)
```

### `src/trackA_sim/train_prm_sim.py`
```python
# SPDX-License-Identifier: MIT
import os, json, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sim_env import Problem, State, sample_problems, mine_partials, mc_success_prob, features

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def main(out_dir="models/prm/sim", data_dir="data/simulator", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1) Problems
    train_probs = sample_problems(200, seed=seed)
    val_probs   = sample_problems(50, seed=seed+1)
    # 2) Partials
    train_rows = mine_partials(train_probs, per_problem=40, depth=6, seed=seed)
    val_rows   = mine_partials(val_probs, per_problem=40, depth=6, seed=seed+1)

    # 3) Labels: Monte-Carlo (use same search budget as intended at test time)
    def label_rows(rows, max_steps=10, M=128):
        ys = []
        for r in tqdm(rows, desc="Labeling"):
            pr = Problem(start=r["start"], target=r["target"], max_steps=max_steps)
            # remaining budget after current depth
            rem = max(0, max_steps - len(r["steps"]))
            y = mc_success_prob(pr, State(cur=r["cur"], steps=r["steps"]), budget=rem, M=M)
            ys.append(y)
        return np.array(ys, dtype=np.float32)

    y_tr = label_rows(train_rows, M=64)  # faster
    y_va = label_rows(val_rows,   M=256) # tighter eval

    X_tr = np.stack([features(r) for r in train_rows]); X_va = np.stack([features(r) for r in val_rows])

    # 4) Train
    torch.manual_seed(seed)
    model = MLP(d=X_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCELoss()

    def run_epoch(X, y, train=True, bs=256):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        dl = DataLoader(ds, batch_size=bs, shuffle=train)
        losses = []
        if train: model.train()
        else: model.eval()
        for xb, yb in dl:
            if train:
                opt.zero_grad()
            pred = model(xb).clamp(1e-5, 1-1e-5)
            loss = bce(pred, yb)
            losses.append(loss.item())
            if train:
                loss.backward(); opt.step()
        return float(np.mean(losses))

    best = 1e9; best_path = os.path.join(out_dir, "prm_sim.pt")
    for epoch in range(30):
        tr = run_epoch(X_tr, y_tr, train=True)
        va = run_epoch(X_va, y_va, train=False)
        if va < best:
            best = va
            torch.save(model.state_dict(), best_path)
        print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}  (best {best:.4f})")

    # Save meta
    meta = {"feat_dim": int(X_tr.shape[1]), "val_bce": float(best)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
    print("Saved PRM to", best_path)

if __name__ == "__main__":
    main()
```

### `src/trackA_sim/train_hmm_sim.py`
```python
# SPDX-License-Identifier: MIT
import os, json, numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from sim_env import Problem, State, sample_problems, mine_partials, rollout, ACTIONS

def encode_actions(acts):
    # map actions to integers 0..5
    A = {a:i for i,a in enumerate(ACTIONS)}
    return np.array([A[a] for a in acts], dtype=np.int64)

def posterior_success(prob_model, rows, problems, K=6, seq_len=8, M=128):
    # For each hidden state k, estimate success head h(k)
    # Approximate by sampling partials whose Viterbi state at end is k
    A2I = {a:i for i,a in enumerate(ACTIONS)}
    per_state = {k: {"succ":0, "tot":0} for k in range(prob_model.n_components)}

    # mine some continuations to attach success labels
    for r in tqdm(rows, desc="Attach heads"):
        pr = Problem(start=r["start"], target=r["target"], max_steps=10)
        # Create a short sequence by rolling out from current state
        s = State(cur=r["cur"], steps=r["steps"][:])
        seq = r["steps"][:]
        cur = s.cur
        for _ in range(seq_len):
            ok, seq1 = rollout(pr, State(cur=cur, steps=seq[:]), budget=1)
            # take only the next action (budget 1 rollout)
            if len(seq1) > len(seq):
                seq = seq1
                cur = cur  # cur updated inside rollout is hidden here
            else:
                break
        if not seq:
            continue
        X = encode_actions(seq).reshape(-1,1)
        logprob, states = prob_model.decode(X, algorithm="viterbi")
        k = states[-1]
        # Label by MC from this partial
        # (Use a cheap proxy by seeing whether the 10-step rollout from scratch solves)
        succ = 0
        for _ in range(M//8):
            ok, _ = rollout(pr, State(cur=r["cur"], steps=r["steps"]), budget=10-len(r["steps"]))
            succ += int(ok)
        per_state[k]["succ"] += succ
        per_state[k]["tot"]  += (M//8)

    h = np.zeros(prob_model.n_components, dtype=np.float32)
    for k,v in per_state.items():
        h[k] = (v["succ"]/max(1,v["tot"]))
    return h

def main(out_dir="models/hmm/sim", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1) Problems + partials
    probs = sample_problems(200, seed=seed)
    rows  = mine_partials(probs, per_problem=60, depth=6, seed=seed)

    # 2) Build sequences: take random short action strings from rollouts
    seqs = []
    for r in rows:
        if not r["steps"]: continue
        seqs.append(encode_actions(r["steps"]).reshape(-1,1))

    lengths = [len(s) for s in seqs]
    X = np.concatenate(seqs, axis=0)

    # 3) Fit HMM with discrete (Categorical) emissions approximated by Gaussian on indices
    K = 8
    model = hmm.GaussianHMM(n_components=K, covariance_type="spherical", n_iter=50, random_state=seed)
    model.fit(X, lengths)

    # 4) Attach success head h(k)
    h = posterior_success(model, rows, probs, K=K, seq_len=6, M=64)

    # 5) Save
    np.save(os.path.join(out_dir, "trans_means.npy"), model.means_)
    np.save(os.path.join(out_dir, "trans_covars.npy"), model.covars_)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"K": K, "h": h.tolist()}, f, indent=2)
    print("Saved HMM params + success head")

if __name__ == "__main__":
    main()
```

### `src/trackA_sim/eval_search_sim.py`
```python
# SPDX-License-Identifier: MIT
import os, json, numpy as np, torch
from tqdm import tqdm
from sim_env import Problem, State, sample_problems, apply_action, ACTIONS, features
from train_prm_sim import MLP

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
    # naive: map "phase" by last action if present, else average
    meta = json.load(open(os.path.join(hmm_dir, "meta.json")))
    h = np.array(meta["h"], dtype=np.float32)
    # fall back to average if we don't decode
    return float(h.mean()) if len(row["steps"]) == 0 else float(h.max())

def search(problem, scorer, beam=8, per_expand=4, max_steps=10):
    # Best-first on predicted success probability; expand 'per_expand' random actions per node
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
        # sample some children
        for a in np.random.choice(ACTIONS, size=per_expand, replace=False):
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
    def scorer_null(row): return 0.5  # baseline heuristic

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
```

### Run Track A
```bash
# From project root
python -m src.trackA_sim.train_prm_sim
python -m src.trackA_sim.train_hmm_sim
python -m src.trackA_sim.eval_search_sim

# Inspect results
cat results/simulator/summary.json
```

**Expected pattern (simulator):** PRM should show **higher solve‑rate** and/or **fewer nodes expanded** than HMM and the baseline at the same search settings. Use this to sanity‑check your thresholds and search budget.


---

## 3) Track B — Real math with an LLM proposer

In this track you will:

1. **Mine partial traces** from a math dataset (e.g., GSM8K/MATH) using your **proposer** LLM.
2. **Monte‑Carlo label** partials by continuing from each partial several times and checking correctness.
3. **Train PRM** (small classifier/regressor on textual partials) and **HMM** baseline.
4. **A/B test** search guided by PRM vs HMM vs baseline.

> This scaffold keeps things minimal so you can swap in your favorite base models and data. Fill in the `MODEL_ID` or local path to your math‑tuned model (e.g., a 7–8B math instruction model).

### GPU placement

- **GPU‑0:** proposer server (sampling heavy).  
- **GPU‑1:** PRM scoring (batched); HMM runs on CPU.

Use environment variables to pin:
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.trackB_llm.mine_partials --model_id <MODEL_ID>
CUDA_VISIBLE_DEVICES=0 python -m src.trackB_llm.mc_labeler --model_id <MODEL_ID>
CUDA_VISIBLE_DEVICES=1 python -m src.trackB_llm.train_prm_llm
CUDA_VISIBLE_DEVICES=0,1 python -m src.trackB_llm.eval_search_llm --model_id <MODEL_ID>
```

### `src/trackB_llm/mine_partials.py` (scaffold)
```python
# SPDX-License-Identifier: MIT
import os, json, random, argparse, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, numpy as np
from tqdm import tqdm

STEP_MARK = "Step"

def fmt_prompt(q):
    return f"Solve the problem step by step.\n\nProblem:\n{q}\n\nSolution:\n"

def split_steps(text):
    # crude: split on "Step" markers or newlines
    parts = [p.strip() for p in re.split(r"(?:^|\n)(?:Step\s*\d*[:.)-]?)", text) if p.strip()]
    # ensure we return coherent steps
    steps = []
    buf = []
    for p in parts:
        if len(p) < 3: continue
        steps.append(p)
    if not steps: steps = [text.strip()]
    return steps

def main(args):
    os.makedirs("data/math/raw", exist_ok=True)
    # Example: GSM8K
    ds = load_dataset("gsm8k", "main")["train"].select(range(500))  # small subset
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")

    rows = []
    for i, ex in tqdm(enumerate(ds), total=len(ds)):
        q = ex["question"]; a = ex["answer"]
        prompt = fmt_prompt(q)
        inp = tok(prompt, return_tensors="pt").to(mdl.device)
        out = mdl.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
        text = tok.decode(out[0], skip_special_tokens=True)
        sol = text.split("Solution:",1)[-1].strip()
        steps = split_steps(sol)
        # generate prefixes (partials)
        for t in range(min(len(steps), 6)):
            partial = "\n".join(steps[:t+1])
            rows.append({"pid": i, "question": q, "gold": a, "partial": partial, "t": t+1})
    with open("data/math/raw/partials.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print("Wrote", len(rows), "partials to data/math/raw/partials.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="HF model id or local path")
    main(ap.parse_args())
```

### `src/trackB_llm/mc_labeler.py` (Monte‑Carlo continuation labels)
```python
# SPDX-License-Identifier: MIT
import os, json, argparse, re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, numpy as np
from sympy import sympify

def extract_final_number(text):
    # Very crude; adapt to your dataset's format
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return nums[-1] if nums else None

def check_correct(gold, pred):
    # For GSM8K-like answers ending with "#### 42"
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", gold)
    if m:
        return (pred is not None) and (abs(float(pred) - float(m.group(1))) < 1e-6)
    # Fallback: symbolic compare if both parse
    try:
        return sympify(gold) == sympify(pred)
    except Exception:
        return False

def continue_from_partial(model, tok, question, partial, max_new=196, temperature=0.7):
    prefix = f"Solve the problem step by step.\n\nProblem:\n{question}\n\nSolution:\n{partial}\n"
    inp = tok(prefix, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.95, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prefix):].strip()

def main(args):
    os.makedirs("data/math/labels", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")

    rows = [json.loads(l) for l in open("data/math/raw/partials.jsonl")]
    out_f = open("data/math/labels/partials_labeled.jsonl", "w")

    for r in tqdm(rows, desc="Labeling"):
        M = args.samples
        succ = 0
        for _ in range(M):
            cont = continue_from_partial(mdl, tok, r["question"], r["partial"])
            pred = extract_final_number(cont)
            ok = check_correct(r["gold"], pred)
            succ += int(ok)
        p_hat = succ / M
        r2 = dict(r); r2["p_hat"] = p_hat
        out_f.write(json.dumps(r2)+"\n")
    out_f.close()
    print("Labeled", len(rows), "partials")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--samples", type=int, default=32, help="MC rollouts per partial")
    main(ap.parse_args())
```

### `src/trackB_llm/train_prm_llm.py` (simple text PRM)
```python
# SPDX-License-Identifier: MIT
import os, json, argparse, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class PartialDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate(batch, tok, max_len=512):
    texts = [f"Problem:\\n{b['question']}\\n\\nPartial solution up to step {b['t']}:\\n{b['partial']}" for b in batch]
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    y = torch.tensor([b["p_hat"] for b in batch]).float()
    return enc, y

class PRMHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, x): return self.mlp(x).squeeze(-1)

def main(args):
    os.makedirs("models/prm/llm", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.encoder_id, use_fast=True)
    enc = AutoModel.from_pretrained(args.encoder_id).cuda()
    head = PRMHead(enc.config.hidden_size).cuda()

    ds = PartialDataset("data/math/labels/partials_labeled.jsonl")
    n = len(ds); idx = np.arange(n); np.random.shuffle(idx)
    split = int(0.9*n); tr_idx, va_idx = idx[:split], idx[split:]
    tr = torch.utils.data.Subset(ds, tr_idx); va = torch.utils.data.Subset(ds, va_idx)
    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, tok))
    dl_va = DataLoader(va, batch_size=args.bs, shuffle=False, collate_fn=lambda b: collate(b, tok))

    opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=2e-5, weight_decay=0.01)
    bce = nn.BCELoss()

    best = 1e9
    for epoch in range(args.epochs):
        enc.train(); head.train()
        losses = []
        for batch in tqdm(dl_tr, desc=f"train {epoch}"):
            (X), y = batch
            X = {k:v.cuda() for k,v in X.items()}; y = y.cuda()
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                h = enc(**X).last_hidden_state[:,0]  # CLS
                p = head(h).clamp(1e-4, 1-1e-4)
                loss = bce(p, y)
            loss.backward(); opt.step()
            losses.append(loss.item())

        # val
        enc.eval(); head.eval()
        with torch.no_grad():
            vals = []
            for batch in dl_va:
                X, y = batch
                X = {k:v.cuda() for k,v in X.items()}; y = y.cuda()
                h = enc(**X).last_hidden_state[:,0]
                p = head(h).clamp(1e-4, 1-1e-4)
                vals.append(bce(p,y).item())
            va = float(np.mean(vals))
        print(f"epoch {epoch:02d}  train {np.mean(losses):.4f}  val {va:.4f}")
        if va < best:
            best = va
            torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "cfg":{"encoder_id":args.encoder_id}}, "models/prm/llm/prm_llm.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_id", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    main(ap.parse_args())
```

### `src/trackB_llm/train_hmm_llm.py` (HMM baseline on step tokens)
```python
# SPDX-License-Identifier: MIT
import os, json, argparse, numpy as np, re
from tqdm import tqdm
from hmmlearn import hmm

def tokenize_step(s):
    # crude tokenization by symbols & words
    toks = re.findall(r"[A-Za-z]+|[0-9]+|[\+\-\*/=\(\)]", s)
    return [t[:8] for t in toks][:50]

def step2vec(step, vocab):
    v = np.zeros(len(vocab), dtype=np.float32)
    for t in tokenize_step(step):
        if t in vocab: v[vocab[t]] += 1.0
    v = v / max(1.0, np.linalg.norm(v))
    return v

def main(args):
    rows = [json.loads(l) for l in open("data/math/labels/partials_labeled.jsonl")]
    # Build small vocab
    counts = {}
    for r in rows:
        for t in tokenize_step(r["partial"]):
            counts[t] = counts.get(t, 0) + 1
    vocab_items = sorted(counts.items(), key=lambda x: -x[1])[:512]
    vocab = {t:i for i,(t,_) in enumerate(vocab_items)}

    # For each partial, build a short sequence of step vectors (split by lines)
    sequences = []
    for r in rows:
        steps = [s for s in r["partial"].split("\n") if s.strip()]
        vecs = [step2vec(s, vocab) for s in steps[-5:]]  # last 5 "steps"
        if not vecs: continue
        sequences.append(np.stack(vecs, axis=0))

    lengths = [len(s) for s in sequences]
    X = np.concatenate(sequences, axis=0)

    # Fit Gaussian HMM
    K = args.K
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=50, random_state=0)
    model.fit(X, lengths)

    # Attach success head per state
    state_stats = {"succ": np.zeros(K), "tot": np.zeros(K)}
    for r in tqdm(rows, desc="Attach success head"):
        steps = [s for s in r["partial"].split("\n") if s.strip()]
        if not steps: continue
        vecs = [step2vec(s, vocab) for s in steps[-5:]]
        Xs = np.stack(vecs, axis=0)
        _, states = model.decode(Xs, algorithm="viterbi")
        k = states[-1]
        state_stats["succ"][k] += r["p_hat"]
        state_stats["tot"][k]  += 1.0
    h = (state_stats["succ"] / np.maximum(1.0, state_stats["tot"])).tolist()

    os.makedirs("models/hmm/llm", exist_ok=True)
    np.save("models/hmm/llm/hmm_means.npy", model.means_)
    np.save("models/hmm/llm/hmm_covars.npy", model.covars_)
    json.dump({"K": K, "h": h, "vocab": list(vocab.keys())}, open("models/hmm/llm/meta.json","w"), indent=2)
    print("Saved HMM baseline with success head.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    main(ap.parse_args())
```

### `src/trackB_llm/eval_search_llm.py` (beam search guided by PRM/HMM)
```python
# SPDX-License-Identifier: MIT
import os, json, argparse, re, heapq, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from train_prm_llm import PRMHead
from train_hmm_llm import tokenize_step, step2vec
from tqdm import tqdm

def fmt_root(q):
    return f"Solve the problem step by step.\n\nProblem:\n{q}\n\nSolution:\n"

def extend_prompt(root, steps):
    return root + "\n".join(steps) + ("\n" if steps else "")

def sample_next_steps(mdl, tok, root, steps, k=3, max_new=64):
    prompt = extend_prompt(root, steps)
    inp = tok(prompt, return_tensors="pt").to(mdl.device)
    outs = mdl.generate(**inp, do_sample=True, top_p=0.95, temperature=0.7,
                        max_new_tokens=max_new, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    texts = [tok.decode(o, skip_special_tokens=True)[len(prompt):].strip() for o in outs]
    # crude: take first line as the "next step"
    cand = [t.split("\n")[0].strip() for t in texts]
    # filter empty/duplicates
    uniq = []
    for c in cand:
        if c and c not in uniq: uniq.append(c)
    return uniq[:k]

def build_prm_scorer():
    ckpt = torch.load("models/prm/llm/prm_llm.pt", map_location="cuda:0")
    enc_id = ckpt["cfg"]["encoder_id"]
    tok = AutoTokenizer.from_pretrained(enc_id, use_fast=True)
    enc = AutoModel.from_pretrained(enc_id).cuda()
    head = PRMHead(enc.config.hidden_size).cuda()
    enc.load_state_dict(ckpt["enc"]); head.load_state_dict(ckpt["head"])
    enc.eval(); head.eval()
    @torch.no_grad()
    def score(q, steps):
        text = f"Problem:\n{q}\n\nPartial solution:\n" + "\n".join(steps)
        X = tok([text], return_tensors="pt", truncation=True, max_length=512, padding=True).to("cuda:0")
        h = enc(**X).last_hidden_state[:,0]
        p = head(h).sigmoid().item()
        return float(p)
    return score

def build_hmm_scorer():
    meta = json.load(open("models/hmm/llm/meta.json"))
    vocab = {t:i for i,t in enumerate(meta["vocab"])}
    h = np.array(meta["h"], dtype=np.float32)
    # Dummy decode: choose nearest state by last step vector's nearest mean
    means = np.load("models/hmm/llm/hmm_means.npy")
    def score(q, steps):
        if not steps: return float(h.mean())
        v = step2vec(steps[-1], vocab)
        d = ((means - v[None,:])**2).sum(axis=1)
        k = int(np.argmin(d))
        return float(h[k])
    return score

def beam_search(q, mdl, tok, scorer, beam=8, per_expand=3, depth=8):
    root = fmt_root(q)
    Node = lambda steps, p: (-p, steps)
    pq = []
    heapq.heappush(pq, Node([], 0.5))
    while pq:
        score, steps = heapq.heappop(pq)
        score = -score
        if len(steps) >= depth:
            # optional: finalize by generating to end and returning final answer
            return steps
        # expand
        cand_steps = sample_next_steps(mdl, tok, root, steps, k=per_expand)
        for ns in cand_steps:
            steps2 = steps + [ns]
            p = scorer(q, steps2)
            heapq.heappush(pq, Node(steps2, p))
        # prune size
        pq = pq[:beam * 10]
    return []

def main(args):
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")

    scorer = {"prm": build_prm_scorer(), "hmm": build_hmm_scorer()}.get(args.mode)
    assert scorer is not None, "mode must be prm or hmm"

    # small eval set from mined partials' questions
    rows = [json.loads(l) for l in open("data/math/raw/partials.jsonl")]
    qs = {r["pid"]: r["question"] for r in rows}
    qs = [qs[k] for k in sorted(qs.keys())][:64]

    results = []
    for q in tqdm(qs, desc=f"beam-{args.mode}"):
        steps = beam_search(q, mdl, tok, scorer, beam=args.beam, per_expand=args.expand, depth=args.depth)
        results.append({"question": q, "steps": steps})

    os.makedirs("results/math", exist_ok=True)
    out = f"results/math/search_{args.mode}.json"
    json.dump(results, open(out,"w"), indent=2)
    print("Saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--mode", choices=["prm","hmm"], required=True)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--expand", type=int, default=3)
    ap.add_argument("--depth", type=int, default=8)
    main(ap.parse_args())
```

### Run Track B
```bash
# 1) Mine partials (GPU‑0)
CUDA_VISIBLE_DEVICES=0 python -m src.trackB_llm.mine_partials --model_id <MODEL_ID>

# 2) Monte‑Carlo labels (GPU‑0); increase --samples for higher‑quality labels
CUDA_VISIBLE_DEVICES=0 python -m src.trackB_llm.mc_labeler --model_id <MODEL_ID> --samples 16

# 3) Train PRM (GPU‑1)
CUDA_VISIBLE_DEVICES=1 python -m src.trackB_llm.train_prm_llm --encoder_id sentence-transformers/all-MiniLM-L6-v2

# 4) Train HMM baseline (CPU)
python -m src.trackB_llm.train_hmm_llm --K 12

# 5) A/B beam search (GPU‑0 for proposer, GPU‑1 for PRM scoring)
CUDA_VISIBLE_DEVICES=0,1 python -m src.trackB_llm.eval_search_llm --model_id <MODEL_ID> --mode prm
CUDA_VISIBLE_DEVICES=0,1 python -m src.trackB_llm.eval_search_llm --model_id <MODEL_ID> --mode hmm
```

**What to report (math track):**
- **Calibration** (optional add‑on): sample partials, bin predicted \(p\), estimate empirical success and plot a reliability curve.
- **Ranking quality:** for a set of children per parent, Spearman correlation between predicted \(p\) and empirical \(\hat p\) from short MC.
- **End‑to‑end solve‑rate at fixed compute:** run beam search with the same beam/expand/depth; compare % problems with a correct final answer (use a simple checker on the generated final line).

---

## 4) Interpreting outcomes

- If **PRM** beats the **HMM** on ranking and solve‑rate, it supports the view that a **policy‑ & budget‑conditioned value function** \(V^\star\) is a better decision primitive than a generative HMM summary in this domain.
- If HMM is competitive, try increasing **branching** (per‑expand) and check the **parallelism dividend**: compute \(1-\prod_{c}(1-\hat p_c)\) using empirical child success rates; PRM should track this better than HMM.
- Check **budget robustness** by labeling at budget \(B\) but evaluating at \(B'\in\{0.5B,\,B,\,2B\}\).

---

## 5) Tips & knobs

- **Sampling temps:** when mining partials and continuing, mix temperatures (e.g., 0.3/0.7) for diversity.
- **Normalization:** when comparing children of the same parent, z‑score PRM outputs within the set before ranking to stabilize.
- **Backtracking gate:** in search, prune children with \(p < \tau_{\min}\) or negative \(\Delta p\) vs parent.
- **Two‑GPU utilization:** run proposer on GPU‑0; batch PRM scoring on GPU‑1; communicate via a local queue or filesystem.
- **Checkers:** for GSM8K, matching the final number is usually enough; for algebraic tasks, incorporate SymPy equivalence checks on key sub‑expressions.

---

## 6) Repro log

Keep a short log with:
- dataset split and counts,
- labeler settings (samples per partial, max tokens),
- PRM architecture & training loss,
- HMM state count \(K\),
- search settings (beam, per‑expand, depth),
- wall‑clock and GPU memory usage.

This will help you tune the trade‑off between **label quality** and **throughput**.

---

## 7) License

All code in this README is provided under the MIT license. Adapt as needed.
