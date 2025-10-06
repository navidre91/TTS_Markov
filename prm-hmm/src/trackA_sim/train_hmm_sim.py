# SPDX-License-Identifier: MIT
import os, json, numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from .sim_env import Problem, State, sample_problems, mine_partials, rollout, ACTIONS, apply_action

def encode_actions(acts):
    A = {a:i for i,a in enumerate(ACTIONS)}
    return np.array([A[a] for a in acts], dtype=np.int64)

def posterior_success(prob_model, rows, K=6, seq_len=8, M=64):
    A2I = {a:i for i,a in enumerate(ACTIONS)}
    per_state = {k: {"succ":0, "tot":0} for k in range(prob_model.n_components)}
    for r in tqdm(rows, desc="Attach heads"):
        pr = Problem(start=r["start"], target=r["target"], max_steps=10)
        # build a short continuation sequence deterministically by greedy distance reduction
        seq = r["steps"][:]
        cur = r["cur"]
        for _ in range(seq_len):
            # greedy single step toward target
            best_a, best_gain = None, -1e9
            for a in ACTIONS:
                nxt = apply_action(cur, a)
                gain = abs(cur - pr.target) - abs(nxt - pr.target)
                if gain > best_gain:
                    best_gain, best_a = gain, a
            cur = apply_action(cur, best_a)
            seq.append(best_a)
        X = encode_actions(seq).reshape(-1,1)
        _, states = prob_model.decode(X, algorithm="viterbi")
        k = states[-1]
        succ = 0
        # small MC
        for _ in range(M):
            ok, _ = rollout(pr, State(cur=r["cur"], steps=r["steps"]), budget=10-len(r["steps"]))
            succ += int(ok)
        per_state[k]["succ"] += succ
        per_state[k]["tot"]  += M

    h = np.zeros(prob_model.n_components, dtype=np.float32)
    for k,v in per_state.items():
        h[k] = (v["succ"]/max(1,v["tot"]))
    return h

def main(out_dir="models/hmm/sim", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    probs = sample_problems(200, seed=seed)
    rows  = mine_partials(probs, per_problem=60, depth=6, seed=seed)

    seqs = []
    for r in rows:
        if not r["steps"]: continue
        seqs.append(encode_actions(r["steps"]).reshape(-1,1))

    if not seqs:
        raise RuntimeError("No sequences mined for HMM training.")

    lengths = [len(s) for s in seqs]
    X = np.concatenate(seqs, axis=0)

    K = 8
    model = hmm.GaussianHMM(n_components=K, covariance_type="spherical", n_iter=50, random_state=seed)
    model.fit(X, lengths)

    h = posterior_success(model, rows, K=K, seq_len=6, M=64)

    np.save(os.path.join(out_dir, "trans_means.npy"), model.means_)
    np.save(os.path.join(out_dir, "trans_covars.npy"), model.covars_)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"K": K, "h": h.tolist()}, f, indent=2)
    print("Saved HMM params + success head")

if __name__ == "__main__":
    main()
