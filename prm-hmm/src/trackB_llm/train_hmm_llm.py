# SPDX-License-Identifier: MIT
import os, json, argparse, numpy as np, re
from tqdm import tqdm
from hmmlearn import hmm

def tokenize_step(s):
    toks = re.findall(r"[A-Za-z]+|[0-9]+|[\+\-\*/=\(\)]", s)
    return [t[:8] for t in toks][:50]

def step2vec(step, vocab):
    v = np.zeros(len(vocab), dtype=np.float32)
    for t in tokenize_step(step):
        if t in vocab: v[vocab[t]] += 1.0
    if np.linalg.norm(v) > 0:
        v = v / np.linalg.norm(v)
    return v

def main(args):
    rows = [json.loads(l) for l in open("data/math/labels/partials_labeled.jsonl", "r", encoding="utf-8")]
    counts = {}
    for r in rows:
        for t in tokenize_step(r["partial"]):
            counts[t] = counts.get(t, 0) + 1
    vocab_items = sorted(counts.items(), key=lambda x: -x[1])[:512]
    vocab = {t:i for i,(t,_) in enumerate(vocab_items)}

    sequences = []
    lengths = []
    for r in rows:
        steps = [s for s in r["partial"].split("\n") if s.strip()]
        vecs = [step2vec(s, vocab) for s in steps[-5:]]
        if not vecs: continue
        seq = np.stack(vecs, axis=0)
        sequences.append(seq)
        lengths.append(len(seq))

    if not sequences:
        raise RuntimeError("No sequences built for HMM training.")

    X = np.concatenate(sequences, axis=0)

    K = args.K
    model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=50, random_state=0)
    model.fit(X, lengths)

    state_succ = np.zeros(K); state_tot = np.zeros(K)
    for r in tqdm(rows, desc="Attach success head"):
        steps = [s for s in r["partial"].split("\n") if s.strip()]
        if not steps: continue
        vecs = [step2vec(s, vocab) for s in steps[-5:]]
        Xs = np.stack(vecs, axis=0)
        _, states = model.decode(Xs, algorithm="viterbi")
        k = states[-1]
        state_succ[k] += r["p_hat"]; state_tot[k] += 1.0
    h = (state_succ / np.maximum(1.0, state_tot)).tolist()

    os.makedirs("models/hmm/llm", exist_ok=True)
    np.save("models/hmm/llm/hmm_means.npy", model.means_)
    np.save("models/hmm/llm/hmm_covars.npy", model.covars_)
    json.dump({"K": K, "h": h, "vocab": list(vocab.keys())}, open("models/hmm/llm/meta.json","w"), indent=2)
    print("Saved HMM baseline with success head.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    main(ap.parse_args())
