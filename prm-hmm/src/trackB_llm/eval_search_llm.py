# SPDX-License-Identifier: MIT
import os, json, argparse, heapq, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from .train_prm_llm import PRMHead
from .train_hmm_llm import step2vec
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
    cand = [t.split("\n")[0].strip() for t in texts]
    uniq = []
    for c in cand:
        if c and c not in uniq: uniq.append(c)
    return uniq[:k]

def build_prm_scorer():
    ckpt = torch.load("models/prm/llm/prm_llm.pt", map_location="cuda:0")
    enc_id = ckpt["cfg"]["encoder_id"]
    tok = AutoTokenizer.from_pretrained(enc_id, use_fast=True)
    enc = AutoModel.from_pretrained(enc_id).to("cuda:0")
    head = PRMHead(enc.config.hidden_size).to("cuda:0")
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
            return steps
        cand_steps = sample_next_steps(mdl, tok, root, steps, k=per_expand)
        for ns in cand_steps:
            steps2 = steps + [ns]
            p = scorer(q, steps2)
            heapq.heappush(pq, Node(steps2, p))
        if len(pq) > beam * 10:
            pq = pq[:beam * 10]
    return []

def main(args):
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")

    scorer = {"prm": build_prm_scorer(), "hmm": build_hmm_scorer()}.get(args.mode)
    assert scorer is not None, "mode must be prm or hmm"

    rows = [json.loads(l) for l in open("data/math/raw/partials.jsonl", "r", encoding="utf-8")]
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
