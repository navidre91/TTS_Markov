# SPDX-License-Identifier: MIT
import os, json, argparse, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

def fmt_prompt(q):
    return f"Solve the problem step by step.\n\nProblem:\n{q}\n\nSolution:\n"

def split_steps(text):
    parts = [p.strip() for p in re.split(r"(?:^|\n)(?:Step\s*\d*[:.)-]?)", text) if p.strip()]
    steps = []
    for p in parts:
        if len(p) < 3: continue
        steps.append(p)
    if not steps: steps = [text.strip()]
    return steps

def main(args):
    os.makedirs("data/math/raw", exist_ok=True)
    ds = load_dataset("gsm8k", "main")["train"].select(range(200))  # small subset
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")

    rows = []
    for i, ex in tqdm(enumerate(ds), total=len(ds)):
        q = ex["question"]; a = ex["answer"]
        prompt = fmt_prompt(q)
        inp = tok(prompt, return_tensors="pt").to(mdl.device)
        out = mdl.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, eos_token_id=tok.eos_token_id)
        text = tok.decode(out[0], skip_special_tokens=True)
        sol = text.split("Solution:",1)[-1].strip()
        steps = split_steps(sol)
        for t in range(min(len(steps), 6)):
            partial = "\n".join(steps[:t+1])
            rows.append({"pid": i, "question": q, "gold": a, "partial": partial, "t": t+1})
    with open("data/math/raw/partials.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print("Wrote", len(rows), "partials to data/math/raw/partials.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="HF model id or local path")
    main(ap.parse_args())
