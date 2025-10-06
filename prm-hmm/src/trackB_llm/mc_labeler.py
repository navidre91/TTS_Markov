# SPDX-License-Identifier: MIT
import os, json, argparse, re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sympy import sympify

def extract_final_number(text):
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    return nums[-1] if nums else None

def check_correct(gold, pred):
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", gold)
    if m:
        return (pred is not None) and (abs(float(pred) - float(m.group(1))) < 1e-6)
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

    rows = [json.loads(l) for l in open("data/math/raw/partials.jsonl", "r", encoding="utf-8")]
    out_f = open("data/math/labels/partials_labeled.jsonl", "w", encoding="utf-8")

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
    ap.add_argument("--samples", type=int, default=16, help="MC rollouts per partial")
    main(ap.parse_args())
