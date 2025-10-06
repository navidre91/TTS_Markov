# SPDX-License-Identifier: MIT
import os, json, argparse, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class PartialDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate(batch, tok, max_len=512):
    texts = [f"Problem:\n{b['question']}\n\nPartial solution up to step {b['t']}:\n{b['partial']}" for b in batch]
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
            X, y = batch
            X = {k:v.cuda() for k,v in X.items()}; y = y.cuda()
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                h = enc(**X).last_hidden_state[:,0]
                p = head(h).clamp(1e-4, 1-1e-4)
                loss = bce(p, y)
            loss.backward(); opt.step()
            losses.append(loss.item())

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
