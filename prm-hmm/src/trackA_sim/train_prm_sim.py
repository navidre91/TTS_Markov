# SPDX-License-Identifier: MIT
import os, json, numpy as np, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .sim_env import Problem, State, sample_problems, mine_partials, mc_success_prob, features

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

    meta = {"feat_dim": int(X_tr.shape[1]), "val_bce": float(best)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
    print("Saved PRM to", best_path)

if __name__ == "__main__":
    main()
