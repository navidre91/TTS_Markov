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
        for _ in range(per_problem):
            s_cur = pr.start
            steps = []
            d = int(rng.integers(0, depth+1))
            for _ in range(d):
                a = rng.choice(ACTIONS)
                s_cur = apply_action(s_cur, a)
                steps.append(a)
            rows.append({"pid": pid, "start": pr.start, "target": pr.target, "cur": s_cur, "steps": steps})
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
