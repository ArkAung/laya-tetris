#!/usr/bin/env python3
"""Build a Tetris training set for fine-tuning Laya (LayaStudio on a Mac, or the upstream RLCD notebook).

Each row is one decision: a state text containing the option table A..H, and the teacher's preference over the
options as a soft label. The teacher is the heuristic that already plays well (greedy cleared ~56 lines per 150 moves).

  uv run make_tetris_dataset.py --n 8000 --out tetris_data
      -> tetris_data/train.jsonl   rows: {"state": ..., "answers": {"placement": {"A": 0.93, "B": 0.05, ...}}, "split": ...}
         tetris_data/questions.json   the question object; keep it identical when asking the trained model

  # on-policy round (fixes the states the model itself reaches): play with the tuned model, then relabel its states
  uv run laya_tetris.py --model <tuned> --compare letters --games 20 --trace trace.jsonl
  uv run make_tetris_dataset.py --from-trace trace.jsonl --out tetris_data_round2
"""
import argparse, json, math, os, random
from laya_tetris import Tetris, heuristic, tetris_state_text, letter_questions, execute_placement, LETTERS


def soft_label(hs, temp):
    m = max(hs); e = [math.exp((h - m) / temp) for h in hs]; z = sum(e)
    return [x / z for x in e]


def make_row(env, k, rng, temp):
    ps = sorted(env.placements(), key=heuristic, reverse=True)[:k]
    if len(ps) < 2: return None
    order = list(range(len(ps))); rng.shuffle(order)          # random letter assignment, so letters carry no information
    shown = [ps[j] for j in order]
    probs = soft_label([heuristic(p) for p in shown], temp)
    return {"state": tetris_state_text(env, shown),
            "answers": {"placement": {LETTERS[i]: round(probs[i], 4) for i in range(len(shown))}}}


def split_of(game, seed):
    r = random.Random(f"{seed}-{game}").random()               # whole games go to one split: no leakage between rows
    return "train" if r < 0.8 else "val" if r < 0.9 else "test"


def from_play(a, rng):
    rows, game = [], 0
    while len(rows) < a.n:
        game += 1; eps = rng.choice([0.0, 0.1, 0.3, 0.6]); env = Tetris(rng.getrandbits(32)); moves = 0
        while not env.over and moves < a.game_len and len(rows) < a.n:
            row = make_row(env, a.k, rng, a.temp)
            if row and (moves % a.stride == 0):
                row["split"] = split_of(game, a.seed); rows.append(row)
            ps = sorted(env.placements(), key=heuristic, reverse=True)[:a.k]   # advance the game (explore with prob eps)
            execute_placement(env, env.step, rng.choice(ps) if rng.random() < eps else ps[0])
            moves += 1
    return rows


def from_trace(a, rng):
    rows = []
    for line in open(a.from_trace):
        t = json.loads(line); env = Tetris(0); env.load(t["state"])
        row = make_row(env, a.k, rng, a.temp)
        if row: row["split"] = split_of(f"trace{t.get('game', 0)}", a.seed); rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="rows to generate from play")
    ap.add_argument("--k", type=int, default=8, help="options per decision, at most 8")
    ap.add_argument("--temp", type=float, default=0.1, help="soft-label temperature over heuristic scores (small = near one-hot)")
    ap.add_argument("--game-len", type=int, default=60, help="moves per generated game; many short games so the splits are by game and diverse")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth decision of each game")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-trace", default=None, help="relabel states from a laya_tetris.py --trace file instead of generating")
    ap.add_argument("--out", default="tetris_data")
    a = ap.parse_args()
    assert 2 <= a.k <= len(LETTERS)
    rng = random.Random(a.seed)
    rows = from_trace(a, rng) if a.from_trace else from_play(a, rng)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/train.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(letter_questions(a.k if a.k == 8 else 8), open(f"{a.out}/questions.json", "w"), indent=1)
    cnt = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    top = sum(max(r["answers"]["placement"].values()) for r in rows) / len(rows)
    print(f"wrote {len(rows)} rows to {a.out}/train.jsonl  splits={cnt}  mean top label prob={top:.2f}")
    print("example state:\n" + rows[0]["state"] + "\nlabel: " + json.dumps(rows[0]["answers"]))


if __name__ == "__main__":
    main()
