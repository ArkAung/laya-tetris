#!/usr/bin/env python3
"""Tetris played by laya-mlx.

Each turn: state text + a question + candidate actions -> laya picks one -> env steps.

  pip install laya-mlx            # Apple silicon Mac, Python 3.11+
  python laya_tetris.py --mode placement --seed 1
  python laya_tetris.py --mode primitive --seed 1 --debug
  python laya_tetris.py --policy greedy       # no model, sanity baseline
  python laya_tetris.py --policy random       # no model, floor
  python laya_tetris.py --games 10            # average over 10 seeds (compare policies this way)
  python laya_tetris.py --question "..."      # try different wording

The env mirrors the web version: same pieces, same seeded RNG, same state line.
"""
import argparse, copy, random, re, sys, time

W, H, M = 10, 20, 0xFFFFFFFF
SH = {'I': ['0000', '1111', '0000', '0000'], 'J': ['100', '111', '000'], 'L': ['001', '111', '000'],
      'O': ['11', '11'], 'S': ['011', '110', '000'], 'T': ['010', '111', '000'], 'Z': ['110', '011', '000']}
ROT = {}
for _t, _rows in SH.items():
    _m = [[int(c) for c in r] for r in _rows]; _n = len(_m); ROT[_t] = []
    for _ in range(4):
        ROT[_t].append([(x, y) for y, r in enumerate(_m) for x, v in enumerate(r) if v])
        _m = [[_m[_n - 1 - j][i] for j in range(_n)] for i in range(_n)]


class Piece:
    def __init__(s, t, r, x, y): s.t, s.r, s.x, s.y = t, r, x, y
    def cells(s, dx=0, dy=0, r=None):
        return [(s.x + x + dx, s.y + y + dy) for x, y in ROT[s.t][s.r if r is None else r]]


class Tetris:
    def __init__(s, seed=None): s.reset(seed)

    def reset(s, seed=None):
        s.rs = (random.getrandbits(32) if seed is None else seed) & M
        s.b = [['.'] * W for _ in range(H)]
        s.hold, s.hc, s.q, s.bag = None, True, [], []
        s.score = s.lines = 0; s.over = False
        s._fill(); s._next()

    # --- rng / queue (bit-identical to the JS version) ---
    def _rnd(s):
        s.rs = (s.rs + 0x6D2B79F5) & M; t = s.rs
        t = ((t ^ (t >> 15)) * (t | 1)) & M
        t = t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M)) & M)
        return ((t ^ (t >> 14)) & M) / 4294967296

    def _draw(s):
        if not s.bag:
            a = list("IJLOSTZ")
            for i in range(6, 0, -1):
                j = int(s._rnd() * (i + 1)); a[i], a[j] = a[j], a[i]
            s.bag = a
        return s.bag.pop(0)

    def _fill(s):
        while len(s.q) < 5: s.q.append(s._draw())

    def _spawn(s, t):
        s.cur = Piece(t, 0, (W - len(SH[t])) // 2, 0)
        if not s.fits(s.cur): s.over = True

    def _next(s):
        t = s.q.pop(0); s._fill(); s._spawn(t)

    # --- mechanics ---
    def level(s): return s.lines // 10 + 1

    def fits(s, c, dx=0, dy=0, r=None):
        return all(0 <= x < W and y < H and (y < 0 or s.b[y][x] == '.') for x, y in c.cells(dx, dy, r))

    def _lock(s):
        for x, y in s.cur.cells():
            if y >= 0: s.b[y][x] = s.cur.t
        keep = [r for r in s.b if '.' in r]; n = H - len(keep)
        s.b = [['.'] * W for _ in range(n)] + keep
        s.lines += n; s.score += [0, 100, 300, 500, 800][n] * s.level(); s.hc = True
        s._next(); return n

    def step(s, a):
        before = s.score; v, cl = True, 0
        if s.over or a not in ('left', 'right', 'cw', 'ccw', 'down', 'tick', 'drop', 'hold', 'noop'):
            return dict(obs=s.compact(), reward=0, done=s.over, valid=False, cleared=0)
        c = s.cur
        if a in ('left', 'right'):
            d = -1 if a == 'left' else 1
            if s.fits(c, d, 0): c.x += d
            else: v = False
        elif a in ('cw', 'ccw'):
            r = (c.r + (1 if a == 'cw' else 3)) % 4; v = False
            for k in (0, -1, 1, -2, 2):
                if s.fits(c, k, 0, r): c.x += k; c.r = r; v = True; break
        elif a in ('down', 'tick'):
            if s.fits(c, 0, 1):
                c.y += 1; s.score += a == 'down'
            else: cl = s._lock()
        elif a == 'drop':
            d = 0
            while s.fits(c, 0, 1): c.y += 1; d += 1
            s.score += 2 * d; cl = s._lock()
        elif a == 'hold':
            if not s.hc: v = False
            else:
                t = c.t
                if s.hold: s._spawn(s.hold)
                else: s._next()
                s.hold = t; s.hc = False
        return dict(obs=s.compact(), reward=s.score - before, done=s.over, valid=v, cleared=cl)

    def compact(s):
        rows = [''.join(r) for r in s.b]
        f = next((i for i, r in enumerate(rows) if set(r) != {'.'}), None)
        c = s.cur
        return (f"TETRIS1 s={s.score} l={s.lines} o={int(s.over)} c={c.t}{c.r}@{c.x},{c.y} h={s.hold or '-'} "
                f"hc={int(s.hc)} q={''.join(s.q)} g={''.join(s.bag) or '-'} rs={s.rs} "
                f"b={'' if f is None else '/'.join(rows[f:])}")

    def load(s, line):
        """Restore the whole game from a TETRIS1 state line."""
        p = dict(kv.split('=', 1) for kv in line.split()[1:])
        m = re.match(r'^([IJLOSTZ])([0-3])@(-?\d+),(-?\d+)$', p['c'])
        rows = p['b'].split('/') if p.get('b') else []
        rows = ['.' * W] * (H - len(rows)) + rows
        s.b = [list(r) for r in rows]
        s.cur = Piece(m[1], int(m[2]), int(m[3]), int(m[4]))
        s.hold = None if p['h'] == '-' else p['h']
        s.hc, s.q = p['hc'] == '1', list(p['q'])
        s.bag = [] if p['g'] == '-' else list(p['g'])
        s.rs, s.score, s.lines, s.over = int(p['rs']), int(p['s']), int(p['l']), p['o'] == '1'

    # --- helpers for the agent ---
    def features(s, board):
        hs = [next((H - y for y in range(H) if board[y][x] != '.'), 0) for x in range(W)]
        holes = sum(1 for x in range(W) for y in range(H - hs[x], H) if board[y][x] == '.')
        bump = sum(abs(hs[i] - hs[i + 1]) for i in range(W - 1))
        return hs, holes, bump

    def placements(s):
        """All distinct resting spots for the current piece: (rot_presses, target_x, info)."""
        out, seen = [], set(); holes0 = s.features(s.b)[1]
        for rp in range(4):
            c = copy.deepcopy(s.cur); ok = True
            for _ in range(rp):
                for k in (0, -1, 1, -2, 2):
                    if s.fits(c, k, 0, (c.r + 1) % 4): c.x += k; c.r = (c.r + 1) % 4; break
                else: ok = False; break
            if not ok: continue
            for tx in range(-3, W):
                p = Piece(c.t, c.r, tx, c.y)
                if not s.fits(p): continue
                while s.fits(p, 0, 1): p.y += 1
                key = frozenset(p.cells())
                if key in seen: continue
                seen.add(key)
                bd = [r[:] for r in s.b]
                for x, y in p.cells():
                    if y >= 0: bd[y][x] = p.t
                full = sum(1 for r in bd if '.' not in r)
                bd = [['.'] * W for _ in range(full)] + [r for r in bd if '.' in r]
                hs, holes, bump = s.features(bd)
                out.append(dict(rp=rp, tx=tx, lines=full, holes=holes, new_holes=holes - holes0, bump=bump, height=max(hs), agg=sum(hs),
                                col=min(x for x, _ in p.cells())))
        return out


def heuristic(p): return 0.76 * p['lines'] - 0.51 * p['agg'] / 10 - 0.36 * p['holes'] - 0.18 * p['bump'] / 4


# --- model-facing text ---
def model_state(env, board=True):
    hs, holes, bump = env.features(env.b)
    s = (f"Tetris. Piece {env.cur.t}, next {''.join(env.q)}, hold {env.hold or 'none'}. "
         f"Column heights {' '.join(map(str, hs))}. Holes {holes}. Score {env.score}, lines {env.lines}.")
    if board:
        rows = [''.join(r) for r in env.b]
        f = next((i for i, r in enumerate(rows) if set(r) != {'.'}), H)
        s += " Board top to bottom: " + '/'.join(rows[f:])
    return s


PRIM = {"move left": 'left', "move right": 'right', "rotate clockwise": 'cw', "rotate counterclockwise": 'ccw',
        "move down one row": 'down', "hard drop": 'drop', "hold the piece": 'hold'}


def pick_label(ans, cands):
    """Best-effort reading of laya's answer for a choice question (schema not documented on the card)."""
    if isinstance(ans, str) and ans in cands: return ans
    if isinstance(ans, dict):
        for k in ('probabilities', 'probs', 'scores', 'distribution'):
            d = ans.get(k)
            if isinstance(d, dict) and d: return max(d, key=d.get)
            if isinstance(d, (list, tuple)) and len(d) == len(cands): return cands[max(range(len(d)), key=d.__getitem__)]
        for k in ('label', 'answer', 'prediction', 'choice', 'selected', 'top'):
            if ans.get(k) in cands: return ans[k]
        if all(isinstance(v, (int, float)) for v in ans.values()): return max(ans, key=ans.get)
    raise ValueError(f"Can't read laya answer, rerun with --debug and adapt pick_label(): {ans!r}")


def ask_laya(agent, state, question, cands, debug=False):
    res = agent.predict(state, {"action": {"type": "choice", "instructions": question, "criteria": cands}})
    if debug: print("RAW:", res)
    return pick_label(res["answers"]["action"], cands)


Q_PLACEMENT = "Where should the piece be placed? Prefer clearing lines, then no holes, then a low flat surface."
Q_PLACEMENT_USER = ("You are playing Tetris. You have the board view. You have actions. You task is to clear rows. "
                    "Take actions based on the piece that is on the board state now.")
Q_PLACEMENT_V1 = Q_PLACEMENT   # kept so older eval code keeps working
Q_PRIMITIVE = "Which action should be taken to stack pieces without gaps and clear lines?"


def placement_label(p):
    """Human-readable description, used for logging."""
    return f"column {p['col']}, rotated {p['rp']}: clears {p['lines']}, holes {p['holes']}, height {p['height']}"


def option_label(p, pos, fmt="feat"):
    """What laya sees. No column/rotation words (laya has strong priors on them); the letter follows the slot.
    feat2 also shows bumpiness (surface roughness), which the scoring heuristic uses and `feat` hides."""
    if fmt == "feat2":
        return f"option {chr(65 + pos)}: clears {p['lines']}, holes {p['holes']}, bumpiness {p['bump']}, height {p['height']}"
    return f"option {chr(65 + pos)}: clears {p['lines']}, holes {p['holes']}, height {p['height']}"


def extract_scores(ans, cands):
    if isinstance(ans, dict):
        for k in ("probabilities", "probs", "scores", "distribution"):
            d = ans.get(k)
            if isinstance(d, dict) and set(d) == set(cands): return {c: float(d[c]) for c in cands}
            if isinstance(d, (list, tuple)) and len(d) == len(cands): return {c: float(v) for c, v in zip(cands, d)}
        if set(ans) == set(cands) and all(isinstance(v, (int, float)) for v in ans.values()):
            return {c: float(ans[c]) for c in cands}
    return None


def pick_placement(agent, state, question, ps, ensemble, rng, debug=False, fmt="feat"):
    """Ask laya `ensemble` times with the options in different random orders and sum the scores.
    Averaging cancels laya's slot/letter bias. Measured offline: ~0.95 on clear-cut pairs, far above the old format."""
    n = len(ps); tot = [0.0] * n
    for _ in range(max(1, ensemble)):
        order = list(range(n)); rng.shuffle(order)
        texts = [option_label(ps[j], pos, fmt) for pos, j in enumerate(order)]
        res = agent.predict(state, {"action": {"type": "choice", "instructions": question, "criteria": texts}})
        if debug: print("RAW:", res)
        ans = res["answers"]["action"]; sc = extract_scores(ans, texts)
        if sc:
            for t, j in zip(texts, order): tot[j] += sc[t]
        else: tot[order[texts.index(pick_label(ans, texts))]] += 1
    return max(range(n), key=tot.__getitem__)


LETTERS = "ABCDEFGH"
Q_LETTERS = "Which option is the best place for the current piece? Prefer clearing lines, then no holes, then a flat, low surface."


def letter_questions(k=8):
    """The question object a fine-tuned model is trained on and asked with. Instructions and option texts are part of
    the model's input, so training data, questions.json and play code must all use exactly this."""
    return {"placement": {"type": "choice", "instructions": Q_LETTERS,
                          "criteria": {L: f"option {L}" for L in LETTERS[:k]}}}


def tetris_state_text(env, ps):
    """State for the letters format: what the engine knows plus a table of the options A, B, C... in `ps` order."""
    hs, holes, bump = env.features(env.b)
    head = (f"Tetris. Piece {env.cur.t}, next {''.join(env.q[:3])}, hold {env.hold or 'none'}. "
            f"Stack height {max(hs)}, holes {holes}, bumpiness {bump}. Options:")
    rows = [f"{LETTERS[i]}: clears {p['lines']}, holes {p['holes']}, bumpiness {p['bump']}, height {p['height']}"
            for i, p in enumerate(ps)]
    return head + "\n" + "\n".join(rows)


def execute_placement(env, do, x):
    """Turn a chosen placement into rotate / move / drop actions. `do` as in play_turn."""
    for _ in range(x["rp"]): do("cw")
    for _ in range(20):
        if env.cur.x == x["tx"]: break
        if not do("left" if env.cur.x > x["tx"] else "right")["valid"]: break
    do("drop")


def load_questions(model):
    """questions.json that ships with a fine-tuned checkpoint (local folder or Hub repo), else None."""
    import json, os
    p = os.path.join(model, "questions.json")
    if os.path.isdir(model) and os.path.exists(p): return json.load(open(p))
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(model, "questions.json")))
    except Exception:
        return None


def pick_letters(agent, env, ps, rng, ensemble=1, debug=False):
    """One question over a state that contains the option table; the options are the letters A..H.
    For a model fine-tuned on this exact format (see make_tetris_dataset.py). `ps` is sorted best-first by the
    heuristic only because play_turn pre-prunes with it; we shuffle before showing, so the order leaks nothing."""
    n = len(ps); tot = [0.0] * n
    qs = getattr(agent, "_tetris_questions", None) or letter_questions()
    for _ in range(max(1, ensemble)):
        order = list(range(n)); rng.shuffle(order)
        state = tetris_state_text(env, [ps[j] for j in order])
        res = agent.predict(state, qs)
        if debug: print("RAW:", res)
        ans = res["answers"]["placement"]; sc = extract_scores(ans, list(qs["placement"]["criteria"]))
        for pos, j in enumerate(order):
            tot[j] += sc[LETTERS[pos]] if sc else float(pick_label(ans, list(LETTERS[:n])) == LETTERS[pos])
    return max(range(n), key=tot.__getitem__)


def pick_pairwise(agent, state, question, ps, debug=False, fmt="feat"):
    """Use laya as a two-way comparator: every pair of options, both orders, sum the win probabilities.
    Measured offline: top-1 agreement with the heuristic 0.80 vs 0.29 for one 8-way question."""
    n = len(ps); tot = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            pi = 0.0
            for first, second in ((i, j), (j, i)):
                t = [option_label(ps[first], 0, fmt), option_label(ps[second], 1, fmt)]
                res = agent.predict(state, {"action": {"type": "choice", "instructions": question, "criteria": t}})
                if debug: print("RAW:", res)
                ans = res["answers"]["action"]; sc = extract_scores(ans, t)
                p0 = sc[t[0]] / (sc[t[0]] + sc[t[1]]) if sc else float(pick_label(ans, t) == t[0])
                pi += (p0 if first == i else 1 - p0) / 2
            tot[i] += pi; tot[j] += 1 - pi
    return max(range(n), key=tot.__getitem__)


def play_turn(env, do, agent=None, rng=None, mode="placement", policy="laya", max_cands=8, board=False,
              debug=False, question=None, ensemble=4, compare="pairwise", fmt="feat", trace=None):
    """One decision: build state + question + candidates, pick one, execute it. Used by every runner.

    `do(action)` must apply the action AND leave `env` reflecting the new state (return the step result dict).
    Terminal runner: do = env.step.  WebSocket runner: do sends to the page, then env.load(page state).
    Returns a text label of what was chosen.
    """
    rng = rng or random.Random()
    state = model_state(env, board)
    if mode == "primitive":
        cands = list(PRIM)
        choice = ask_laya(agent, state, question or Q_PRIMITIVE, cands, debug) if policy == "laya" else rng.choice(cands)
        do(PRIM[choice]); return choice
    ps = sorted(env.placements(), key=heuristic, reverse=True)[:max_cands]   # NB: the heuristic pre-prunes to the best N
    if policy == "greedy": i = 0
    elif policy == "laya" and compare == "letters": i = pick_letters(agent, env, ps, rng, ensemble if ensemble > 1 else 1, debug)
    elif policy == "laya" and compare == "pairwise": i = pick_pairwise(agent, state, question or Q_PLACEMENT, ps, debug, fmt)
    elif policy == "laya": i = pick_placement(agent, state, question or Q_PLACEMENT, ps, ensemble, rng, debug, fmt)
    else: i = rng.randrange(len(ps))
    if trace is not None:   # on-policy log: features of every option and which one was taken (index 0 = heuristic's pick)
        hs = [heuristic(p) for p in ps]
        trace.append(dict(state=env.compact(), picked=i, regret=round(hs[0] - hs[i], 3),
                          feats=[[p['lines'], p['holes'], p['bump'], p['height'], p['new_holes']] for p in ps]))
    x = ps[i]
    execute_placement(env, do, x)
    return placement_label(x)


def load_agent(model, backend="mlx"):
    """backend 'mlx': pip package laya-mlx (Apple silicon).  'torch': pip package `laya` (PyTorch) for checkpoints
    that only exist in transformers format, e.g. convaiinnovations/laya-typed-decisions."""
    if backend == "torch":
        import os
        os.environ.setdefault("USE_TF", "0")   # model card: avoids a TensorFlow/abseil deadlock while loading
        import laya
    else:
        import laya_mlx as laya
    agent = laya.load(model)
    try: agent._tetris_questions = load_questions(model)   # the questions this checkpoint was trained with, if it ships them
    except Exception: pass
    return agent


def add_agent_args(ap):
    ap.add_argument("--policy", default="laya", choices=["laya", "greedy", "random"])
    ap.add_argument("--mode", default="placement", choices=["placement", "primitive"])
    ap.add_argument("--model", default="aac6fef/laya-mlx")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"], help="torch needs: uv run --with laya ...")
    ap.add_argument("--max-cands", type=int, default=8, help="placement mode: keep the N best by heuristic")
    ap.add_argument("--compare", default="pairwise", choices=["pairwise", "choice", "letters"],
                    help="pairwise: laya compares every pair of options (about 1 s/move); choice: one N-way question; letters: ONE question on the option table, for a model fine-tuned with make_tetris_dataset.py")
    ap.add_argument("--format", default="feat", choices=["feat", "feat2"], help="feat2 adds bumpiness to each option")
    ap.add_argument("--ensemble", type=int, default=4, help="ask laya this many times with shuffled options and average")
    ap.add_argument("--question", default=None, help="override the question text sent to laya")
    ap.add_argument("--board", action="store_true", help="also send the raw board rows (measured: no help, more tokens)")
    ap.add_argument("--debug", action="store_true")


def turn_kwargs(a):
    return dict(mode=a.mode, policy=a.policy, max_cands=a.max_cands, board=a.board, debug=a.debug,
                question=a.question, ensemble=a.ensemble, compare=a.compare, fmt=a.format)


def summarize_trace(tr):
    if not tr: return ""
    n = len(tr); opp = [t for t in tr if max(f[0] for f in t['feats']) > 0]
    taken = sum(t['feats'][t['picked']][0] == max(f[0] for f in t['feats']) for t in opp)
    return (f"agree_with_heuristic={sum(t['regret'] <= 1e-9 for t in tr) / n:.2f} "
            f"regret={sum(t['regret'] for t in tr) / n:.2f} clears_taken={taken}/{len(opp)}")


def main():
    ap = argparse.ArgumentParser()
    add_agent_args(ap)
    ap.add_argument("--trace", default=None, help="write every decision (state, option features, pick) to this .jsonl file")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--games", type=int, default=1, help="play this many games (seeds seed, seed+1, ...) and average")
    ap.add_argument("--max-moves", type=int, default=200)
    a = ap.parse_args()

    agent = None
    if a.policy == "laya":
        agent = load_agent(a.model, a.backend)
    res = []; all_tr = []
    for g in range(a.games):
        seed = a.seed + g
        env = Tetris(seed); rng = random.Random(seed); t0 = time.time(); n = 0; tr = []
        while not env.over and n < a.max_moves:
            before = env.lines
            label = play_turn(env, env.step, agent, rng, trace=tr, **turn_kwargs(a))
            n += 1
            if a.debug or (a.games == 1 and (n % 10 == 0 or env.lines > before)):
                print(f"#{n:3d} {label:70s} score={env.score} lines={env.lines}")
        res.append((env.lines, env.score, n, env.over))
        hs, holes, _ = env.features(env.b)
        for k, t in enumerate(tr): t.update(game=g + 1, seed=seed, move=k + 1)
        all_tr += tr
        print(f"game {g + 1}/{a.games} seed={seed}: moves={n} lines={env.lines} score={env.score} over={env.over} "
              f"time={time.time() - t0:.1f}s | end: holes={holes} max_height={max(hs)} | {summarize_trace(tr)}", flush=True)
    if a.trace:
        import json
        with open(a.trace, 'w') as f:
            for t in all_tr: f.write(json.dumps(t) + '\n')
        print(f'wrote {len(all_tr)} decisions to {a.trace}')
    if a.games > 1:
        k = len(res)
        print(f"\nmean over {k} games: lines={sum(r[0] for r in res) / k:.1f} score={sum(r[1] for r in res) / k:.0f} "
              f"moves={sum(r[2] for r in res) / k:.0f} died={sum(r[3] for r in res)}/{k}   "
              f"on-policy: {summarize_trace(all_tr)}   "
              f"(policy={a.policy}, mode={a.mode}, compare={a.compare}, format={a.format}, ensemble={a.ensemble}, max_cands={a.max_cands}, max_moves={a.max_moves})")


if __name__ == "__main__":
    main()
