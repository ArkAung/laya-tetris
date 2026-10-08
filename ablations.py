#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["laya-mlx"]
# ///
"""Re-run each thing that went wrong, at several levels, on the same boards.

Each experiment changes ONE factor and keeps everything else fixed, so the tables can go straight into the write-up.
Everything is measured against the heuristic teacher on the same seeded boards, with the options shuffled the same way
in every condition.

  A  state    how the board is shown: nothing / summary only / rows of letters / rows of # and . / newline-separated
  B  options  how each option is written: names only / original / plain English / change in holes / numbers / numbers+bumpiness
  C  count    how many options are on offer: 2, 3, 4, 6, 8
  D  averaging how many shuffled orderings are averaged: 1, 2, 4, 8
  E  moves    the original idea: raw moves (left, rotate, drop...) with the board as the state
  F  tuned    what the fine-tuned model actually reads (needs --tuned PATH)

  uv run ablations.py                               # A to E on the zero-shot base model, 300 boards each
  uv run ablations.py --only A,B --n 500            # a subset, tighter error bars
  uv run ablations.py --tuned LayaStudio/workspace/runs/<run>/model --only F
  uv run ablations.py --dry-run                     # fake model, only checks that the script runs

Writes ablations_summary.txt (paste this into the chat) and ablations_report.json (everything, for later charts).
Rough cost: about 25 ms per call; A and B are about 2,000 calls each at --n 300, C and D about 3,000 and 5,000.
"""
import argparse, collections, json, math, random, re, sys, time

from laya_tetris import (Tetris, heuristic, model_state, option_label, load_agent, tetris_state_text, letter_questions,
                         extract_scores, PRIM, LETTERS, Q_PLACEMENT, Q_PRIMITIVE)
from eval_laya import gen_states, build_pools, make_record, summarize, ask, cand_text


# ---------- how the state can be written ----------
def board_rows(env):
    rows = [''.join(r) for r in env.b]
    first = next((i for i, r in enumerate(rows) if set(r) != {'.'}), None)
    return [] if first is None else rows[first:]


def state_variant(env, kind):
    if kind == "blank": return "Tetris."
    base = model_state(env, board=False)                     # piece, queue, hold, column heights, holes, score
    if kind == "summary": return base
    rows = board_rows(env)
    if kind == "letters": return base + " Board top to bottom: " + "/".join(rows)
    if kind == "letters_nl": return base + " Board, top to bottom:\n" + "\n".join(rows)
    hashed = [re.sub(r"[^.]", "#", r) for r in rows]
    if kind == "hash": return base + " Board top to bottom: " + "/".join(hashed)
    if kind == "hash_nl": return base + " Board, top to bottom:\n" + "\n".join(hashed)
    raise ValueError(kind)


def option_text(p, fmt, pos):
    if fmt in ("feat", "feat2"): return option_label(p, pos, fmt)   # lettered options, numbers only
    return cand_text(p, fmt, pos)                                    # num (original), names, plain, delta


# ---------- one condition on one pool ----------
def run_choice(agent, data, state_kind, fmt, question, reps, seed, store, k=None):
    recs = []
    for i, (line, cands) in enumerate(data):
        if k: cands = cands[:k]
        if len(cands) < 2: continue
        env = Tetris(0); env.load(line); st = state_variant(env, state_kind)
        tot = [0.0] * len(cands); dt = 0.0; usage = {}; first = None; ok = True
        for r in range(reps):
            order = list(range(len(cands))); random.Random(seed * 7919 + i + 100003 * r).shuffle(order)
            texts = [option_text(cands[j], fmt, pos) for pos, j in enumerate(order)]
            if len(set(texts)) != len(texts): ok = False; break
            try: label, scores, d, usage = ask(agent, st, question, texts, store)
            except Exception as e:
                store["errors"].append(f"{state_kind}/{fmt}/#{i}: {e!r}"[:300]); ok = False; break
            dt += d
            if r == 0: first = (order, texts, scores)
            if scores:
                for t, j in zip(texts, order): tot[j] += scores[t]
            else: tot[order[texts.index(label)]] += 1
        if not ok:
            if len(store["errors"]) >= 3: raise RuntimeError("repeated errors: " + " | ".join(store["errors"]))
            continue
        order0, texts0, sc0 = first
        pick = max(range(len(cands)), key=tot.__getitem__)
        sbt = {texts0[p]: tot[order0[p]] for p in range(len(cands))} if sc0 else None
        recs.append(make_record(cands, pick, order0, sbt, texts0, dt, usage))
    return summarize(recs, None)


def row(name, s):
    if not s: return f"  {name:30s} (no data)"
    extra = ""
    if "missed_line_clear" in s: extra += f"  missedClear={s['missed_line_clear']}({s['missed_line_clear_random']})"
    if "picked_more_holes" in s: extra += f" moreHoles={s['picked_more_holes']}({s['picked_more_holes_random']})"
    return (f"  {name:30s} n={s['n']:4d} top1={s['top1']:.2f}±{s['top1_ci95']:.2f} chance={s['chance']:.2f} "
            f"regret={s['regret']:.3f} (random {s['rand_regret']:.3f}){extra}")


def grid(agent, pools, pool_names, conditions, question, store, seed, reps=1, k=None):
    out = {}
    for pool in pool_names:
        for name, kw in conditions:
            t0 = time.time()
            out.setdefault(name, {})[pool] = run_choice(agent, pools[pool], question=question, seed=seed, store=store,
                                                        reps=kw.get("reps", reps), k=kw.get("k", k),
                                                        state_kind=kw.get("state", "summary"), fmt=kw.get("fmt", "num"))
            s = out[name][pool]
            print(f"   {pool:5s} {name:28s} top1={s.get('top1')} ({time.time() - t0:.0f}s)", flush=True)
    return out


def render(title, results, pool_names, note=""):
    L = ["", title, note] if note else ["", title]
    for pool in pool_names:
        L.append(f" pool: {pool}")
        for name, per in results.items(): L.append(row(name, per.get(pool)))
    return L


# ---------- E: raw moves with the board as the state ----------
def teacher_next(env):
    ps = env.placements()
    if not ps: return None
    p = max(ps, key=heuristic)
    if p["rp"] > 0: return "cw"
    if env.cur.x != p["tx"]: return "left" if env.cur.x > p["tx"] else "right"
    return "drop"


def primitive_states(lines, seed):
    rng = random.Random(seed); out = []
    for line in lines:
        env = Tetris(0); env.load(line)
        for _ in range(rng.randrange(0, 6)):                       # follow the teacher a few steps, so we also see mid-move states
            a = teacher_next(env)
            if a in (None, "drop"): break
            env.step(a)
        a = teacher_next(env)
        if a: out.append((env.compact(), a))
    return out


ACTION_TEXT = list(PRIM)   # "move left", "move right", "rotate clockwise", ...


def experiment_moves(agent, lines, store, seed):
    items = primitive_states(lines, seed); res = {}
    labels = [a for _, a in items]; maj = collections.Counter(labels).most_common(1)[0]
    for kind in ("blank", "summary", "letters", "hash"):
        hits = 0; picked = collections.Counter(); per = collections.defaultdict(lambda: [0, 0]); n = 0
        for i, (line, want) in enumerate(items):
            env = Tetris(0); env.load(line)
            st = state_variant(env, kind) + f" The piece is at rotation {env.cur.r}, left edge in column {env.cur.x}."
            order = list(range(len(ACTION_TEXT))); random.Random(seed * 31 + i).shuffle(order)
            texts = [ACTION_TEXT[j] for j in order]
            label, scores, d, usage = ask(agent, st, Q_PRIMITIVE, texts, store)
            got = PRIM[label]; n += 1; hits += got == want; picked[got] += 1
            per[want][0] += got == want; per[want][1] += 1
        res[kind] = dict(n=n, acc=round(hits / n, 3), ci95=round(1.96 * math.sqrt(max(hits / n * (1 - hits / n), 1e-9) / n), 3),
                         picked=dict(picked), per_teacher_action={k: f"{v[0]}/{v[1]}" for k, v in per.items()})
        print(f"   moves {kind:8s} acc={res[kind]['acc']}", flush=True)
    res["_baselines"] = dict(chance=round(1 / len(ACTION_TEXT), 3), always_most_common=round(maj[1] / len(labels), 3), most_common=maj[0],
                             teacher_label_counts=dict(collections.Counter(labels)))
    return res


# ---------- F: what the fine-tuned model reads ----------
def mask_field(text, field):
    head, _, rest = text.partition("\n")
    return head + "\n" + re.sub(rf"\b{field} \d+", f"{field} 0", rest)


def tuned_variant(env, shown, mod):
    text = tetris_state_text(env, shown)
    if mod == "full": return text
    if mod.startswith("mask:"): return mask_field(text, mod[5:])
    if mod == "no_table": return text.split("\n")[0]
    if mod == "plus_board": return text + "\nBoard top to bottom: " + "/".join(board_rows(env))
    raise ValueError(mod)


def run_tuned(agent, data, mod, k, seed, store):
    qs = getattr(agent, "_tetris_questions", None) or letter_questions()
    allowed = list(qs["placement"]["criteria"]); recs = []
    for i, (line, cands) in enumerate(data):
        cands = cands[:k]
        if len(cands) < 2: continue
        env = Tetris(0); env.load(line)
        order = list(range(len(cands))); random.Random(seed * 7919 + i).shuffle(order)
        shown = [cands[j] for j in order]
        t0 = time.time(); res = agent.predict(tuned_variant(env, shown, mod), qs); dt = time.time() - t0
        sc = extract_scores(res["answers"]["placement"], allowed)
        letters = LETTERS[:len(shown)]
        pos = max(range(len(shown)), key=lambda p: sc[letters[p]])
        recs.append(make_record(cands, order[pos], order, {letters[p]: sc[letters[p]] for p in range(len(shown))},
                                list(letters), dt, res.get("usage") or {}))
    return summarize(recs, None)


class OracleLetters:   # stand-in for --dry-run
    _tetris_questions = None

    def predict(self, state, qs):
        rows = re.findall(r"^([A-H]): clears (\d+), holes (\d+), bumpiness (\d+), height (\d+)", state, re.M)
        sc = {r[0]: 0.76 * int(r[1]) - 0.36 * int(r[2]) - 0.18 * int(r[3]) / 4 for r in rows} or {"A": 0.0}
        z = sum(math.exp(3 * v) for v in sc.values())
        full = {L: math.exp(3 * sc[L]) / z if L in sc else 0.0 for L in LETTERS}
        return {"answers": {"placement": {"probabilities": full}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="boards per experiment (error bar is about +-0.06 at 300, +-0.04 at 500)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="A,B,C,D,E", help="comma list of experiments; add F with --tuned")
    ap.add_argument("--model", default="aac6fef/laya-mlx", help="the zero-shot base model for A to E")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--tuned", default=None, help="path or Hub id of the fine-tuned model, for experiment F")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    only = [x.strip().upper() for x in a.only.split(",") if x.strip()]
    if a.tuned and "F" not in only and a.only == "A,B,C,D,E": only = ["F"]

    from eval_laya import Fake
    store = {"raw": [], "errors": []}
    lines = gen_states(a.n, a.seed); pools = build_pools(lines, 8, a.seed)
    print(f"{len(lines)} boards; pool sizes: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()), flush=True)
    report, text = {}, ["LAYA TETRIS ABLATIONS", f"n={a.n} seed={a.seed} model={a.model} tuned={a.tuned} only={only}",
                        "top1 compares with the heuristic's best option; chance is 1/options; regret = score gap to the best option;",
                        "missedClear / moreHoles show (model rate) against (random rate); +- is a 95% interval on top1."]

    if set(only) & set("ABCDE"):
        agent = Fake("random", a.seed) if a.dry_run else load_agent(a.model, a.backend)
        if "A" in only:
            print("A: state representation", flush=True)
            conds = [("blank ('Tetris.')", dict(state="blank")), ("summary only", dict(state="summary")),
                     ("rows of letters, one line", dict(state="letters")), ("rows of letters, one per line", dict(state="letters_nl")),
                     ("rows of # and ., one line", dict(state="hash")), ("rows of # and ., one per line", dict(state="hash_nl"))]
            report["A"] = grid(agent, pools, ["pair", "top", "mixed"], conds, Q_PLACEMENT, store, a.seed)
            text += render("A. How the board is shown (options written the original way, one call, shuffled)", report["A"],
                           ["pair", "top", "mixed"], "pair = clear best vs clear worst; top = the real shortlist; mixed = best + 7 random")
        if "B" in only:
            print("B: option text", flush=True)
            conds = [("names only (column, rotation)", dict(fmt="names")), ("original: names + numbers", dict(fmt="num")),
                     ("plain English", dict(fmt="plain")), ("holes as a change (+0)", dict(fmt="delta")),
                     ("lettered, numbers only", dict(fmt="feat")), ("lettered, numbers + bumpiness", dict(fmt="feat2"))]
            report["B"] = grid(agent, pools, ["pair", "top", "mixed"], conds, Q_PLACEMENT, store, a.seed)
            text += render("B. How each option is written (state = summary only, one call, shuffled)", report["B"], ["pair", "top", "mixed"])
        if "C" in only:
            print("C: number of options", flush=True)
            conds = [(f"{k} options", dict(k=k)) for k in (2, 3, 4, 6, 8)]
            report["C"] = {}
            for name, kw in conds:
                for pool in ("top", "mixed"):
                    s = run_choice(agent, pools[pool], "summary", "feat", Q_PLACEMENT, 4, a.seed, store, k=kw["k"])
                    report["C"].setdefault(name, {})[pool] = s; print(f"   {pool} {name} top1={s.get('top1')}", flush=True)
            text += render("C. Number of options (lettered numbers, 4 orderings averaged)", report["C"], ["top", "mixed"],
                           "top = the k best by heuristic; mixed = the best plus k-1 random")
        if "D" in only:
            print("D: averaging", flush=True)
            conds = [(f"{r} ordering{'s' if r > 1 else ''}", dict(reps=r)) for r in (1, 2, 4, 8)]
            report["D"] = grid(agent, pools, ["pair", "top", "mixed"], [(n, dict(fmt="feat", **kw)) for n, kw in conds], Q_PLACEMENT, store, a.seed)
            text += render("D. How many shuffled orderings are averaged (lettered numbers, summary state)", report["D"], ["pair", "top", "mixed"])
        if "E" in only:
            print("E: raw moves", flush=True)
            report["E"] = experiment_moves(agent, lines, store, a.seed)
            b = report["E"]["_baselines"]
            text += ["", "E. Raw moves with the board as the state (the original idea): can it pick the teacher's next move?",
                     f"  options: {', '.join(ACTION_TEXT)}.  chance={b['chance']}  always '{b['most_common']}'={b['always_most_common']}  labels={b['teacher_label_counts']}"]
            for kind in ("blank", "summary", "letters", "hash"):
                r = report["E"][kind]
                text.append(f"  state={kind:8s} n={r['n']} accuracy={r['acc']}±{r['ci95']}  picked={r['picked']}  correct per teacher action={r['per_teacher_action']}")

    if "F" in only:
        if not (a.tuned or a.dry_run): sys.exit("Experiment F needs --tuned PATH_OR_HUB_ID")
        print("F: the fine-tuned model", flush=True)
        agent = OracleLetters() if a.dry_run else load_agent(a.tuned, a.backend)
        mods = [("full table (as trained)", "full", 8), ("clears column set to 0", "mask:clears", 8), ("holes column set to 0", "mask:holes", 8),
                ("bumpiness column set to 0", "mask:bumpiness", 8), ("height column set to 0", "mask:height", 8),
                ("options table removed", "no_table", 8), ("board rows added after the table", "plus_board", 8),
                ("6 options", "full", 6), ("4 options", "full", 4), ("2 options", "full", 2)]
        report["F"] = {}
        for name, mod, k in mods:
            for pool in ("top", "mixed"):
                s = run_tuned(agent, pools[pool], mod, k, a.seed, store)
                report["F"].setdefault(name, {})[pool] = s; print(f"   {pool:5s} {name:34s} top1={s.get('top1')}", flush=True)
        text += render("F. What the fine-tuned model reads (one call per board, options shuffled)", report["F"], ["top", "mixed"],
                       "a big drop when a column is blanked means the model relies on it")

    if store["errors"]: text += ["", "errors:"] + store["errors"]
    if store["raw"]: text += ["", "first raw model output:", store["raw"][0]]
    out = a.out_dir.rstrip("/")
    open(f"{out}/ablations_summary.txt", "w").write("\n".join(text))
    json.dump(report, open(f"{out}/ablations_report.json", "w"), indent=1)
    print("\n" + "\n".join(text) + f"\n\nWrote {out}/ablations_summary.txt (paste this) and {out}/ablations_report.json")


if __name__ == "__main__":
    main()
