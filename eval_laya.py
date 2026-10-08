#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["laya-mlx"]
# ///
"""Offline test: how often does laya pick the same Tetris placement as a simple heuristic?

No browser, no game loop. We generate board states, build candidate placements, ask laya to
choose under several prompt/state variants, and compare against the heuristic and against chance.

  uv run eval_laya.py                    # ~150 states, all variants, both candidate pools
  uv run eval_laya.py --n 60             # quicker
  uv run eval_laya.py --dry-run oracle   # no model: checks the script itself (should score ~100%)

Outputs (in --out-dir, default .):
  laya_eval_summary.txt   <- paste this into the chat
  laya_eval_report.json   <- full detail (states, candidates, picks, raw laya outputs); upload if you can
"""
import argparse, collections, importlib.metadata as md, json, math, platform, random, re, sys, time
from laya_tetris import W, load_agent, Tetris, heuristic, model_state, pick_label, placement_label, option_label, extract_scores, Q_PLACEMENT, Q_PLACEMENT_USER, Q_PLACEMENT_V1

Q_MAIN = Q_PLACEMENT_V1
Q_SHORT = "Which placement is best?"
NUM = ["no", "one", "two", "three", "four"]

# name, state kind, question, candidate text kind, keep best-first order?
VARIANTS = [
    ("base_shuffled", "full", Q_MAIN, "num", False),
    ("current_question", "full", Q_PLACEMENT_USER, "num", False),  # your own wording
    ("base_sorted", "full", Q_MAIN, "num", True),       # best candidate listed first: checks order leakage
    ("no_board", "noboard", Q_MAIN, "num", False),
    ("blank_state", "blank", Q_MAIN, "num", False),     # state removed: does the model use state at all?
    ("plain_language", "full", Q_MAIN, "plain", False),
    ("short_question", "full", Q_SHORT, "num", False),
    ("names_only", "full", Q_MAIN, "names", False),     # no numbers in candidates: must read the board
    ("no_board_ens4", "noboard", Q_MAIN, "num", False, 4),  # no_board, scores averaged over 4 candidate orders
    # candidates without column/rotation words (laya favours column 0 / 7), lettered options, 4 orders averaged:
    ("feat_ens4", "noboard", Q_MAIN, "feat", False, 4),
    ("delta_ens4", "noboard", Q_MAIN, "delta", False, 4),   # same, but holes shown as change caused by this placement
]


# ---------- data ----------
def run_placement(env, p):
    for _ in range(p["rp"]): env.step("cw")
    for _ in range(20):
        if env.cur.x == p["tx"]: break
        if not env.step("left" if env.cur.x > p["tx"] else "right")["valid"]: break
    env.step("drop")


def gen_states(n, seed):
    """Play greedy (75%) / random (25%) placements to get a spread of tidy and messy boards."""
    rng = random.Random(seed); out = []; env = Tetris(rng.getrandbits(32)); t = 0
    while len(out) < n:
        if env.over: env = Tetris(rng.getrandbits(32))
        ps = env.placements()
        if not ps: env = Tetris(rng.getrandbits(32)); continue
        if t % 2 == 0 and len(ps) >= 4: out.append(env.compact())
        run_placement(env, max(ps, key=heuristic) if rng.random() < 0.75 else rng.choice(ps))
        t += 1
    return out


def build_pools(lines, k, seed):
    pools = {"top": [], "mixed": [], "pair": []}
    for i, line in enumerate(lines):
        env = Tetris(0); env.load(line)
        ps = sorted(env.placements(), key=heuristic, reverse=True)
        if len(ps) < 3: continue
        rng = random.Random(seed * 1000 + i)
        pools["top"].append((line, ps[:k]))
        if heuristic(ps[0]) - heuristic(ps[-1]) > 0.5: pools["pair"].append((line, [ps[0], ps[-1]]))  # clearly best vs clearly worst
        pools["mixed"].append((line, [ps[0]] + rng.sample(ps[1:], min(k - 1, len(ps) - 1))))  # best + random others
    return pools


def cand_text(p, kind, pos=0):
    L = chr(65 + pos)   # option letter; its position changes with the shuffle, so averaging cancels letter bias
    if kind == "feat": return f"option {L}: clears {p['lines']}, holes {p['holes']}, height {p['height']}"
    if kind == "delta": return f"option {L}: clears {p['lines']}, holes {p['new_holes']:+d}, height {p['height']}"
    base = f"column {p['col']}, rotated {p['rp']}"
    if kind == "names": return base
    if kind == "num": return placement_label(p)
    l, h, ht = p["lines"], p["holes"], p["height"]
    a = "clears no lines" if l == 0 else f"clears {NUM[l]} line{'s' if l > 1 else ''}"
    b = "leaves no gaps" if h == 0 else f"leaves {h} gap{'s' if h > 1 else ''} in the stack"
    c = "stack stays low" if ht <= 6 else "stack is mid height" if ht <= 12 else "stack gets tall"
    return f"{base}: {a}, {b}, {c}"


def state_text(line, kind):
    if kind == "blank": return "Tetris."
    env = Tetris(0); env.load(line)
    return model_state(env, board=(kind == "full"))


# ---------- laya ----------
class Fake:
    def __init__(self, mode, seed): self.mode, self.rng = mode, random.Random(seed)

    def predict(self, state, qs):
        crit = qs["action"]["criteria"]; sc = []
        for c in crit:
            m = re.search(r"clears (\d+), holes (\d+), height (\d+)", c)
            sc.append(0.76 * int(m[1]) - 0.36 * int(m[2]) - 0.05 * int(m[3]) if (m and self.mode == "oracle") else self.rng.random())
        e = [math.exp(3 * s) for s in sc]; z = sum(e)
        return {"answers": {"action": {"probabilities": {c: v / z for c, v in zip(crit, e)}}},
                "usage": {"state_tokens": len(state) // 3, "state_tokens_dropped": 0, "truncated": False, "input_tokens": len(state) // 3 + 100}}


def ask(agent, state, q, cands, store):
    t = time.time()
    res = agent.predict(state, {"action": {"type": "choice", "instructions": q, "criteria": cands}})
    dt = time.time() - t
    if len(store["raw"]) < 3: store["raw"].append(repr(res)[:1500])
    ans = res["answers"]["action"]
    scores = extract_scores(ans, cands)
    label = max(scores, key=scores.get) if scores else pick_label(ans, cands)
    return label, scores, dt, res.get("usage") or {}


# ---------- metrics ----------
def summarize(recs, pool_data):
    n = len(recs)
    if not n: return {}
    top1 = sum(r["correct"] for r in recs) / n
    chance = sum(r["n_best"] / r["n_cands"] for r in recs) / n
    regret = sum(r["regret"] for r in recs) / n
    rand_regret = sum(r["rand_regret"] for r in recs) / n
    out = dict(n=n, top1=round(top1, 3), chance=round(chance, 3), regret=round(regret, 3), rand_regret=round(rand_regret, 3),
               mean_ms=round(1000 * sum(r["dt"] for r in recs) / n))
    for key, label in (("line_miss", "missed_line_clear"), ("hole_pick", "picked_more_holes")):
        rel = [r for r in recs if r[key] is not None]
        if rel:
            out[label] = round(sum(r[key][0] for r in rel) / len(rel), 3)
            out[label + "_random"] = round(sum(r[key][1] for r in rel) / len(rel), 3)
            out[label + "_n"] = len(rel)
    out["top1_ci95"] = round(1.96 * math.sqrt(max(top1 * (1 - top1), 1e-9) / n), 3)
    us = [r["usage"] for r in recs if r["usage"]]
    if us:
        out["mean_state_tokens"] = round(sum(u.get("state_tokens", 0) for u in us) / len(us))
        out["max_state_tokens"] = max(u.get("state_tokens", 0) for u in us)
        out["n_truncated"] = sum(bool(u.get("truncated")) or u.get("state_tokens_dropped", 0) > 0 or bool(u.get("truncated_questions")) for u in us)
    mrr = [r["rr"] for r in recs if r["rr"] is not None]
    if mrr: out["mrr_of_best"] = round(sum(mrr) / len(mrr), 3)
    pos = collections.Counter(r["pos"] for r in recs)
    out["pick_col"] = [sum(r["col"] == k for r in recs) for k in range(W)]
    out["avail_col"] = [round(sum(r["avail_col"][k] for r in recs), 1) for k in range(W)]
    out["pick_rot"] = [sum(r["rot"] == k for r in recs) for k in range(4)]
    out["avail_rot"] = [round(sum(r["avail_rot"][k] for r in recs), 1) for k in range(4)]
    out["pos_hist"] = [pos.get(i, 0) for i in range(max(r["n_cands"] for r in recs))]
    return out


def make_record(cands, picked_i, order, scores_by_text, texts, dt, usage=None):
    hs = [heuristic(c) for c in cands]; mx = max(hs)
    best = [i for i, h in enumerate(hs) if h >= mx - 1e-9]
    p = cands[picked_i]; n = len(cands)
    rec = dict(orig=picked_i, pos=order.index(picked_i), n_cands=n, n_best=len(best), correct=picked_i in best,
               regret=mx - hs[picked_i], rand_regret=sum(mx - h for h in hs) / n, dt=dt, rr=None,
               line_miss=None, hole_pick=None, usage=usage or {}, col=p["col"], rot=p["rp"],
               avail_col=[sum(c["col"] == k for c in cands) / n for k in range(W)],
               avail_rot=[sum(c["rp"] == k for c in cands) / n for k in range(4)])
    ml = max(c["lines"] for c in cands)
    if ml > 0: rec["line_miss"] = (float(p["lines"] < ml), sum(c["lines"] < ml for c in cands) / n)
    mh, xh = min(c["holes"] for c in cands), max(c["holes"] for c in cands)
    if mh < xh: rec["hole_pick"] = (float(p["holes"] > mh), sum(c["holes"] > mh for c in cands) / n)
    if scores_by_text:
        ranked = sorted(range(n), key=lambda pos: -scores_by_text[texts[pos]])   # by presented position
        rec["rr"] = 1 / (1 + min(ranked.index(order.index(b)) for b in best))
    return rec


GAP_BINS = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1e9)]


def run_roundrobin(agent, pools, n_states, store, fmt="feat"):
    """Use laya only as a two-way comparator: every pair of candidates, both orders, tally the win probabilities."""
    summary, gaps = {}, [[0.0, 0] for _ in GAP_BINS]
    clr = {k: [0.0, 0] for k in ('heuristic prefers the clearing option', '... and it clears 3+ lines', 'heuristic prefers the non-clearing option')}
    for pool in ("top", "mixed"):
        recs = []
        for idx, (line, cands) in enumerate(pools[pool][:n_states]):
            n = len(cands); tot = [0.0] * n; dt = 0.0; usage = {}; st = state_text(line, "noboard")
            for i in range(n):
                for j in range(i + 1, n):
                    pi = 0.0
                    for first, second in ((i, j), (j, i)):
                        t = [option_label(cands[first], 0, fmt), option_label(cands[second], 1, fmt)]
                        label, sc, d, usage = ask(agent, st, Q_MAIN, t, store); dt += d
                        p0 = sc[t[0]] / (sc[t[0]] + sc[t[1]]) if sc else float(label == t[0])
                        pi += (p0 if first == i else 1 - p0) / 2
                    tot[i] += pi; tot[j] += 1 - pi
                    gap = heuristic(cands[i]) - heuristic(cands[j])
                    if (cands[i]["lines"] > 0) != (cands[j]["lines"] > 0):   # exactly one option clears a line
                        c, o, pc = (i, j, pi) if cands[i]["lines"] > 0 else (j, i, 1 - pi)
                        ks = ['heuristic prefers the clearing option'] if heuristic(cands[c]) > heuristic(cands[o]) else ['heuristic prefers the non-clearing option']
                        if ks[0].startswith('heuristic prefers the clearing') and cands[c]["lines"] >= 3: ks.append('... and it clears 3+ lines')
                        for k in ks: clr[k][0] += pc; clr[k][1] += 1
                    if abs(gap) > 1e-9:
                        b = next(k for k, (lo, hi) in enumerate(GAP_BINS) if lo <= abs(gap) < hi)
                        gaps[b][1] += 1; gaps[b][0] += 0.5 if pi == 0.5 else float((pi > 0.5) == (gap > 0))
            pick = max(range(n), key=tot.__getitem__)
            recs.append(make_record(cands, pick, list(range(n)), {f"c{k}": tot[k] for k in range(n)},
                                    [f"c{k}" for k in range(n)], dt, usage))
            if (idx + 1) % 20 == 0: print(f"  {pool}/roundrobin: {idx + 1}/{n_states}", flush=True)
        summary[f"{pool}/roundrobin" + ("" if fmt == "feat" else "_" + fmt)] = summarize(recs, None)
    clr_out = {k: dict(n=n, mean_p_picks_clearing_option=round(s / n, 3) if n else None) for k, (s, n) in clr.items()}
    return summary, clr_out, [dict(gap=f"{lo}-{hi if hi < 1e8 else 'inf'}", n=n, acc=round(c / n, 3) if n else None) for (lo, hi), (c, n) in zip(GAP_BINS, gaps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="number of board states")
    ap.add_argument("--k", type=int, default=8, help="candidates per decision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="aac6fef/laya-mlx")
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"], help="torch needs: uv run --with laya ...")
    ap.add_argument("--variants", default="", help="comma list to run a subset, e.g. base_shuffled,blank_state")
    ap.add_argument("--question", default=None, help="extra variant: your own question text")
    ap.add_argument("--roundrobin", type=int, default=0, help="also test pairwise round-robin on the first N states (56 calls per state)")
    ap.add_argument("--rr-format", default="feat", choices=["feat", "feat2"], help="option text for the round-robin test")
    ap.add_argument("--dry-run", choices=["random", "oracle"], help="fake model, to test this script")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    info = dict(python=sys.version.split()[0], platform=platform.platform(), model=a.model, args=vars(a))
    for pkg in ("laya-mlx", "mlx"):
        try: info[pkg] = md.version(pkg)
        except Exception: pass
    if a.dry_run: agent = Fake(a.dry_run, a.seed)
    else:
        agent = load_agent(a.model, a.backend)
        try:
            tok = getattr(agent, "tokenizer", None)
            if tok is not None:
                s = state_text(gen_states(1, a.seed)[0], "full")
                info["state_tokens_sample"] = len(tok.encode(s)); info["state_chars_sample"] = len(s)
        except Exception as e: info["token_probe_error"] = repr(e)[:200]

    lines = gen_states(a.n, a.seed); pools = build_pools(lines, a.k, a.seed)
    variants = [v for v in VARIANTS + ([("custom", "full", a.question, "num", False)] if a.question else []) if not a.variants or v[0] in a.variants.split(",")]
    print(f"{len(lines)} states, {len(variants)} variants x {len(pools)} pools = "
          f"{len(variants) * len(pools) * len(lines)} model calls", flush=True)

    store = {"raw": [], "errors": []}; summary = {}; details = {}; picks = {}; examples = []
    for pool, data in pools.items():
        for v in variants:
            name, skind, q, ckind, keep_order = v[:5]; reps = v[5] if len(v) > 5 else 1
            recs, det = [], []
            for i, (line, cands) in enumerate(data):
                tot = [0.0] * len(cands); dt = 0.0; usage = {}; bad = False
                for r in range(reps):   # reps > 1: average scores over different candidate orders (cancels position bias)
                    order = list(range(len(cands)))
                    if not keep_order: random.Random(a.seed * 7919 + i + 100003 * r).shuffle(order)  # r=0 matches other variants
                    texts = [cand_text(cands[j], ckind, pos) for pos, j in enumerate(order)]
                    if len(set(texts)) != len(texts): bad = True; break
                    try:
                        label, scores, dt1, usage = ask(agent, state_text(line, skind), q, texts, store)
                    except Exception as e:
                        store["errors"].append(f"{name}/{pool}/#{i}: {e!r}"[:400]); bad = True; break
                    dt += dt1
                    if r == 0: order0, texts0, label0, scores0 = order, texts, label, scores
                    if scores:
                        for t, j in zip(texts, order): tot[j] += scores[t]
                    else: tot[order[texts.index(label)]] += 1
                if bad:
                    if len(store["errors"]) >= 3: break
                    continue
                pi = max(range(len(cands)), key=tot.__getitem__)
                label = texts0[order0.index(pi)]
                sbt = {texts0[pos]: tot[order0[pos]] for pos in range(len(cands))} if scores0 else None
                scores = sbt
                order, texts = order0, texts0
                rec = make_record(cands, pi, order, sbt, texts, dt, usage); recs.append(rec)
                det.append(dict(i=i, state=line, cands=texts, picked=label, correct=rec["correct"], regret=round(rec["regret"], 3),
                                scores=None if not scores else [round(scores[t] / reps, 4) for t in texts]))
                picks.setdefault((pool, name), {})[i] = pi
                if (i + 1) % 25 == 0: print(f"  {pool}/{name}: {i + 1}/{len(data)}", flush=True)
            if len(store["errors"]) >= 3:
                print("Stopping: repeated errors:\n" + "\n".join(store["errors"])); break
            summary[f"{pool}/{name}"] = summarize(recs, data); details[f"{pool}/{name}"] = det
            print(f"{pool:5s} {name:15s} top1={summary[f'{pool}/{name}'].get('top1')} "
                  f"chance={summary[f'{pool}/{name}'].get('chance')}", flush=True)
        if len(store["errors"]) >= 3: break

    gap_acc = None; clear_stats = None
    if a.roundrobin and len(store["errors"]) < 3:
        print(f"round-robin on {a.roundrobin} states x 2 pools (about {a.roundrobin * 2 * 56} calls)", flush=True)
        s_rr, clear_stats, gap_acc = run_roundrobin(agent, pools, a.roundrobin, store, a.rr_format); summary.update(s_rr)
    # reference rows that need no model: what a pure reader of the candidate text could reach
    for pool, data in pools.items():
        rr = random.Random(a.seed + 1); recs = []
        for i, (line, cands) in enumerate(data):
            order = list(range(len(cands))); random.Random(a.seed * 7919 + i).shuffle(order)
            pi = max(range(len(cands)), key=lambda j: (cands[j]["lines"], -cands[j]["holes"], -cands[j]["height"], rr.random()))
            recs.append(make_record(cands, pi, order, None, [], 0.0))
        summary[f"{pool}/RULE_visible"] = summarize(recs, data)

    # how much do picks depend on state? agreement of each variant's pick with base_shuffled (same candidates, same order)
    agree = {}
    for (pool, name), pk in picks.items():
        base = picks.get((pool, "base_shuffled"))
        if base and name != "base_shuffled" and name in ("no_board", "blank_state", "short_question", "current_question", "custom", "no_board_ens4"):
            common = [i for i in pk if i in base]
            if common: agree[f"{pool}/{name}_vs_base"] = round(sum(pk[i] == base[i] for i in common) / len(common), 3)

    # worst misses for the mixed pool baseline (the most diagnostic single view)
    worst = sorted(details.get("mixed/base_shuffled", []), key=lambda d: -d["regret"])[:3]

    report = dict(info=info, summary=summary, pick_agreement_between_variants=agree, pairwise_accuracy_by_heuristic_gap=gap_acc, clear_preference=clear_stats, raw_samples=store["raw"],
                  errors=store["errors"], worst_mixed_base=worst, details=details)
    out = a.out_dir.rstrip("/")
    json.dump(report, open(f"{out}/laya_eval_report.json", "w"), indent=1)

    L = ["LAYA TETRIS OFFLINE EVAL", json.dumps({k: v for k, v in info.items() if k != "args"}), "args: " + json.dumps(info["args"]), ""]
    L.append("pool/variant        n  top1(±95%) chance  regret rand_regret  missLine(rand)  moreHoles(rand)  MRR   ms  stateTok(mean/max/trunc)")
    for k, s in summary.items():
        if not s: continue
        ml = f"{s.get('missed_line_clear', '-')}({s.get('missed_line_clear_random', '-')})"
        mh = f"{s.get('picked_more_holes', '-')}({s.get('picked_more_holes_random', '-')})"
        L.append(f"{k:24s}{s['n']:3d} {s['top1']:5.2f}±{s['top1_ci95']:.2f} {s['chance']:5.2f} {s['regret']:7.3f} {s['rand_regret']:10.3f}  "
                 f"{ml:15s} {mh:16s} {s.get('mrr_of_best', '-')}  {s['mean_ms']}  "
                 f"{s.get('mean_state_tokens', '-')}/{s.get('max_state_tokens', '-')}/{s.get('n_truncated', '-')}")
    L += ["", "RULE_visible = no model: picks by (lines, holes, height) from the candidate text only; it is the realistic ceiling for reading those numbers.",
          "Plain heuristic also uses stack bumpiness/total height, which the text omits, so top1 below 1.0 is expected even for a perfect reader."]
    L += ["", "position histogram of picks (shuffled variants; flat = no position bias):"]
    L += [f"  {k}: {s['pos_hist']}" for k, s in summary.items() if s and k.endswith("base_shuffled")]
    L += ["", "what laya picks by column 0-9 / rotation 0-3 vs what was on offer (picked | offered), base_shuffled:"]
    for k, s in summary.items():
        if s and k.endswith("base_shuffled"):
            L += [f"  {k}: col {s['pick_col']} | {s['avail_col']}", f"  {k}: rot {s['pick_rot']} | {s['avail_rot']}"]
    if gap_acc: L += ["", "pairwise accuracy by |heuristic gap| (round-robin; 0.5 = coin flip, how fine can laya tell options apart):"] + [f"  gap {g['gap']:9s} n={g['n']:5d} acc={g['acc']}" for g in gap_acc]
    if clear_stats:
        L += ["", "does laya favor line clears? mean probability it prefers the clearing option over a non-clearing one",
              "(ideal: ~1.0 in the first two rows, ~0.0 in the last; 0.5 = indifferent):"]
        L += [f"  {k:45s} n={v['n']:5d} p={v['mean_p_picks_clearing_option']}" for k, v in clear_stats.items()]
    L += ["", "pick agreement with base_shuffled (1.0 = state/wording changes nothing):", json.dumps(agree)]
    L += ["", "raw laya outputs (first 3):"] + store["raw"][:3]
    if store["errors"]: L += ["", "errors:"] + store["errors"]
    L += ["", "worst misses, mixed pool, base_shuffled:"] + [json.dumps(w) for w in worst[:2]]
    text = "\n".join(L)
    open(f"{out}/laya_eval_summary.txt", "w").write(text)
    print("\n" + text + f"\n\nWrote {out}/laya_eval_summary.txt (paste this) and {out}/laya_eval_report.json")


if __name__ == "__main__":
    main()
