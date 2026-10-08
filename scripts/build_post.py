#!/usr/bin/env python3
"""Build docs/index.html from scripts/post_template.html. Run from the repo root:  python scripts/build_post.py

Fills in the charts and the two example blocks, and adds a GIF only if its file exists in docs/img/, so the page never shows a
broken image. Missing GIFs are listed at the end; make them with scripts/record_gifs.sh and run this again.
"""
import html, json, os, re, sys

sys.path.insert(0, ".")
from laya_tetris import Tetris, heuristic, model_state, placement_label, execute_placement

BLUE, GREY = "#2f5bd8", "#c9d1dc"
esc = lambda s: html.escape(str(s))


def vbars(title, desc, labels, series, ymax, w=640, h=270, ticks=None):
    L, R, T, B = 40, 10, 34, 34
    pw, ph = w - L - R, h - T - B
    gw, n = pw / len(labels), len(series)
    bw = gw * 0.74 / n
    out = [f'<svg class="ch" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d"><title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>']
    for tv in ticks:
        y = T + ph - ph * tv / ymax
        out.append(f'<line x1="{L}" x2="{w-R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e9ef"/><text x="{L-6}" y="{y+4:.1f}" text-anchor="end">{tv:g}</text>')
    for gi, lab in enumerate(labels):
        gx = L + gi * gw + (gw - bw * n) / 2
        for si, (name, vals, color) in enumerate(series):
            v = vals[gi]; bh = ph * v / ymax; x = gx + si * bw; y = T + ph - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-1.5:.1f}" height="{bh:.1f}" rx="2" fill="{color}"/>')
        out.append(f'<text x="{L+gi*gw+gw/2:.1f}" y="{h-12}" text-anchor="middle">{esc(lab)}</text>')
    lx = L
    for name, _, color in series:
        out.append(f'<rect x="{lx}" y="8" width="11" height="11" rx="2" fill="{color}"/><text x="{lx+16}" y="18">{esc(name)}</text>')
        lx += 24 + 7 * len(name)
    return "".join(out) + "</svg>"


chart_cols = vbars("Which column was picked, against random picking",
                   "Picks per column 0 to 9: 137, 53, 34, 44, 21, 20, 63, 87, 32, 9. Random picking would give about 59, 57, 58, 57, 57, 55, 54, 52, 43, 8.",
                   [str(i) for i in range(10)],
                   [("what it picked", [137, 53, 34, 44, 21, 20, 63, 87, 32, 9], BLUE), ("random picking", [59, 57, 58, 57, 57, 55, 54, 52, 43, 8], GREY)],
                   150, ticks=[0, 50, 100, 150])

# the prompt the model saw in the first attempt (board as letters, options with column and rotation words)
e = Tetris(11)
for _ in range(13): execute_placement(e, e.step, max(e.placements(), key=heuristic))
opts = sorted(e.placements(), key=heuristic, reverse=True)[:8]
first_prompt = (model_state(e, True) + "\n\nWhere should the piece be placed? Prefer clearing lines, then no holes, then a low flat surface.\n"
                + "\n".join(placement_label(p) for p in opts[:4]) + "\n...")

# a real training row, the one the dataset card uses
rows = [json.loads(l) for l in open("data_sample/train.jsonl")]
row = next(r for r in rows if re.search(r"clears [1-9]", r["state"]) and max(r["answers"]["placement"].values()) > 0.9)
winner = max(row["answers"]["placement"], key=row["answers"]["placement"].get)

GIFS = {  # file, caption, alt text
    "random": ("random.gif", "A random pick from the same eight options.", "A Tetris game played by random choice, which fills the board within about thirty moves."),
    "one-question": ("one-question.gif", "Laya, zero-shot, asked one question over eight options.", "A Tetris game played by the untuned model asking one question per piece."),
    "pairwise": ("pairwise.gif", "Laya, zero-shot, comparing every pair of options.", "A Tetris game played by the untuned model comparing every pair of options."),
    "raw-moves": ("raw-moves.gif", "Laya, zero-shot, choosing raw moves with the board as the state.", "A Tetris game in which the untuned model chooses raw moves such as rotate and move down."),
    "heuristic": ("heuristic.gif", "The heuristic that taught the final model.", "A Tetris game played by the hand-written heuristic, clearing lines steadily."),
    "tuned": ("tuned.gif", "Laya after fine-tuning.", "A Tetris game played by the fine-tuned model, clearing lines steadily."),
}
missing = []


def gifs(keys):
    figs = []
    for k in keys:
        fn, cap, alt = GIFS[k]
        if os.path.exists(f"docs/img/{fn}"):
            figs.append(f'<figure><img src="img/{fn}" alt="{esc(alt)}" loading="lazy"><figcaption>{esc(cap)}</figcaption></figure>')
        else:
            missing.append(fn)
    return '<div class="gifs">' + "".join(figs) + "</div>" if figs else ""


page = open("scripts/post_template.html").read()
for key, val in {"{{CHART_COLS}}": chart_cols, "{{FIRST_PROMPT}}": esc(first_prompt), "{{ROW_STATE}}": esc(row["state"]),
                 "{{WINNER}}": winner, "{{GIFS_FAIL}}": gifs(["random", "one-question", "pairwise", "heuristic"]),
                 "{{GIFS_RAW}}": gifs(["raw-moves"]), "{{GIFS_GOOD}}": gifs(["tuned", "heuristic"])}.items():
    page = page.replace(key, val)
open("docs/index.html", "w").write(page)
print(f"wrote docs/index.html ({len(page) // 1024} KB)")
if missing: print("GIFs not found yet (their figures are left out until you make them):", ", ".join(sorted(set(missing))))
