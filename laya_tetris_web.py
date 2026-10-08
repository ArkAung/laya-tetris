#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["laya-mlx", "playwright"]
# ///
"""laya-mlx plays the Tetris web app in a visible browser window.

Setup (Apple silicon Mac, with uv):
  uv run --with playwright playwright install chromium    # one time
Keep laya_tetris.py and tetris-env.html in the same folder as this file, then:
  uv run laya_tetris_web.py                     # laya, placement mode
  uv run laya_tetris_web.py --mode primitive
  uv run laya_tetris_web.py --policy greedy     # no model, to check the wiring
  uv run laya_tetris_web.py --delay 0.4 --seed 7

Every action is applied to the web page (what you watch) and to the Python env (what laya
reasons over). The two state lines are compared after each action, so drift is caught at once.
"""
import argparse, pathlib, random, sys, time
from playwright.sync_api import sync_playwright
from laya_tetris import Tetris, play_turn, add_agent_args, turn_kwargs, load_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--html", default=str(pathlib.Path(__file__).with_name("tetris-env.html")))
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between actions")
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--keep-open", action="store_true", help="leave the window open when the game ends")
    add_agent_args(ap)
    a = ap.parse_args()

    agent = None
    if a.policy == "laya":
        agent = load_agent(a.model, a.backend)
    env, rng = Tetris(a.seed), random.Random(a.seed)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=a.headless)
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.goto(pathlib.Path(a.html).resolve().as_uri() + f"?auto=0&seed={a.seed}")
        if page.evaluate("tetris.state().obs") != env.compact():
            sys.exit("Page and Python env start from different states; are the files from the same version?")

        def do(action):
            r = env.step(action)
            web = page.evaluate("a => tetris.step(a)", action)
            if web["obs"] != r["obs"]:
                browser.close()
                sys.exit(f"State drift after '{action}':\n web: {web['obs']}\n py:  {r['obs']}")
            time.sleep(a.delay)
            return r

        n = 0
        while not env.over and n < a.max_moves:
            label = play_turn(env, do, agent, rng, **turn_kwargs(a))
            n += 1
            print(f"#{n:3d} {label:70s} score={env.score} lines={env.lines}")

        print(f"\nfinished: moves={n} score={env.score} lines={env.lines} over={env.over}")
        if a.keep_open and not a.headless:
            input("Press Enter to close the browser...")
        browser.close()


if __name__ == "__main__":
    main()
