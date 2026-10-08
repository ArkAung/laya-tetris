# Teaching a decision model to play Tetris

[Laya](https://huggingface.co/convaiinnovations/laya) is a 421M-parameter *decision model*: it reads a text state and a typed
question, and answers in a single forward pass with calibrated probabilities. It doesn't generate text. Out of the box it plays
Tetris like a coin flip. This repo is the record of finding out why, and of the fine-tune that fixed it.

The write-up of the whole journey: **https://arkaung.github.io/laya-tetris/**

## Result

150-move games, seeds 1-10 unless noted (a game ends early if the stack reaches the top):

| Player | Lines per game | Survived 150 moves |
|---|---|---|
| Random placement | 0.2 | 0 / 10 (dead by move ~28) |
| Laya, zero-shot, one 8-way question (lettered numbers, 4 orderings averaged) | 0.1 | 0 / 10 (dead by move ~32) |
| Laya, zero-shot, 28 pairwise comparisons per move | 4.2 | 0 / 5 (dead by move ~47) |
| **Laya, fine-tuned (this repo)** | **56.6** | **10 / 10** |
| Heuristic teacher (greedy) | 56.4 | 10 / 10 |

Read this table with the caveats in mind:

- 150 pieces fill 600 cells, so **60 lines is the ceiling**. The test can't separate the tuned model from the teacher. Longer games are the fair comparison.
- The tuned model agrees with the teacher on essentially every move, so **it can't be better than the teacher**. It is a distilled heuristic.
- It never sees the board. The code enumerates every landing spot, computes four numbers for each (lines cleared, holes, bumpiness, height), and shows the best eight as a text table. The model learned to read that table.

## Try it (Apple silicon Mac, [uv](https://docs.astral.sh/uv/))

```bash
# no model needed: the heuristic player, to check your setup
uv run laya_tetris.py --policy greedy --games 3

# the fine-tuned model (downloads from the Hub)
uv run laya_tetris.py --model aaung/laya-tetris-placement --compare letters --games 10 --max-moves 150

# watch it play in your browser (serves the page on http://localhost:8765)
uv run laya_tetris_ws.py --model aaung/laya-tetris-placement --compare letters --delay 0.15
```

You can also just open `tetris-env.html` in a browser and play it yourself. The whole game state fits on one line, for example
`TETRIS1 s=0 l=0 o=0 c=O0@4,0 h=- hc=1 q=JITZL g=S rs=2399460328 b=`, and every move is a named action
(`left right cw ccw down drop hold tick noop`). The page exposes `window.tetris.step(...)`, a `postMessage` API and a WebSocket
client, so any external system can drive it.

## How the pieces fit

1. **The game** (`laya_tetris.py`, `tetris-env.html`): one-line state, discrete actions, seeded and deterministic. The Python
   copy and the web copy are bit-identical (same RNG, same rules).
2. **The agent loop** (`play_turn` in `laya_tetris.py`): build state + question + candidates, ask the model, execute the pick
   as rotate / move / drop actions. The terminal runner, the WebSocket runner and the browser-automation runner all share it.
3. **The model:** Laya, fine-tuned with LoRA in [LayaStudio](https://github.com/biplovgautam/LayaStudio) on rows made by
   `make_tetris_dataset.py`. Each row is a state text containing the option table plus the teacher's preference over options
   A to H as a soft label.

## Reproduce

```bash
# 1. make the training set (writes tetris_data/train.jsonl and questions.json)
uv run make_tetris_dataset.py --n 8000 --out tetris_data

# 2. fine-tune in LayaStudio (upload train.jsonl and questions.json), then play the result
uv run laya_tetris.py --model path/to/model --compare letters --games 10 --max-moves 150

# 3. the offline experiments behind the story (zero-shot prompt variants, pairwise round-robin)
uv run eval_laya.py --n 500
uv run eval_laya.py --n 100 --variants feat_ens4 --roundrobin 100 --rr-format feat2

# 4. re-run each failing factor at several levels (state, option text, option count, averaging, raw moves, what the tuned model reads)
uv run ablations.py --n 300
uv run ablations.py --tuned path/to/model --only F
```

`data_sample/` holds 60 rows so you can see the format without generating anything. `notes/journey-notes.md` has every number
from the experiments, in order. Unit tests (no model needed): `python -m unittest discover -s tests -v`.

## Limitations

- The tuned model depends on our code building the option table. It is not a general Tetris player, and it was only tested
  with the table format it was trained on.
- The teacher is a one-move-lookahead heuristic. Nothing here goes beyond it.
- Tested on a Mac with `laya-mlx` only.
- A harder follow-up, not done: put the board itself in the state and let the model choose raw moves, as in the
  [Snake demo](https://huggingface.co/madhavbiplov/laya-snake-mlx).

## License

Apache-2.0. See `LICENSE`.
