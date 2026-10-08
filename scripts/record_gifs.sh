#!/usr/bin/env bash
# Records one GIF per approach, all from seed 1 so the piece sequence is identical and the GIFs compare side by side.
# usage: scripts/record_gifs.sh PATH_OR_HUB_ID_OF_THE_TUNED_MODEL   (run from the repo root, on the Mac with laya-mlx)
set -euo pipefail
TUNED="${1:?usage: scripts/record_gifs.sh PATH_OR_HUB_ID_OF_THE_TUNED_MODEL}"
OUT=docs/img

uv run record_gif.py --policy random --seed 1 --max-moves 60 \
  --label "Random pick from the same eight options" --out $OUT/random.gif

uv run record_gif.py --policy laya --compare choice --seed 1 --max-moves 60 \
  --label "Laya, zero-shot, one question over eight options" --out $OUT/one-question.gif

uv run record_gif.py --policy laya --compare pairwise --seed 1 --max-moves 80 \
  --label "Laya, zero-shot, comparing every pair" --out $OUT/pairwise.gif

uv run record_gif.py --policy laya --mode primitive --seed 1 --max-moves 300 \
  --label "Laya, zero-shot, raw moves with the board as the state" --out $OUT/raw-moves.gif

uv run record_gif.py --policy greedy --seed 1 --max-moves 50 \
  --label "The heuristic that taught the final model" --out $OUT/heuristic.gif

uv run record_gif.py --policy laya --model "$TUNED" --compare letters --seed 1 --max-moves 50 \
  --label "Laya after fine-tuning" --out $OUT/tuned.gif

echo "Now rebuild the page:  python scripts/build_post.py"
