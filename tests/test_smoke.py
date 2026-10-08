"""Smoke tests. No model needed: they check the game, the state format, the dataset rows and the letters pipeline."""
import math, os, random, re, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_tetris import Tetris, play_turn, tetris_state_text, LETTERS          # noqa: E402
from make_tetris_dataset import make_row, soft_label                           # noqa: E402


class OracleLetters:
    """Stands in for a fine-tuned model: reads the option table in the state text and prefers the best row."""
    _tetris_questions = None

    def predict(self, state, qs):
        rows = re.findall(r"^([A-H]): clears (\d+), holes (\d+), bumpiness (\d+), height (\d+)", state, re.M)
        sc = {r[0]: 0.76 * int(r[1]) - 0.36 * int(r[2]) - 0.18 * int(r[3]) / 4 for r in rows}
        z = sum(math.exp(3 * v) for v in sc.values())
        return {"answers": {"placement": {"probabilities": {L: math.exp(3 * v) / z for L, v in sc.items()}}}}


def play(env, policy, agent=None, moves=150, **kw):
    rng = random.Random(1)
    for _ in range(moves):
        if env.over: break
        play_turn(env, env.step, agent, rng, policy=policy, **kw)
    return env


class Tests(unittest.TestCase):
    def test_state_line_round_trips(self):
        env = Tetris(5); rng = random.Random(1)
        for a in rng.choices(["left", "right", "cw", "drop", "hold", "down"], k=200):
            env.step(a)
            line = env.compact(); copy = Tetris(0); copy.load(line)
            self.assertEqual(copy.compact(), line)

    def test_same_seed_same_game(self):
        a, b = Tetris(42), Tetris(42)
        for act in ["drop", "left", "cw", "drop", "hold", "right", "drop"]:
            a.step(act); b.step(act)
        self.assertEqual(a.compact(), b.compact())

    def test_greedy_policy_plays_well(self):
        env = play(Tetris(1), "greedy")
        self.assertFalse(env.over)
        self.assertGreater(env.lines, 40)

    def test_dataset_rows_are_valid(self):
        env = Tetris(3); rng = random.Random(0); n = 0
        for _ in range(30):
            row = make_row(env, 8, rng, 0.1)
            if row:
                probs = row["answers"]["placement"]
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
                self.assertTrue(set(probs) <= set(LETTERS))
                self.assertEqual(row["state"].count("\n"), len(probs))   # header line + one line per option
                n += 1
            play_turn(env, env.step, None, rng, policy="greedy")
        self.assertGreater(n, 20)

    def test_soft_label_splits_ties(self):
        p = soft_label([1.0, 1.0, 0.0], 0.1)
        self.assertAlmostEqual(p[0], p[1]); self.assertGreater(p[0], 0.49)

    def test_letters_pipeline_with_stand_in_model(self):
        env = play(Tetris(2), "laya", OracleLetters(), compare="letters")
        self.assertFalse(env.over)
        self.assertGreater(env.lines, 40)

    def test_state_text_shape(self):
        env = Tetris(1); ps = env.placements()[:3]
        text = tetris_state_text(env, ps)
        self.assertIn("Options:", text)
        self.assertEqual(len(text.splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
