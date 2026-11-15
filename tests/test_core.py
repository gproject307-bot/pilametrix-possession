"""Unit tests for the parts that do not need the match data.

    python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import features as F  # noqa: E402
from possession import hmm, io, model  # noqa: E402


class Viterbi(unittest.TestCase):
    def test_follows_strong_evidence(self):
        scores = np.zeros((60, 2))
        scores[:30, 0] = 1.0
        scores[30:, 1] = 1.0
        path = hmm.viterbi(scores, np.array([[0.0, 5.0], [5.0, 0.0]]))
        self.assertTrue((path[:30] == 0).all())
        self.assertTrue((path[30:] == 1).all())

    def test_ignores_a_blip_cheaper_than_two_switches(self):
        scores = np.zeros((60, 2))
        scores[:, 0] = 1.0
        scores[28:31] = (0.0, 1.0)  # three frames of contrary evidence
        path = hmm.viterbi(scores, np.array([[0.0, 5.0], [5.0, 0.0]]))
        self.assertTrue((path == 0).all())

    def test_keeps_state_through_silence(self):
        scores = np.zeros((90, 2))
        scores[:10, 1] = 1.0  # evidence only at the start, then nothing
        path = hmm.viterbi(scores, np.array([[0.0, 5.0], [5.0, 0.0]]))
        self.assertTrue((path == 1).all())


class SwitchingViterbi(unittest.TestCase):
    def test_constant_costs_agree_with_the_plain_decoder(self):
        rng = np.random.RandomState(1)
        scores = rng.randn(400, 3)
        plain = hmm.viterbi(scores, np.array([[0, 3.0, 2.5], [3.0, 0, 2.5], [2.5, 2.5, 0]]))
        varying = hmm.viterbi_switching(scores, np.full(400, 3.0), 2.5)
        self.assertTrue((plain == varying).all())

    def test_a_change_waits_for_a_cheap_frame(self):
        scores = np.zeros((40, 3))
        scores[:, 2] = -10.0          # never dead
        scores[:20, 0] = 0.3          # home for the first half ...
        scores[20:, 1] = 0.3          # ... away for the second
        cost = np.full(40, 50.0)
        cost[25] = 1.0                # the only affordable moment to change
        path = hmm.viterbi_switching(scores, cost, 100.0)
        self.assertEqual(path.tolist(), [0] * 25 + [1] * 15)


class Helpers(unittest.TestCase):
    def test_runs(self):
        starts, stops = F.runs(np.array([0, 1, 1, 0, 1], dtype=bool))
        self.assertEqual(starts.tolist(), [1, 4])
        self.assertEqual(stops.tolist(), [3, 5])

    def test_box_mean_ignores_nan(self):
        values = np.array([1.0, np.nan, 3.0, np.nan, np.nan])
        out = F.box_mean(values, 1)
        self.assertEqual(out[1], 2.0)
        self.assertEqual(out[0], 1.0)
        self.assertTrue(np.isnan(out[4]))

    def test_ball_out_and_at_rest(self):
        n = 100
        bx = np.full(n, 50.0, dtype=np.float32)
        by = np.full(n, 30.0, dtype=np.float32)
        bx[60:] = -3.0  # off the pitch behind the goal line
        ball = F.ball_features(bx, by)
        self.assertTrue(ball["at_rest"][30])
        self.assertEqual(ball["outside"][30], 0.0)
        self.assertAlmostEqual(float(ball["outside"][80]), 3.0, places=5)

    def test_nearest_by_team(self):
        bx = np.array([10.0, np.nan], dtype=np.float32)
        by = np.array([10.0, np.nan], dtype=np.float32)
        px = np.array([[11.0, 14.0], [11.0, 14.0]], dtype=np.float32)
        py = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
        home, away = F.nearest_by_team(bx, by, px, py, np.array([0, 1], dtype=np.int8))
        self.assertAlmostEqual(float(home[0]), 1.0)
        self.assertAlmostEqual(float(away[0]), 4.0)
        self.assertTrue(np.isnan(home[1]))


class Decode(unittest.TestCase):
    def test_short_stoppage_between_same_side_is_absorbed(self):
        path = np.array([0] * 20 + [2] * 5 + [0] * 20, dtype=np.int8)
        out = model.absorb_short_stoppages(path, 10)
        self.assertTrue((out == 0).all())

    def test_short_stoppage_between_sides_is_split(self):
        path = np.array([0] * 20 + [2] * 6 + [1] * 20, dtype=np.int8)
        out = model.absorb_short_stoppages(path, 10)
        self.assertEqual(out.tolist(), [0] * 23 + [1] * 23)

    def test_short_stoppage_at_the_start_takes_the_next_label(self):
        path = np.array([2] * 3 + [1] * 20, dtype=np.int8)
        self.assertTrue((model.absorb_short_stoppages(path, 10) == 1).all())

    def test_long_stoppage_is_kept(self):
        path = np.array([0] * 20 + [2] * 30 + [1] * 20, dtype=np.int8)
        self.assertTrue((model.absorb_short_stoppages(path, 10) == path).all())

    def test_team_confidence_does_not_change_dead_vs_alive(self):
        n = 100
        evidence = {"team": np.zeros(n), "dead": np.zeros(n), "visible": np.ones(n, dtype=bool)}
        sources = {"team_self": np.full(n, 0.9), "dead_self": np.full(n, 0.5)}
        evidence["team"][:10] = 1.0
        scores = model.frame_scores(evidence, sources)
        best_alive = scores[:, :2].max(axis=1)
        self.assertTrue(np.allclose(best_alive, scores[:, 2]))


class Kickoffs(unittest.TestCase):
    def _tag(self, kind, start):
        return {"code": "SET PIECES", "type": kind, "start": start, "end": start + 125,
                "team": -1, "tags": {}}

    def test_kick_is_where_the_ball_appears_on_the_centre_spot(self):
        n = 60000
        second_half = 30000
        start = second_half + 3000
        match = {"bx": np.full(n, np.nan, dtype=np.float32),
                 "by": np.full(n, np.nan, dtype=np.float32),
                 "n_frames": n, "halves": [(100, 29000), (second_half, n - 1)],
                 "tags": [self._tag("kick off (start)", start),
                          self._tag("kick off (after goal)", start + 20000)]}
        match["bx"][start + 80:start + 90] = io.PITCH[0] / 2.0
        match["by"][start + 80:start + 90] = io.PITCH[1] / 2.0
        kicks = [frame for _, frame in model.kickoff_frames(match)]
        self.assertEqual(kicks, [start + 80, start + 20000 + 80])  # second one by the median

        forced = model.forced_dead(match)
        self.assertTrue(forced[second_half:start + 80].all())
        self.assertFalse(forced[start + 80])
        self.assertTrue(forced[start + 20000 - 750:start + 20000 + 80].all())
        self.assertFalse(forced[start + 20000 - 751])


class ExternalLabels(unittest.TestCase):
    def test_ownership_follows_on_ball_events_and_blank_ball_is_dead(self):
        from possession import external
        n = 100
        blank = np.zeros(n, dtype=bool)
        blank[40:60] = True          # a stoppage
        blank[70:73] = True          # a three-frame tracking glitch, not a stoppage
        events = [(0, 0, "SET PIECE"), (0, 0, "PASS"), (20, 1, "RECOVERY"),
                  (25, 0, "CHALLENGE"), (26, 0, "BALL LOST"), (60, 0, "SET PIECE")]
        labels = external._labels(n, [(0, n - 1)], blank, events)
        self.assertTrue((labels[:20] == 0).all())
        self.assertTrue((labels[20:40] == 1).all())   # challenge / ball lost do not move it
        self.assertTrue((labels[40:60] == 2).all())
        self.assertTrue((labels[60:] == 0).all())      # the glitch stays live


class OwnerFeatures(unittest.TestCase):
    def test_twin_is_the_mirror_image(self):
        n = 400
        rng = np.random.RandomState(0)
        ball = F.ball_features(np.cumsum(rng.randn(n)).astype(np.float32) + 50,
                               np.cumsum(rng.randn(n)).astype(np.float32) + 30)
        d_home, d_away = rng.rand(n) * 6, rng.rand(n) * 6
        X, twin, names = F.owner_features(ball, d_home, d_away)
        X2, twin2, _ = F.owner_features(ball, d_away, d_home)
        self.assertTrue(np.allclose(twin, X2, equal_nan=True))
        self.assertTrue(np.allclose(X, twin2, equal_nan=True))
        self.assertEqual(X.shape[1], len(names))


class TouchPoints(unittest.TestCase):
    def test_crossing_of_the_paths_recovers_a_rounded_off_corner(self):
        from scipy.ndimage import gaussian_filter1d
        n = 200
        t = np.arange(n, dtype=np.float64)
        # Ball travels along +x at 10 m/s, is redirected at frame 100 to +y at 10 m/s.
        x = np.where(t < 100, 20 + 0.4 * t, 60.0)
        y = np.where(t < 100, 30.0, 30 + 0.4 * (t - 100))
        bx = gaussian_filter1d(x, 7).astype(np.float32)
        by = gaussian_filter1d(y, 7).astype(np.float32)
        ball = F.ball_features(bx, by)
        touches, _ = F.ball_touches(ball, span=7)
        k = int(np.argmin(np.abs(touches - 100)))   # ignore artefacts at the ends of the track
        self.assertLessEqual(abs(int(touches[k]) - 100), 1)
        px, py = F.touch_points(ball, bx, by, touches, 7)
        tracked = np.hypot(bx[touches[k]] - 60.0, by[touches[k]] - 30.0)
        rebuilt = np.hypot(px[k] - 60.0, py[k] - 30.0)
        self.assertGreater(tracked, 1.5)   # the smoothed track cuts the corner
        self.assertLess(rebuilt, 0.5)      # the crossing of the two paths does not


class SequenceModel(unittest.TestCase):
    def test_swapping_the_sides_flips_the_output_exactly(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from possession import refine
        torch.manual_seed(0)
        net = refine._network(c_in=5, width=8, layers=3, dropout=0.0).eval()
        anti = torch.randn(2, 2, 200)
        sym = torch.randn(2, 3, 200)
        with torch.no_grad():
            one = refine._odd(net, anti, sym)
            other = refine._odd(net, -anti, sym)
        self.assertTrue(torch.equal(one, -other))

    def test_weights_shrink_with_possession_length(self):
        from possession import transfer
        labels = np.array([0] * 25 + [1] * 250 + [2] * 50 + [0] * 125, dtype=np.int8)
        match = {"labels": labels, "n_frames": len(labels), "halves": [(0, len(labels) - 1)]}
        w = transfer.spell_weights(match, 1.0)
        self.assertAlmostEqual(float(w[0]), 5.0)       # 1-second possession: capped at 5
        self.assertAlmostEqual(float(w[30]), 0.5)      # 10-second possession
        self.assertAlmostEqual(float(w[280]), 1.0)     # dead frames are left alone
        self.assertAlmostEqual(float(w[-1]), 1.0)      # a typical 5-second possession
        self.assertTrue((transfer.spell_weights(match, 0.0) == 1.0).all())


class Replay(unittest.TestCase):
    def test_export_writes_what_the_viewer_reads(self):
        import json
        import replay
        n, slots = 60, 3
        px = np.full((n, slots), np.nan)
        py = np.full((n, slots), np.nan)
        px[:, 0], py[:, 0] = 10.25, 20.0
        px[:30, 2], py[:30, 2] = 50.0, 34.0
        bx = np.full(n, np.nan)
        by = np.full(n, np.nan)
        bx[5], by[5] = 52.5, 33.94
        match = {"px": px, "py": py, "bx": bx, "by": by, "n_frames": n,
                 "team": np.array([0, 1, 1], dtype=np.int8), "halves": [(0, 29), (30, n - 1)]}
        predicted = np.array([0] * 20 + [2] * 20 + [1] * 20, dtype=np.uint8)
        dead = np.linspace(0.0, 1.0, n)
        with tempfile.TemporaryDirectory() as root:
            folder = replay.export(os.path.join(root, "m"), match, "T", "S", ["Home", "Away"],
                                   predicted=predicted, chance={"dead": dead})
            with open(os.path.join(folder, "meta.json")) as handle:
                meta = json.load(handle)
            table = np.fromfile(os.path.join(folder, "positions.bin"), dtype="<i2")
            series = np.fromfile(os.path.join(folder, "series.bin"), dtype=np.uint8)
            self.assertTrue(os.path.isfile(os.path.join(folder, "index.html")))
            self.assertTrue(os.path.isfile(os.path.join(root, "index.html")))
        self.assertEqual(meta["positions"]["columns"], 2 * slots + 2)
        table = table.reshape(n, -1)
        self.assertEqual(table[0, 0], 102)                    # 10.25 m, in decimetres
        self.assertEqual(table[0, 2], replay.MISSING)         # slot 1 never seen
        self.assertEqual(table[40, 4], replay.MISSING)        # slot 2 gone after frame 29
        self.assertEqual(table[5, -2], 525)
        self.assertEqual(table[6, -2], replay.MISSING)
        blocks = series.reshape(len(meta["series"]["names"]), n)
        names = meta["series"]["names"]
        self.assertTrue((blocks[names.index("pred")] == predicted).all())
        self.assertTrue((blocks[names.index("truth")] == replay.NONE).all())
        self.assertEqual(blocks[names.index("p_dead")][-1], 250)
        self.assertTrue((blocks[names.index("p_home")] == replay.NO_VALUE).all())
        self.assertEqual(meta["kickoffs"], [0, 30])

    def test_labels_are_read_back_as_codes(self):
        import replay
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "s.csv")
            with open(path, "w") as handle:
                handle.write("frame,label\n0,X\n1,H\n2,a\n3, D\n")
            codes = replay.read_labels(path, 5)
        self.assertEqual(codes.tolist(), [3, 0, 1, 2, 3])


class Submission(unittest.TestCase):
    def test_written_file_is_well_formed(self):
        labels = np.array(["X", "H", "A", "D"])
        with tempfile.TemporaryDirectory() as folder:
            path = io.write_submission(os.path.join(folder, "s.csv"), labels)
            with open(path) as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines, ["frame,label", "0,X", "1,H", "2,A", "3,D"])


if __name__ == "__main__":
    unittest.main()
