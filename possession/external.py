"""Other matches with known answers: Metrica Sports' public sample data.

https://github.com/metrica-sports/sample-data -- three anonymised matches with corrected tracking
and event data, published by the challenge organiser for the analytics community. They are *other
games*, so the challenge rules allow using them, and they are the only place a method can be
compared against real labels over a whole match.

Ground truth is derived mechanically, the way the challenge describes its own:

- **dead** where the corrected tracking has no ball (it is blank exactly from the ball going out,
  a foul or a goal until the restart), ignoring blanks shorter than ten frames;
- otherwise the **side that owns the ball**, which changes at the first on-ball event of the other
  side (a recovery, pass, carry, shot or set piece). Challenges, fouls and "ball lost" events
  never move ownership on their own: a loose ball stays with the side that last had it.

The corrected tracking is far cleaner than the automatic tracking the challenge provides, so
``degrade`` roughens it in the ways that matter: the ball disappears for long stretches, players
drop out, positions get noisy.
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import features as F
from .io import HOME, AWAY, PITCH

H, A, D = 0, 1, 2
OWNER_EVENTS = {"RECOVERY", "PASS", "CARRY", "SHOT", "SET PIECE"}
MIN_DEAD_BLANK = 10
# Bumped whenever ``degrade`` changes, so cached predictions made under an older noise model are
# never mistaken for current ones.
NOISE_VERSION = "n2"


def _labels(n, halves, ball_blank, events):
    """events: [(frame_index, team, type)] in file order."""
    owner = np.full(n, -1, dtype=np.int8)
    for first, last in halves:
        inside = [(f, team) for f, team, kind in events
                  if first <= f <= last and kind in OWNER_EVENTS]
        if not inside:
            continue
        frames = np.array([f for f, _ in inside])
        teams = np.array([t for _, t in inside], dtype=np.int8)
        order = np.argsort(frames, kind="stable")
        frames, teams = frames[order], teams[order]
        index = np.arange(first, last + 1)
        position = np.searchsorted(frames, index, side="right") - 1
        owner[first:last + 1] = teams[np.clip(position, 0, len(teams) - 1)]

    dead = np.zeros(n, dtype=bool)
    for first, last in halves:
        starts, stops = F.runs(ball_blank[first:last + 1])
        for start, stop in zip(starts, stops):
            if stop - start >= MIN_DEAD_BLANK:
                dead[first + start:first + stop] = True
    labels = np.where(dead, D, owner).astype(np.int8)
    graded = np.zeros(n, dtype=bool)
    for first, last in halves:
        graded[first:last + 1] = True
    labels[~graded] = 3
    return labels


def _match(px, py, bx, by, team, halves, events, name):
    n = len(bx)
    blank = np.isnan(bx)
    return {
        "name": name,
        "px": (px * PITCH[0]).astype(np.float32), "py": (py * PITCH[1]).astype(np.float32),
        "bx": (bx * PITCH[0]).astype(np.float32), "by": (by * PITCH[1]).astype(np.float32),
        "team": team, "n_frames": n, "halves": halves, "tags": [],
        "labels": _labels(n, halves, blank, events),
    }


def _load_csv_game(folder, game):
    stem = os.path.join(folder, "Sample_Game_{0}", "Sample_Game_{0}_").format(game)

    def tracking(side):
        with open(stem + "RawTrackingData_{}_Team.csv".format(side)) as handle:
            rows = list(csv.reader(handle))[3:]
        table = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in row]
                          for row in rows], dtype=np.float64)
        return table

    home, away = tracking("Home"), tracking("Away")
    period = home[:, 0].astype(int)
    hx, hy = home[:, 3:-2:2], home[:, 4:-2:2]
    ax, ay = away[:, 3:-2:2], away[:, 4:-2:2]
    px, py = np.hstack([hx, ax]), np.hstack([hy, ay])
    team = np.array([HOME] * hx.shape[1] + [AWAY] * ax.shape[1], dtype=np.int8)
    halves = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]))
              for p in sorted(set(period))]

    events = []
    with open(stem + "RawEventsData.csv") as handle:
        for row in csv.DictReader(handle):
            events.append((int(row["Start Frame"]) - 1,
                           HOME if row["Team"] == "Home" else AWAY, row["Type"]))
    return _match(px, py, home[:, -2], home[:, -1], team, halves, events,
                  "sample{}".format(game))


def _load_epts_game(folder):
    stem = os.path.join(folder, "Sample_Game_3", "Sample_Game_3_")
    root = ET.parse(stem + "metadata.xml").getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    players = [(player.get("id"), player.get("teamId")) for player in root.iter("Player")]
    slot = {player_id: i for i, (player_id, _) in enumerate(players)}
    team = np.array([HOME if team_id == team_ids[0] else AWAY for _, team_id in players],
                    dtype=np.int8)
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    parameters = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    halves = [(int(parameters["first_half_start"]) - 1, int(parameters["first_half_end"]) - 1),
              (int(parameters["second_half_start"]) - 1, int(parameters["second_half_end"]) - 1)]

    # Which players a line holds changes with every substitution.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        order = [slot[channel[ref]] for ref in refs[::2]]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), np.array(order)))

    n = halves[-1][1] + 1
    px = np.full((n, len(players)), np.nan)
    py = np.full((n, len(players)), np.nan)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    with open(stem + "tracking.txt") as handle:
        for line in handle:
            head, body, ball = line.strip().split(":")
            frame = int(head)
            if frame > n:
                continue
            order = next(o for a, b, o in specs if a <= frame <= b)
            values = np.array(body.replace(";", ",").split(","), dtype=np.float64)
            px[frame - 1, order] = values[0::2]
            py[frame - 1, order] = values[1::2]
            bx[frame - 1], by[frame - 1] = [float(v) for v in ball.split(",")[:2]]

    with open(stem + "events.json") as handle:
        raw = json.load(handle)["data"]
    events = [(int(e["start"]["frame"]) - 1,
               HOME if e["team"]["id"] == team_ids[0] else AWAY, e["type"]["name"])
              for e in raw]
    return _match(px, py, bx, by, team, halves, events, "sample3")


def load(folder, cache_dir=None):
    """The three sample matches as match dicts with a ``labels`` array (0 H, 1 A, 2 D, 3 X)."""
    cache = os.path.join(cache_dir, "external.npz") if cache_dir else None
    keys = ("px", "py", "bx", "by", "team", "labels")
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        out = []
        for i in range(3):
            match = {key: z["{}_{}".format(key, i)] for key in keys}
            match["halves"] = [tuple(int(v) for v in row) for row in z["halves_{}".format(i)]]
            match["n_frames"] = len(match["bx"])
            match["name"] = "sample{}".format(i + 1)
            match["tags"] = []
            out.append(match)
        return out

    out = [_load_csv_game(folder, 1), _load_csv_game(folder, 2), _load_epts_game(folder)]
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        blob = {}
        for i, match in enumerate(out):
            for key in keys:
                blob["{}_{}".format(key, i)] = match[key]
            blob["halves_{}".format(i)] = np.array(match["halves"])
        np.savez_compressed(cache, **blob)
    return out


def _offsets(rng, shape, sigma, scale=40):
    """Slowly varying position error of standard deviation ``sigma``: white noise low-passed
    over ``scale`` frames. Automatic tracking is smooth from frame to frame; its error drifts."""
    from scipy.ndimage import gaussian_filter1d
    out = gaussian_filter1d(rng.standard_normal(shape), scale, axis=0, mode="nearest")
    return out * (sigma / out.std())


def _smooth_track(values, sigma):
    """Gaussian smoothing over ``sigma`` frames that ignores missing observations."""
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0:
        return values.copy()
    seen = ~np.isnan(values)
    total = gaussian_filter1d(np.where(seen, values, 0.0), sigma, mode="nearest")
    weight = gaussian_filter1d(seen.astype(np.float64), sigma, mode="nearest")
    return np.where(seen, total / np.maximum(weight, 1e-9), np.nan)


def degrade(match, seed=0, ball_smooth=7.0, ball_noise=0.7, player_noise=0.3,
            alive_drop_share=0.19, player_drop_share=0.07, early_loss=40, late_return=75):
    """A copy of ``match`` roughened to resemble the automatic tracking.

    - the ball track is smoothed over ``ball_smooth`` frames, the way a tracker's filter rounds
      off the corners of a pass or a touch, and ball and player positions get slowly drifting
      offsets. These three numbers were set so that, on the challenge match and without any
      label, the rate of detectable ball touches, the distance from the ball to the nearest
      player at a touch, with a slow ball and with a fast ball, and the ball's acceleration all
      come out the same;
    - the ball is lost a little before most stoppages and sometimes found a little after the
      restart, and it also disappears for long stretches of open play (lengths drawn around six
      seconds, more often near the touchlines) until ``alive_drop_share`` of live frames are gone;
    - each player vanishes for stretches of a few seconds, ``player_drop_share`` of the time.
    """
    rng = np.random.RandomState(seed)
    out = dict(match)
    n = match["n_frames"]
    labels = match["labels"]
    bx = match["bx"].astype(np.float64).copy()
    by = match["by"].astype(np.float64).copy()
    hidden = np.isnan(bx)

    alive = (labels == H) | (labels == A)
    starts, stops = F.runs(labels == D)
    for start, stop in zip(starts, stops):
        if rng.rand() < 0.6:
            hidden[max(start - rng.randint(0, early_loss + 1), 0):start] = True
        if rng.rand() < 0.3:
            hidden[stop:min(stop + rng.randint(0, late_return + 1), n)] = True
    hidden &= alive | np.isnan(bx)

    target = alive_drop_share * alive.sum()
    near_line = np.minimum.reduce([bx, PITCH[0] - bx, by, PITCH[1] - by]) < 8.0
    candidates = np.flatnonzero(alive)
    weights = np.where(near_line[candidates], 3.0, 1.0)
    weights = weights / weights.sum()
    for _ in range(5000):
        if (hidden & alive).sum() >= target:
            break
        start = int(rng.choice(candidates, p=weights))
        length = int(np.clip(rng.lognormal(np.log(150.0), 0.8), 50, 1500))
        hidden[start:start + length] = True
    hidden &= alive | np.isnan(bx)

    bx = _smooth_track(bx, ball_smooth) + _offsets(rng, (n,), ball_noise)
    by = _smooth_track(by, ball_smooth) + _offsets(rng, (n,), ball_noise)
    bx[hidden] = np.nan
    by[hidden] = np.nan

    px = match["px"].astype(np.float64).copy()
    py = match["py"].astype(np.float64).copy()
    px += _offsets(rng, px.shape, player_noise)
    py += _offsets(rng, py.shape, player_noise)
    present = ~np.isnan(px)
    for column in range(px.shape[1]):
        on_pitch = np.flatnonzero(present[:, column])
        if not len(on_pitch):
            continue
        budget = player_drop_share * len(on_pitch)
        gone = 0
        while gone < budget:
            start = int(rng.choice(on_pitch))
            length = int(rng.exponential(125.0)) + 10
            px[start:start + length, column] = np.nan
            py[start:start + length, column] = np.nan
            gone += length
    out.update(px=px.astype(np.float32), py=py.astype(np.float32),
               bx=bx.astype(np.float32), by=by.astype(np.float32))
    return out
