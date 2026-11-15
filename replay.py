"""Replay a match in the browser: the players, the ball, our possession call and the truth.

The challenge match (its truth covers the first five minutes only):

    python run.py --data /path/to/data --samples sample-data/data --evidence out/evidence.npz
    python replay.py --data /path/to/data --labels out/submission.csv \
        --truth /path/to/more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv \
        --evidence out/evidence.npz --open

One of the three public sample matches, with the truth over the whole match. Its predictions come
from benchmark.py's cache, so run the benchmark first:

    python replay.py --sample 1 --samples sample-data/data --open

Each replay is written to out/replay/<name>/ and --open serves out/replay on
http://127.0.0.1:8765 and opens it in the browser. ``--serve`` serves what is already there.

Everything stays on this computer. The challenge data is licensed for the competition only:
nothing this script writes for it may be uploaded, published or shared.
"""

import argparse
import functools
import html
import http.server
import json
import os
import shutil
import webbrowser

import numpy as np

from possession import external, io, model

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "viewer", "index.html")
MISSING = -32768   # positions.bin: no observation
NONE = 3           # series.bin labels: no label on this frame
NO_VALUE = 255     # series.bin probabilities: no value on this frame
# The tracking metadata spells the away side "Atletic Club".
DISPLAY_NAMES = {"Atletic Club": "Athletic Club"}


def read_labels(path, n_frames):
    """Label codes (0 home, 1 away, 2 dead, 3 none) from a frame,label CSV."""
    codes = np.full(n_frames, NONE, dtype=np.uint8)
    lookup = {str(name): code for code, name in enumerate(model.LABELS)}
    with open(path) as handle:
        header = [part.strip().lower() for part in handle.readline().split(",")]
        if header != ["frame", "label"]:
            raise ValueError("{}: expected a frame,label header".format(path))
        for line in handle:
            frame, label = line.split(",")
            frame = int(frame)
            if 0 <= frame < n_frames:
                codes[frame] = lookup[label.strip().upper()]
    return codes


def _probabilities(values, n_frames, where=None):
    """0..250 for a probability, NO_VALUE where there is none."""
    out = np.full(n_frames, NO_VALUE, dtype=np.uint8)
    if values is None:
        return out
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values) if where is None else np.isfinite(values) & where
    out[keep] = np.round(np.clip(values[keep], 0.0, 1.0) * 250).astype(np.uint8)
    return out


def _decimetres(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, MISSING, dtype=np.int16)
    seen = np.isfinite(values)
    out[seen] = np.clip(np.round(values[seen] * 10), -32000, 32000).astype(np.int16)
    return out


def chances(match, evidence, params=model.PARAMS):
    """The models' own probabilities, for the strip under the pitch: the home side has the ball
    (the ownership models averaged with the weights the decode gives them, each where it applies)
    and the ball is dead (the dead/alive model). The decode itself uses tempered versions of
    these, so a strip read as "chance" would otherwise look far less certain than the models
    are."""
    n = match["n_frames"]
    seen = np.isfinite(np.asarray(match["bx"], dtype=np.float64))
    everywhere = np.ones(n, dtype=bool)
    parts = [("refined_transfer", params["w_refined_transfer"], everywhere),
             ("all_transfer", params["w_all_transfer"], everywhere),
             ("owner_transfer", params["w_owner_transfer"], seen),
             ("team_transfer", params["w_team_transfer_hidden"], ~seen)]
    total = np.zeros(n)
    weight = np.zeros(n)
    for key, w, where in parts:
        if w and key in evidence:
            total += np.where(where, w * np.asarray(evidence[key], dtype=np.float64), 0.0)
            weight += np.where(where, w, 0.0)
    home = np.where(weight > 0, total / np.maximum(weight, 1e-9), np.nan)
    dead = evidence.get("dead_transfer")
    return {"home": home, "dead": None if dead is None else np.asarray(dead, dtype=np.float64)}


def export(folder, match, title, subtitle, teams, predicted=None, truth=None, chance=None,
           kickoffs=None):
    """Write one replay: index.html (the viewer), meta.json, positions.bin and series.bin.

    positions.bin  int16 little-endian, one row per frame: x, y of every track slot, then the
                   ball, in decimetres; MISSING where there is no observation
    series.bin     uint8, one block of n_frames per series in meta["series"]["names"]
    """
    os.makedirs(folder, exist_ok=True)
    n = match["n_frames"]
    slots = match["px"].shape[1]

    table = np.empty((n, 2 * slots + 2), dtype="<i2")
    table[:, 0:2 * slots:2] = _decimetres(match["px"])
    table[:, 1:2 * slots:2] = _decimetres(match["py"])
    table[:, -2] = _decimetres(match["bx"])
    table[:, -1] = _decimetres(match["by"])
    table.tofile(os.path.join(folder, "positions.bin"))

    none = np.full(n, NONE, dtype=np.uint8)
    chance = chance or {}
    series = [
        ("pred", none if predicted is None else np.asarray(predicted, dtype=np.uint8)),
        ("truth", none if truth is None else np.asarray(truth, dtype=np.uint8)),
        ("p_home", _probabilities(chance.get("home"), n)),
        ("p_dead", _probabilities(chance.get("dead"), n)),
    ]
    np.concatenate([values for _, values in series]).tofile(os.path.join(folder, "series.bin"))

    halves = [[int(first), int(last)] for first, last in match["halves"]]
    meta = {
        "title": title,
        "subtitle": subtitle,
        "teams": list(teams),
        "fps": 25,
        "n_frames": int(n),
        "pitch": [float(v) for v in io.PITCH],
        "halves": halves,
        "kickoffs": [int(k) for k in (kickoffs or [first for first, _ in halves])],
        "slot_team": [int(t) for t in match["team"]],
        "positions": {"file": "positions.bin", "columns": 2 * slots + 2, "scale": 10,
                      "missing": MISSING},
        "series": {"file": "series.bin", "names": [name for name, _ in series]},
    }
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(meta, handle)
    shutil.copyfile(VIEWER, os.path.join(folder, "index.html"))
    write_index(os.path.dirname(os.path.abspath(folder)))
    return folder


def write_index(root):
    """A page at the root of the replay folder that links to every replay in it."""
    items = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "meta.json")
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            meta = json.load(handle)
        items.append('<li><a href="{0}/">{1}</a><span>{2}</span></li>'.format(
            html.escape(name), html.escape(meta["title"]), html.escape(meta["subtitle"])))
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Possession replays</title>
<style>
:root { color-scheme: light dark; }
body { margin: 0; padding: 32px 16px; background: Canvas; color: CanvasText;
       font-family: "Barlow Semi Condensed", system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 640px; margin: 0 auto; }
h1 { font-size: 26px; font-weight: 600; margin: 0 0 16px; }
ul { list-style: none; padding: 0; margin: 0; }
li { padding: 10px 0; border-top: 1px solid color-mix(in srgb, CanvasText 12%, transparent); }
a { font-size: 19px; font-weight: 600; display: block; }
span { opacity: 0.75; font-size: 15px; }
</style></head>
<body><main><h1>Possession replays</h1><ul>
""" + "\n".join(items) + "\n</ul></main></body></html>\n"
    with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(page)


def challenge(args, folder):
    match = io.load_target(args.data, cache_dir=args.cache or None)
    teams = [DISPLAY_NAMES.get(name, name) for name in match["names"]]
    predicted = None
    if args.labels and os.path.isfile(args.labels):
        predicted = read_labels(args.labels, match["n_frames"])
    else:
        print("no labels at {}: showing the tracking only".format(args.labels))
    truth = read_labels(args.truth, match["n_frames"]) if args.truth else None
    evidence = None
    if args.evidence:
        with np.load(args.evidence) as z:
            evidence = {key: z[key] for key in z.files}
    starts = sorted(frame for tag, frame in model.kickoff_frames(match)
                    if "start" in tag["type"])
    if len(starts) != len(match["halves"]):
        starts = [first for first, _ in match["halves"]]
    subtitle = "Challenge match, automatic tracking"
    if truth is not None:
        subtitle += ". The truth covers the public window only"
    params = {"transfer": model.PARAMS, "frame": model.FRAME_ONLY,
              "sequence": model.SEQUENCE}[args.method]
    chance = chances(match, evidence, params) if evidence else None
    return export(folder, match, "{} v {}".format(*teams), subtitle, teams, predicted, truth,
                  chance, starts)


def sample(args, folder):
    """A public sample match as the benchmark saw it: roughened tracking, held-out predictions."""
    index = args.sample - 1
    games = external.load(args.samples, cache_dir=args.cache or None)
    match = external.degrade(games[index], seed=index)   # the realisation benchmark.py scores
    params = {"transfer": model.PARAMS, "frame": model.FRAME_ONLY,
              "sequence": model.SEQUENCE}[args.method]
    stem = os.path.join(args.cache, "benchmark_{}_seed0_i{}_l{}_a{}_w{}".format(
        external.NOISE_VERSION, params["boost_iters"], params["boost_leaves"],
        params["augment"], params["spell_weight"]))
    # The sequence model's held-out predictions are cached by device; the frame models' are not.
    found = [path for path in (stem + "_cpu.npz", stem + "_cuda.npz", stem + ".npz")
             if os.path.isfile(path)]
    predicted = chance = None
    if found:
        with np.load(found[0]) as z:
            evidence = {key: z["{}_{}".format(key, index)]
                        for key in ("dead_transfer", "team_transfer", "owner_transfer",
                                    "all_transfer", "refined_transfer")
                        if "{}_{}".format(key, index) in z.files}
        if params["w_refined_transfer"] and "refined_transfer" not in evidence:
            print("the cached predictions have no sequence model: showing --method frame")
            params = model.FRAME_ONLY
        scores = model.frame_scores(model.ball_evidence(match, params), evidence, params)
        predicted = model.decode(match, scores, params)
        chance = chances(match, evidence, params)
    else:
        print("no benchmark predictions under {}: run benchmark.py first. Showing the truth only."
              .format(stem))
    return export(folder, match, "Sample match {}".format(args.sample),
                  "Metrica Sports public sample data, tracking roughened as in the benchmark. "
                  "The truth comes from the published events",
                  ["Home", "Away"], predicted, match["labels"], chance)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(root, name, port, open_browser=True):
    handler = functools.partial(_QuietHandler, directory=root)
    for candidate in range(port, port + 20):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("no free port between {} and {}".format(port, port + 19))
    url = "http://127.0.0.1:{}/{}".format(candidate, name + "/" if name else "")
    print("replay at {}   (Ctrl+C to stop)".format(url), flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"),
                        help="folder containing tactical/ (challenge match)")
    parser.add_argument("--labels", default=os.path.join("out", "submission.csv"),
                        help="our frame,label CSV for the challenge match")
    parser.add_argument("--truth", help="a frame,label CSV of true labels, e.g. the public window")
    parser.add_argument("--evidence", help="per-frame model evidence saved by run.py --evidence")
    parser.add_argument("--method", choices=("transfer", "frame", "sequence"), default="transfer",
                        help="the run.py method the labels and evidence come from, so the strip "
                             "of chances weighs the evidence the way that method did")
    parser.add_argument("--sample", type=int, choices=(1, 2, 3),
                        help="replay a public sample match instead of the challenge match")
    parser.add_argument("--samples", default=os.environ.get("MTAH_SAMPLES_DIR"),
                        help="data folder of metrica-sports/sample-data (with --sample)")
    parser.add_argument("--out", default=os.path.join("out", "replay"),
                        help="folder that holds the replays")
    parser.add_argument("--name", help="sub-folder for this replay (default: challenge or sampleN)")
    parser.add_argument("--cache", default="cache", help="folder of the parsed-tracking caches")
    parser.add_argument("--open", action="store_true", help="serve the replays and open this one")
    parser.add_argument("--serve", action="store_true",
                        help="only serve what is already in --out, and open it")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    root = os.path.abspath(args.out)
    if args.serve:
        os.makedirs(root, exist_ok=True)
        write_index(root)
        serve(root, args.name or "", args.port)
        return

    name = args.name or ("sample{}".format(args.sample) if args.sample else "challenge")
    folder = os.path.join(root, name)
    if args.sample:
        if not args.samples:
            parser.error("--samples is required with --sample")
        sample(args, folder)
    else:
        challenge(args, folder)
    print("wrote {}".format(folder))
    if args.open:
        serve(root, name, args.port)


if __name__ == "__main__":
    main()
