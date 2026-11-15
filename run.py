"""Produce the possession submission from the provided match data.

    python run.py --data path/to/data --samples path/to/sample-data/data --out out/<team_name>.csv

``--data`` is the folder that contains ``tactical/`` (the challenge bundle unpacked as described
in the challenge repository's DATA.md). Only the tactical tracking (ATD) and the Smart Tagging are
read; the videos, the homography and the broadcast recording are not needed.

``--samples`` is the ``data`` folder of https://github.com/metrica-sports/sample-data, the three
public matches the learned models are trained on.

``--method frame`` leaves the sequence model out (the previous submission, about ten minutes);
``--method sequence`` uses the sequence model alone; ``--method self`` runs the earliest method,
which needs no other match.
"""

import argparse
import os
import time

import numpy as np

from possession import diagnostics, io, model, transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", "data"),
                        help="folder containing tactical/ (default: $MTAH_DATA_DIR or ./data)")
    parser.add_argument("--samples", default=os.environ.get("MTAH_SAMPLES_DIR"),
                        help="data folder of metrica-sports/sample-data (default: "
                             "$MTAH_SAMPLES_DIR); required unless --method self")
    parser.add_argument("--out", default=os.path.join("out", "submission.csv"),
                        help="where to write the frame,label CSV")
    parser.add_argument("--cache", default="cache",
                        help="folder for the parsed-tracking caches ('' disables them)")
    parser.add_argument("--method", choices=("transfer", "frame", "sequence", "self"),
                        default="transfer",
                        help="transfer: frame models and sequence model trained on other matches "
                             "(the submission). frame: without the sequence model. sequence: the "
                             "sequence model alone. self: nearest-player rule plus this match's "
                             "own pseudo-labels")
    parser.add_argument("--evidence",
                        help="also save the per-frame model evidence here (.npz), for replay.py")
    parser.add_argument("--diagnostics", action="store_true",
                        help="also print the label-free checks described in the technical document")
    args = parser.parse_args()
    if args.method != "self" and not args.samples:
        parser.error("--samples is required unless --method self")

    started = time.time()

    def log(message):
        print("[{:5.0f}s] {}".format(time.time() - started, message), flush=True)

    cache = args.cache or None
    match = io.load_target(args.data, cache_dir=cache)
    log("tracking loaded: H = {}, A = {}, {} Smart Tagging instances".format(
        match["names"][0], match["names"][1], len(match["tags"])))

    if args.method != "self":
        params = {"transfer": model.PARAMS, "frame": model.FRAME_ONLY,
                  "sequence": model.SEQUENCE}[args.method]
        sources = transfer.from_samples(args.samples, match, params, cache_dir=cache, log=log)
        log("learned evidence ready for the challenge match")
        result = model.run(match, params, log=log, transferred=sources, self_train=False)
    else:
        params = model.SELF_TRAINED
        result = model.run(match, params, log=log)

    labels = model.LABELS[result["labels"]]
    path = io.write_submission(args.out, labels)
    graded = model.graded_mask(match)
    counts = {name: int(np.sum(labels[graded] == name)) for name in "HAD"}
    log("wrote {}  (graded frames: H {H}  A {A}  D {D})".format(path, **counts))
    if args.evidence:
        folder = os.path.dirname(os.path.abspath(args.evidence))
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(args.evidence, **{key: np.asarray(value, dtype=np.float32)
                                              for key, value in result["sources"].items()})
        log("saved the model evidence to {}".format(args.evidence))
    if args.diagnostics:
        diagnostics.report(match, result, params, log=log, masked=args.method == "self")


if __name__ == "__main__":
    main()
