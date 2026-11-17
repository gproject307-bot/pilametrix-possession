"""Compare methods on other matches where the answer is known.

    python benchmark.py --samples /path/to/sample-data/data \
                        --scorer /path/to/more-than-a-hack-2027/evaluation

``--samples`` is the ``data`` folder of https://github.com/metrica-sports/sample-data (three
public matches with corrected tracking and events). ``--scorer`` is the challenge repository's
``evaluation`` folder, so every number below comes from the official metric code.

Each sample match is roughened to resemble the automatic tracking (possession/external.py), then
decoded by every method and scored against ground truth derived from its events. Methods that
learn from labels are trained on the *other two* matches, never on the one being scored. The
sequence model sits on top of the frame models, so the nesting goes one level deeper: the traces
it learns from on each training match come from frame models fitted on the *one* remaining match.

``--tune`` is how the fusion weights and switch costs in possession/model.py were chosen.
"""

import argparse
import os
import sys
import time

import numpy as np

from possession import external, model, refine, transfer

H, A, D = model.H, model.A, model.D
SOURCES = ("dead_transfer", "team_transfer", "owner_transfer", "all_transfer",
           "refined_transfer")


def baseline(match, window):
    """The challenge's reference method: nearest tracked player to the ball, then a majority vote
    over +/- ``window`` frames; dead where no ball was seen in the window."""
    evidence = model.ball_evidence(match)
    visible = evidence["visible"]
    home = visible & (evidence["d_home"] <= evidence["d_away"])
    away = visible & ~home
    index = np.arange(match["n_frames"])
    low = np.maximum(0, index - window)
    high = np.minimum(match["n_frames"], index + window + 1)
    hc = np.concatenate(([0], np.cumsum(home)))
    ac = np.concatenate(([0], np.cumsum(away)))
    nh, na = hc[high] - hc[low], ac[high] - ac[low]
    out = np.full(match["n_frames"], D, dtype=np.int8)
    out[(nh + na > 0) & (nh >= na)] = H
    out[(nh + na > 0) & (nh < na)] = A
    out[~model.graded_mask(match)] = 3
    return out


def auc(score, positive):
    """Area under the ROC curve by ranks (ties averaged)."""
    from scipy.stats import rankdata
    ranks = rankdata(score)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prepare(args, log):
    """Degraded matches, their evidence, and held-out predictions of the transferred models."""
    params = dict(model.PARAMS, boost_iters=args.iters, boost_leaves=args.leaves,
                  spell_weight=args.spell_weight, refine_device=args.device)
    if params.get("ball_sensor") or params.get("facing"):
        return prepare_sensor(args, params, log)
    games = external.load(args.samples, cache_dir=args.cache)
    matches = [external.degrade(game, seed=args.seed + i) for i, game in enumerate(games)]
    log("loaded and degraded {} matches".format(len(matches)))

    stem = os.path.join(args.cache, "benchmark_{}_seed{}_i{}_l{}_a{}_w{}".format(
        external.NOISE_VERSION, args.seed, args.iters, args.leaves, args.augment,
        args.spell_weight))
    frame_cache = stem + ".npz"                       # the frame models' held-out predictions
    cache = "{}_{}.npz".format(stem, args.device)     # ... plus the sequence model's
    evidences = [model.ball_evidence(match) for match in matches]
    if os.path.isfile(cache) and not args.refit:
        z = np.load(cache)
        sources = [{key: z["{}_{}".format(key, i)] for key in SOURCES}
                   for i in range(len(matches))]
        log("held-out predictions read from cache")
        return matches, evidences, sources

    built = []
    rows = []
    for i, match in enumerate(matches):
        b = transfer.design(match)
        built.append(b)
        rows.append([transfer.training_rows(match, b, params)])
        # Extra noise realisations of the same match, for training only.
        for k in range(1, args.augment):
            again = external.degrade(games[i], seed=args.seed + i + 100 * k)
            rows[-1].append(transfer.training_rows(again, transfer.design(again), params))
        log("features ready for {}".format(match["name"]))
    if os.path.isfile(frame_cache) and not args.refit:
        z = np.load(frame_cache)
        sources = [{key: z["{}_{}".format(key, i)] for key in SOURCES[:4]}
                   for i in range(len(matches))]
        log("frame models' held-out predictions read from cache")
    else:
        sources = []
        for i, match in enumerate(matches):
            models = transfer.train([r for j in range(len(matches)) if j != i for r in rows[j]],
                                    params)
            sources.append(transfer.apply(models, built[i]))
            log("models trained without {} and applied to it".format(match["name"]))
        np.savez_compressed(frame_cache, **{"{}_{}".format(key, i): value
                                            for i, source in enumerate(sources)
                                            for key, value in source.items()})

    # Traces for the sequence model: match j as seen by frame models fitted on match k alone.
    alone = {}
    for k, match in enumerate(matches):
        models = transfer.train(rows[k], params)
        for j in range(len(matches)):
            if j != k:
                alone[(j, k)] = transfer.apply(models, built[j])
        log("models trained on {} alone".format(match["name"]))
    for i, match in enumerate(matches):
        j, k = [x for x in range(len(matches)) if x != i]
        sequence_list = (refine.sequences(matches[j], alone[(j, k)], evidences[j], params)
                         + refine.sequences(matches[k], alone[(k, j)], evidences[k], params))
        nets = refine.fit(sequence_list, params)
        sources[i]["refined_transfer"] = refine.apply(nets, match, sources[i], evidences[i],
                                                      params)
        log("sequence model trained without {} and applied to it".format(match["name"]))
    blob = {"{}_{}".format(key, i): value
            for i, source in enumerate(sources) for key, value in source.items()}
    np.savez_compressed(cache, **blob)
    return matches, evidences, sources


def prepare_sensor(args, params, log):
    """``prepare`` with the ball sensor and facing. Nothing ever sees the held-out match: for
    held-out match i, with j and k the other two, the ball sensor for i is fitted on j and k, for
    training match j on k alone and for k on j alone, and the sequence model's training traces
    (j scored by frame models fitted on k alone, and k by models fitted on j) use those too."""
    from possession import sensor
    games = external.load(args.samples, cache_dir=args.cache)
    copies = [[external.degrade(game, seed=args.seed + i + 100 * k) for k in range(args.augment)]
              for i, game in enumerate(games)]
    matches = [group[0] for group in copies]
    evidences = [model.ball_evidence(match) for match in matches]
    log("loaded and degraded {} matches, {} copies each".format(len(matches), args.augment))
    n = len(matches)
    stem = os.path.join(args.cache, "benchmark_{}_seed{}_i{}_l{}_a{}_w{}_bs{}_f{}".format(
        external.NOISE_VERSION, args.seed, args.iters, args.leaves, args.augment,
        args.spell_weight, int(bool(params.get("ball_sensor"))), int(bool(params.get("facing")))))
    frame_cache = stem + ".npz"
    cache = "{}_r{}_{}.npz".format(stem, params.get("refine_reach"), args.device)
    if os.path.isfile(cache) and not args.refit:
        z = np.load(cache)
        log("held-out predictions read from cache")
        return matches, evidences, [{key: z["{}_{}".format(key, i)] for key in SOURCES}
                                    for i in range(n)]

    keys = ("held", "alone_j", "alone_k")
    if os.path.isfile(frame_cache) and not args.refit:
        z = np.load(frame_cache)
        frame = {(i, part): {key: z["{}_{}_{}".format(part, key, i)] for key in SOURCES[:4]}
                 for i in range(n) for part in keys}
        log("frame models' held-out predictions read from cache")
    else:
        use_sensor = bool(params.get("ball_sensor"))
        if use_sensor:
            reg_rows = [[sensor.regressor_rows(c, games[i]["bx"], games[i]["by"],
                                               model.graded_mask(c)) for c in copies[i]]
                        for i in range(n)]
            single = {k: sensor.fit_ball(reg_rows[k]) for k in range(n)}
            pair = {i: sensor.fit_ball([r for j in range(n) if j != i for r in reg_rows[j]])
                    for i in range(n)}
            log("ball sensors fitted")

        def ball(regressor, match):
            return sensor.predict_ball(regressor, match) if use_sensor else None

        # Feature matrices without the pseudo-ball (heavy, once per copy); the pseudo-ball
        # columns are cheap and are added per fold.
        base = [[transfer.design(c, params) for c in copies[i]] for i in range(n)]
        log("features ready")
        frame = {}
        for i in range(n):
            j, k = [x for x in range(n) if x != i]

            def rows_of(m, by):
                return [transfer.training_rows(c, transfer.with_ball(b, c, ball(single[by], c)
                                                                     if use_sensor else None),
                                               params)
                        for c, b in zip(copies[m], base[m])]

            rows_j, rows_k = rows_of(j, k), rows_of(k, j)
            target = transfer.with_ball(base[i][0], matches[i],
                                        ball(pair[i], matches[i]) if use_sensor else None)
            frame[(i, "held")] = transfer.apply(transfer.train(rows_j + rows_k, params), target)
            built_j = transfer.with_ball(base[j][0], matches[j],
                                         ball(single[k], matches[j]) if use_sensor else None)
            built_k = transfer.with_ball(base[k][0], matches[k],
                                         ball(single[j], matches[k]) if use_sensor else None)
            frame[(i, "alone_j")] = transfer.apply(transfer.train(rows_k, params), built_j)
            frame[(i, "alone_k")] = transfer.apply(transfer.train(rows_j, params), built_k)
            log("frame models done without {}".format(matches[i]["name"]))
            del rows_j, rows_k, target, built_j, built_k
        np.savez_compressed(frame_cache, **{"{}_{}_{}".format(part, key, i): value
                                            for (i, part), source in frame.items()
                                            for key, value in source.items()})

    sources = []
    for i in range(n):
        j, k = [x for x in range(n) if x != i]
        sequence_list = (refine.sequences(matches[j], frame[(i, "alone_j")], evidences[j], params)
                         + refine.sequences(matches[k], frame[(i, "alone_k")], evidences[k],
                                            params))
        nets = refine.fit(sequence_list, params)
        source = dict(frame[(i, "held")])
        source["refined_transfer"] = refine.apply(nets, matches[i], source, evidences[i], params)
        sources.append(source)
        log("sequence model trained without {} and applied to it".format(matches[i]["name"]))
    np.savez_compressed(cache, **{"{}_{}".format(key, i): value
                                  for i, source in enumerate(sources)
                                  for key, value in source.items()})
    return matches, evidences, sources


def components(matches, evidences, sources, log):
    """How good is each piece of evidence on its own, on the match it never saw?"""
    for match, evidence, source in zip(matches, evidences, sources):
        labels = match["labels"]
        graded = model.graded_mask(match)
        alive = graded & (labels != D)
        visible = evidence["visible"]
        seen, hidden = alive & visible, alive & ~visible
        is_home = labels == H
        rule = evidence["d_home"] < evidence["d_away"]

        def right(key, rows):
            return float(np.mean((source[key][rows] > 0.5) == is_home[rows]))

        log("{}: dead/alive AUC from players {:.3f} | who has it, ball seen: nearest player "
            "{:.3f}, ball model {:.3f}, player model {:.3f}, combined {:.3f} | ball not seen: "
            "player model {:.3f}, combined {:.3f} | ball and player models agree {:.3f}".format(
                match["name"], auc(source["dead_transfer"][graded], labels[graded] == D),
                float(np.mean(rule[seen] == is_home[seen])),
                right("owner_transfer", seen), right("team_transfer", seen),
                right("all_transfer", seen), right("team_transfer", hidden),
                right("all_transfer", hidden),
                float(np.mean((source["owner_transfer"][seen] > 0.5)
                              == (source["team_transfer"][seen] > 0.5)))))
        log("    sequence model: ball seen {:.3f}, ball not seen {:.3f}".format(
            right("refined_transfer", seen), right("refined_transfer", hidden)))


def score(pm, matches, predictions):
    """Mean and per-match (macroF1, spellF1, predicted spells, true spells), official metric."""
    rows = []
    for match, predicted in zip(matches, predictions):
        result = pm.evaluate(match["labels"], predicted)
        rows.append((result["macro_f1"], result["spell_f1_mean"], result["n_pred_spells"],
                     result["n_gt_spells"]))
    rows = np.array(rows, dtype=np.float64)
    return rows.mean(axis=0), rows


def decode_all(matches, evidences, sources, params):
    return [model.decode(match, model.frame_scores(evidence, source, params), params)
            for match, evidence, source in zip(matches, evidences, sources)]


# Every method below gets the transferred dead/alive model for the frames without a ball, because
# nothing else can see a stoppage there. They differ in how they decide who has the ball.
RULE = dict(model.FRAME_ONLY, w_rule_team=1.0, w_owner_transfer=0.0, w_all_transfer=0.0,
            w_dead_transfer=0.25, w_missing=0.0, switch_team=16.0, switch_dead=12.0)
METHODS = [
    ("nearest-player rule", RULE),
    ("rule + player model on ball-less frames", dict(RULE, w_team_transfer_hidden=0.25)),
    ("ball model + player model on ball-less", dict(RULE, w_rule_team=0.0, w_owner_transfer=0.25,
                                                     w_team_transfer_hidden=0.25)),
    ("combined ball-and-player model", dict(RULE, w_rule_team=0.0, w_all_transfer=0.5)),
    ("ball + combined model, no sequence model", model.FRAME_ONLY),
    ("sequence model alone", model.SEQUENCE),
    ("sequence + ball + combined model (submission)", model.PARAMS),
]

# Coordinate ascent over the fusion weights and switch costs, on a coarse grid.
TUNE_GRID = {
    "switch_team": (3, 4, 6, 8, 12, 16),
    "switch_dead": (9, 12, 16),
    "w_refined_transfer": (0.0, 0.5, 1.0, 1.5),
    "w_all_transfer": (0.0, 0.25, 0.5, 1.0),
    "w_owner_transfer": (0.0, 0.25, 0.5, 1.0),
    "w_rule_team": (0.0, 0.5, 1.0),
    "w_team_transfer_hidden": (0.0, 0.25, 0.5),
    "w_team_transfer_visible": (0.0, 0.05, 0.1, 0.2),
    "w_dead_transfer": (0.15, 0.25, 0.5),
    "w_missing": (0.0, 0.03, 0.06, 0.1),
    "clip_logit": (2.0, 3.0, 4.0, 6.0),
    "min_dead": (25, 50),
}
TUNE_START = dict(model.PARAMS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--seed", type=int, default=0, help="seed of the simulated tracking noise")
    parser.add_argument("--refit", action="store_true", help="ignore cached model predictions")
    parser.add_argument("--iters", type=int, default=model.PARAMS["boost_iters"])
    parser.add_argument("--leaves", type=int, default=model.PARAMS["boost_leaves"])
    parser.add_argument("--augment", type=int, default=model.PARAMS["augment"],
                        help="noise realisations of each training match")
    parser.add_argument("--spell-weight", type=float, default=model.PARAMS["spell_weight"],
                        help="0: frames weigh the same in training; 1: possessions do")
    parser.add_argument("--device", default=model.PARAMS["refine_device"],
                        help="cpu or cuda, for the sequence model")
    parser.add_argument("--sweep", action="store_true",
                        help="also vary the final method's parameters one at a time")
    parser.add_argument("--tune", action="store_true",
                        help="coordinate ascent over the fusion weights and switch costs")
    args = parser.parse_args()

    sys.path.insert(0, args.scorer)
    import possession_metrics as pm

    started = time.time()

    def log(message):
        print("[{:5.0f}s] {}".format(time.time() - started, message), flush=True)

    matches, evidences, sources = prepare(args, log)
    components(matches, evidences, sources, log)

    def show(name, predictions):
        mean, rows = score(pm, matches, predictions)
        print("{:<46} macroF1 {:.3f}  spellF1 {:.3f}   per match {}   possessions {} vs {}".format(
            name, mean[0], mean[1], " ".join("{:.3f}".format(r[1]) for r in rows),
            int(rows[:, 2].sum()), int(rows[:, 3].sum())), flush=True)
        return mean[1]

    def value(params):
        if 2 * params["switch_dead"] <= params["switch_team"]:
            return -1.0
        mean, _ = score(pm, matches, decode_all(matches, evidences, sources, params))
        return float(mean[1])

    print()
    for window in (25, 50, 125):
        show("baseline, vote over +/-{} frames".format(window),
             [baseline(match, window) for match in matches])
    for name, params in METHODS:
        show(name, decode_all(matches, evidences, sources, params))

    if args.tune:
        best = dict(TUNE_START)
        best_value = value(best)
        print()
        print("tuning from spellF1 {:.4f}".format(best_value), flush=True)
        for sweep in range(2):
            for key, values in TUNE_GRID.items():
                for candidate in values:
                    if candidate == best[key]:
                        continue
                    trial = dict(best, **{key: candidate})
                    trial_value = value(trial)
                    if trial_value > best_value + 0.002:   # ignore gains inside the noise
                        best, best_value = trial, trial_value
                print("  pass {} after {:<24} spellF1 {:.4f}".format(sweep + 1, key, best_value),
                      flush=True)
        print("tuned: {}".format({key: best[key] for key in TUNE_GRID}))
        show("tuned", decode_all(matches, evidences, sources, best))

    if args.sweep:
        print()
        print("the final method with one parameter changed at a time")
        for key, values in TUNE_GRID.items():
            for candidate in values:
                trial = dict(model.PARAMS, **{key: candidate})
                if 2 * trial["switch_dead"] <= trial["switch_team"]:
                    continue
                show("  {} = {}".format(key, candidate),
                     decode_all(matches, evidences, sources, trial))


if __name__ == "__main__":
    main()
