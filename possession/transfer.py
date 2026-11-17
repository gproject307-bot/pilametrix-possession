"""Classifiers trained on real labels from *other* matches, then applied to the match being decoded.

Four frame models, all trained on the roughened copies of the public sample matches
(external.py), and a sequence model on top of them (refine.py):

  dead    P(dead ball)              from the players' collective behaviour
  team    P(home in possession)     from the players' collective behaviour, on live frames
  owner   P(home in possession)     from the ball's neighbourhood, on live frames with a ball
  all     P(home in possession)     from both together, on every live frame

None of them sees Smart Tagging (the other matches have none) and none of them sees whether the
ball is tracked when judging dead/alive: in the sample data a missing ball *is* a dead ball, in
the challenge data it is not, and a model allowed to learn that shortcut would not transfer.
The two possession models are trained on both sides' views, so they cannot favour a side.
"""

import os

import numpy as np

from . import model

H, A, D = model.H, model.A, model.D


def design(match, params=model.PARAMS, ball=None):
    """Every feature matrix the transferred models need for one match.

    ``ball`` is the ball sensor's prediction for this match (sensor.predict_ball), from a
    regressor that never saw this match; with it the pseudo-ball features join the player
    features. ``params["facing"]`` adds the facing features to the ball-neighbourhood ones.
    """
    from . import sensor
    evidence = model.ball_evidence(match, params)
    Xp, Xp_twin, player_names = model.player_design(match, with_tags=False)
    Xo, Xo_twin, owner_names = model.owner_design(match, evidence)
    if params.get("facing"):
        extra, extra_twin, names = sensor.facing_features(match, evidence)
        Xo = np.hstack([Xo, extra])
        Xo_twin = np.hstack([Xo_twin, extra_twin])
        owner_names = owner_names + names
    # "Xp_dead" is what the dead/alive model reads: the player features WITHOUT the pseudo-ball
    # columns. Those are blank exactly where the ball is tracked, so they would tell the model
    # whether a ball is tracked, and in the sample matches a tracked ball is never a dead ball,
    # which is not true of the challenge match (see the module docstring).
    built = {"evidence": evidence, "Xp": Xp, "Xp_twin": Xp_twin, "Xo": Xo, "Xo_twin": Xo_twin,
             "Xp_dead": Xp, "player_names": player_names, "owner_names": owner_names}
    return with_ball(built, match, ball)


def with_ball(built, match, ball):
    """``built`` with the pseudo-ball features of the ball sensor's prediction ``ball`` appended
    to the player features (unchanged when ``ball`` is None)."""
    from . import sensor
    if ball is None:
        return built
    extra, extra_twin, names = sensor.pseudo_ball_features(match, built["evidence"], ball)
    return dict(built, Xp=np.hstack([built["Xp"], extra]),
                Xp_twin=np.hstack([built["Xp_twin"], extra_twin]),
                player_names=built["player_names"] + names)


def ball_regressors(games, matches, log=None):
    """The ball sensor's regressors: one per sample match trained on the other two (for that
    match's own features), and one on all of them (for the challenge match).

    ``matches[i]`` lists the roughened copies of ``games[i]``; the regressors learn from those
    copies, with the clean match's ball as the target."""
    from . import sensor
    say = log or (lambda *_: None)
    rows = []
    for game, copies in zip(games, matches):
        rows.append([sensor.regressor_rows(copy, game["bx"], game["by"], model.graded_mask(copy))
                     for copy in copies])
    others = {i: sensor.fit_ball([r for j in range(len(games)) if j != i for r in rows[j]])
              for i in range(len(games))}
    say("ball sensor fitted without each sample match")
    everything = sensor.fit_ball([r for group in rows for r in group])
    say("ball sensor fitted on all sample matches")
    return others, everything


def spell_weights(match, exponent, typical=125.0, low=0.2, high=5.0):
    """Per-frame training weight that shrinks with the length of the possession the frame is in.

    The ranking metric counts every possession once, however long, while a classifier trained on
    frames lets a 40-second possession outvote forty 1-second ones. With ``exponent`` 1 a
    possession's frames share one unit of weight (a possession of ``typical`` frames gets weight
    1 per frame), within [low, high]; with 0 every frame weighs the same.
    """
    labels = match["labels"]
    weight = np.ones(match["n_frames"])
    if not exponent:
        return weight
    for first, last in match["halves"]:
        window = labels[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] != D:
                weight[first + a:first + b] = np.clip(((b - a) / typical) ** -exponent, low, high)
    return weight


def training_rows(match, built, params=model.PARAMS):
    """The strided rows of one labelled match that the three models learn from."""
    labels = match["labels"]
    weight = spell_weights(match, params["spell_weight"])
    stride = np.zeros(match["n_frames"], dtype=bool)
    stride[::params["train_stride"]] = True
    graded = model.graded_mask(match) & stride
    alive = graded & (labels != D)
    seen = alive & built["evidence"]["visible"]
    return {
        "dead_X": built["Xp_dead"][graded], "dead_y": (labels[graded] == D).astype(np.int8),
        "team_X": built["Xp"][alive], "team_twin": built["Xp_twin"][alive],
        "team_y": (labels[alive] == H).astype(np.int8),
        "owner_X": built["Xo"][seen], "owner_twin": built["Xo_twin"][seen],
        "owner_y": (labels[seen] == H).astype(np.int8),
        "all_X": np.hstack([built["Xo"][alive], built["Xp"][alive]]),
        "all_twin": np.hstack([built["Xo_twin"][alive], built["Xp_twin"][alive]]),
        "team_w": weight[alive], "owner_w": weight[seen],
    }


MODELS = ("dead", "team", "owner", "all")


def train(row_sets, params=model.PARAMS, which=MODELS):
    """Fit the named models on the stacked rows of several labelled matches."""
    def stack(key):
        return np.concatenate([rows[key] for rows in row_sets])

    models = {}
    if "dead" in which:
        models["dead"] = model.fit(stack("dead_X"), stack("dead_y"), params,
                                   class_weight="balanced")
    if "team" in which:
        models["team"] = model.fit(stack("team_X"), stack("team_y"), params,
                                   X_twin=stack("team_twin"), sample_weight=stack("team_w"))
    if "owner" in which:
        models["owner"] = model.fit(stack("owner_X"), stack("owner_y"), params,
                                    X_twin=stack("owner_twin"), sample_weight=stack("owner_w"))
    if "all" in which:
        models["all"] = model.fit(stack("all_X"), stack("team_y"), params,
                                  X_twin=stack("all_twin"), sample_weight=stack("team_w"))
    return models


def apply(models, built):
    """Per-frame probabilities for one match, in the form model.frame_scores expects."""
    out = {}
    if "dead" in models:
        out["dead_transfer"] = model.predict(models["dead"], built["Xp_dead"])
    if "team" in models:
        out["team_transfer"] = model.predict(models["team"], built["Xp"], built["Xp_twin"])
    if "owner" in models:
        visible = built["evidence"]["visible"]
        owner = np.full(len(visible), 0.5)
        owner[visible] = model.predict(models["owner"], built["Xo"][visible],
                                       built["Xo_twin"][visible])
        out["owner_transfer"] = owner
    if "all" in models:
        out["all_transfer"] = model.predict(
            models["all"], np.hstack([built["Xo"], built["Xp"]]),
            np.hstack([built["Xo_twin"], built["Xp_twin"]]))
    return out


def from_samples(samples_dir, target, params=model.PARAMS, cache_dir=None, log=None,
                 frames=None):
    """Train on the public sample matches and return the evidence for ``target``.

    1. Each sample match is roughened ``params["augment"]`` times with different noise.
    2. The frame models are fitted on all of them and applied to the target.

    Only if the sequence model is switched on (``w_refined_transfer`` > 0):

    3. For it to learn from honest inputs, every sample match is also scored by frame models
       that never saw it (trained on the other two).
    4. It is fitted on those held-out traces and applied to the target's.
    """
    from . import external, refine, sensor
    say = log or (lambda *_: None)
    games = external.load(samples_dir, cache_dir=cache_dir)
    matches = [[external.degrade(game, seed=i + 100 * k) for k in range(params["augment"])]
               for i, game in enumerate(games)]
    # ``frames``: a file the frame models' outputs are saved to and, when it exists, read from,
    # so that only the sequence model and the decode are re-run. Delete it after changing
    # anything the frame models depend on.
    if frames and os.path.isfile(frames):
        z = np.load(frames)
        sources = {key: z["target_" + key] for key in ("dead_transfer", "team_transfer",
                                                        "owner_transfer", "all_transfer")}
        say("frame models' outputs read from {}".format(frames))
        if not params["w_refined_transfer"]:
            return sources
        sequence_list = []
        for i in range(len(games)):
            for k, match in enumerate(matches[i]):
                trace = {key: z["trace_{}_{}_{}".format(i, k, key)] for key in sources}
                sequence_list += refine.sequences(match, trace, model.ball_evidence(match, params),
                                                  params)
        nets = refine.fit(sequence_list, params, log=say)
        sources["refined_transfer"] = refine.apply(nets, target, sources,
                                                   model.ball_evidence(target, params), params)
        return sources
    sensors, sensor_all = (ball_regressors(games, matches, log=say) if params.get("ball_sensor")
                           else ({}, None))
    built, rows = [], []
    for i, game in enumerate(games):
        built.append([design(match, params,
                             sensor.predict_ball(sensors[i], match) if sensors else None)
                      for match in matches[i]])
        rows.append([training_rows(match, b, params)
                     for match, b in zip(matches[i], built[-1])])
        say("features ready for {}".format(game["name"]))

    models = train([r for group in rows for r in group], params)
    target_built = design(target, params,
                          sensor.predict_ball(sensor_all, target) if sensor_all else None)
    sources = apply(models, target_built)
    say("frame models trained on all sample matches and applied to the challenge match")
    if not params["w_refined_transfer"]:
        return sources

    sequence_list = []
    saved = {"target_" + key: value for key, value in sources.items()}
    for i in range(len(games)):
        held_out = train([r for j in range(len(games)) if j != i for r in rows[j]], params)
        for k, (match, b) in enumerate(zip(matches[i], built[i])):
            trace = apply(held_out, b)
            saved.update({"trace_{}_{}_{}".format(i, k, key): value for key, value in trace.items()})
            sequence_list += refine.sequences(match, trace, b["evidence"], params)
        say("held-out traces ready for {}".format(games[i]["name"]))
    del built, rows
    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(frames)), exist_ok=True)
        np.savez_compressed(frames, **saved)

    nets = refine.fit(sequence_list, params, log=say)
    sources["refined_transfer"] = refine.apply(nets, target, sources, target_built["evidence"],
                                               params)
    return sources
