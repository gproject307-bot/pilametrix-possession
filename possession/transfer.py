"""Classifiers taught by real labels on *other* matches, then applied to the match being decoded.

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

import numpy as np

from . import model

H, A, D = model.H, model.A, model.D


def design(match, params=model.PARAMS):
    """Every feature matrix the transferred models need for one match."""
    evidence = model.ball_evidence(match, params)
    Xp, Xp_twin, player_names = model.player_design(match, with_tags=False)
    Xo, Xo_twin, owner_names = model.owner_design(match, evidence)
    return {"evidence": evidence, "Xp": Xp, "Xp_twin": Xp_twin, "Xo": Xo, "Xo_twin": Xo_twin,
            "player_names": player_names, "owner_names": owner_names}


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
        "dead_X": built["Xp"][graded], "dead_y": (labels[graded] == D).astype(np.int8),
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
        out["dead_transfer"] = model.predict(models["dead"], built["Xp"])
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


def from_samples(samples_dir, target, params=model.PARAMS, cache_dir=None, log=None):
    """Train on the public sample matches and return the evidence for ``target``.

    1. Each sample match is roughened ``params["augment"]`` times with different noise.
    2. The frame models are fitted on all of them and applied to the target.

    Only if the sequence model is switched on (``w_refined_transfer`` > 0):

    3. For it to learn from honest inputs, every sample match is also scored by frame models
       that never saw it (trained on the other two).
    4. It is fitted on those held-out traces and applied to the target's.
    """
    from . import external, refine
    say = log or (lambda *_: None)
    games = external.load(samples_dir, cache_dir=cache_dir)
    matches, built, rows = [], [], []
    for i, game in enumerate(games):
        matches.append([external.degrade(game, seed=i + 100 * k)
                        for k in range(params["augment"])])
        built.append([design(match, params) for match in matches[-1]])
        rows.append([training_rows(match, b, params)
                     for match, b in zip(matches[-1], built[-1])])
        say("features ready for {}".format(game["name"]))

    models = train([r for group in rows for r in group], params)
    target_built = design(target, params)
    sources = apply(models, target_built)
    say("frame models trained on all sample matches and applied to the challenge match")
    if not params["w_refined_transfer"]:
        return sources

    sequence_list = []
    for i in range(len(games)):
        held_out = train([r for j in range(len(games)) if j != i for r in rows[j]], params)
        for match, b in zip(matches[i], built[i]):
            sequence_list += refine.sequences(match, apply(held_out, b), b["evidence"], params)
        say("held-out traces ready for {}".format(games[i]["name"]))
    del built, rows

    nets = refine.fit(sequence_list, params, log=say)
    sources["refined_transfer"] = refine.apply(nets, target, sources, target_built["evidence"],
                                               params)
    return sources
