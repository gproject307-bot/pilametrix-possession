"""Evidence, pseudo-labels, classifiers and the decode.

Everything here works on a *match*: a dict with ``px, py`` (N x S player positions in metres),
``bx, by`` (ball), ``team`` (side of every track slot), ``n_frames``, ``halves`` (inclusive graded
blocks) and ``tags`` (Smart Tagging instances, possibly empty). The same code therefore runs on
the challenge match and on the other matches used as a benchmark.

Three layers.

1. Ball evidence. Where the ball is tracked, the nearest player of each side says who is on it,
   and its position and motion say whether it can be in play.
2. Learned evidence, of two kinds, both optional:
   - *self-trained*: classifiers that read the players' collective behaviour, taught by
     pseudo-labels that layer 1 produces on the match's own ball-visible frames;
   - *transferred*: the same kind of classifiers, plus one that reads the ball's neighbourhood,
     taught by real labels on other matches (see transfer.py).
3. A three-state (home / away / dead) Viterbi decode that fuses whatever evidence is present,
   with a cost for changing state, so a possession is an interval rather than a string of frames.
"""

import numpy as np

from . import features as F
from . import hmm
from .io import FPS, HOME, AWAY, PITCH

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D", "X"])
SET_PIECES = "SET PIECES"

PARAMS = {
    # -- ball evidence, team ---------------------------------------------------------------------
    "near_full": 2.0,      # ball within this of a player: full-weight evidence
    "near_zero": 4.0,      # beyond this of every player: no evidence (ball in flight / loose)
    "gap_scale": 2.0,      # metres of nearest-player advantage that saturate the evidence
    "w_rule_team": 0.0,    # the nearest-player rule; 1.0 in SELF_TRAINED
    # -- ball evidence, dead/alive ---------------------------------------------------------------
    "outside_margin": 0.5, # metres beyond a pitch line before the ball counts as out
    "w_outside": 2.0,
    "w_rest": 0.5,         # lying still for two seconds: usually a restart being set up
    "w_moving": -0.2,      # tracked on the pitch and moving: mild evidence of play (the ball is
                           # also carried around during stoppages, so only mild)
    "w_missing": 0.1,      # not tracked at all: most ball-less frames are dead ball
    # -- learned evidence: weight on each source's log-odds, per frame ---------------------------
    # The fusion weights and switch costs below were chosen on other matches with known answers
    # (benchmark.py --tune), never on the challenge match.
    "w_dead_transfer": 0.25,
    "w_owner_transfer": 0.25,
    "w_all_transfer": 0.25,
    "w_refined_transfer": 1.0,     # the sequence model (refine.py); 0.5 in the first submission
    "w_team_transfer_hidden": 0.5,
    "w_team_transfer_visible": 0.0,
    "w_dead_self": 0.0,
    "w_team_self_hidden": 0.0,     # where the ball is not tracked
    "w_team_self_visible": 0.0,    # where it is
    "clip_logit": 2.0,
    # -- cost of changing state ------------------------------------------------------------------
    "switch_team": 12.0,
    "switch_dead": 12.0,   # must exceed switch_team / 2
    "min_dead": 50,        # frames; shorter stoppages are absorbed
    # A change of possession happens when the other side plays the ball, so its cost can depend on
    # whether a ball touch is nearby. All three factors at 1.0 switch this off.
    "anchor_near": 1.0,    # within anchor_reach frames of a detected touch
    "anchor_far": 1.0,     # ball tracked, no touch nearby
    "anchor_unseen": 1.0,  # ball not tracked, so a touch cannot be seen
    "anchor_reach": 4,
    "after_goal_dead": 30 * FPS,  # frames before an after-goal kickoff that are certainly dead
    # -- classifiers -----------------------------------------------------------------------------
    "train_stride": 5,
    "boost_iters": 400,
    "boost_leaves": 31,
    "augment": 2,          # noise realisations of each training match (transfer.py)
    "spell_weight": 0.0,   # 0: every frame counts the same when training the ownership models;
                           # 1: every possession counts the same, however short (transfer.py)
    # -- sequence model (refine.py) --------------------------------------------------------------
    "refine_spell_weight": 0.5,
    "refine_epochs": 30,
    "refine_width": 64,
    "refine_layers": 10,   # dilations 1 .. 512 frames, unless refine_reach is set
    "refine_reach": 125,   # frames of context each side (5 s); 5 to 45 s all score alike
    # -- extra evidence (sensor.py) ---------------------------------------------------------------
    "ball_sensor": True,   # predict the missing ball from the players
    "facing": True,        # guessed facing of the players nearest the ball
    "refine_dropout": 0.2,
    "refine_crop": 6000,   # frames per training window
    "refine_batch": 8,
    "refine_lr": 1e-3,
    "refine_seeds": (0, 1, 2),
    "refine_device": "cpu",  # deterministic; "cuda" is faster and gives slightly different files
    "n_folds": 5,          # per half, contiguous in time
    "self_training_rounds": 1,
    "self_training_margin": 25,  # frames from a decoded boundary before a frame is trusted
    "seed": 0,
}


# Alternatives, all benchmarked (benchmark.py):
# the frame models alone, without the sequence model (the previous submission) ...
FRAME_ONLY = dict(PARAMS, w_refined_transfer=0.0, w_all_transfer=0.5, w_team_transfer_hidden=0.0,
                  w_dead_transfer=0.25, clip_logit=3.0, switch_team=8.0, switch_dead=12.0,
                  w_rule_team=0.0, ball_sensor=False, facing=False)
# ... and the sequence model alone, which finds a few more short possessions and splits many
# more long ones.
SEQUENCE = dict(PARAMS, w_refined_transfer=1.0, w_all_transfer=0.0, w_dead_transfer=0.15,
                switch_team=6.0, w_rule_team=0.0)

# The first submission (sha256 3bf5c3e8...): no ball sensor, no facing, 40 s of context.
FIRST_SUBMISSION = dict(PARAMS, ball_sensor=False, facing=False, refine_reach=None,
                        w_rule_team=0.0, w_all_transfer=0.25, w_refined_transfer=0.5,
                        switch_team=12.0, switch_dead=12.0)

# The earlier method, kept for comparison: the nearest-player rule plus classifiers taught only by
# this match's own ball-derived pseudo-labels. It needs no other match.
SELF_TRAINED = dict(PARAMS, w_rule_team=1.0, w_dead_transfer=0.0, w_owner_transfer=0.0,
                    w_all_transfer=0.0, w_refined_transfer=0.0, w_dead_self=0.25,
                    w_team_self_hidden=0.25,
                    switch_team=16.0, switch_dead=12.0, w_missing=0.0, boost_iters=200,
                    boost_leaves=15)


# -- layout of a match ---------------------------------------------------------------------------

def graded_mask(match):
    mask = np.zeros(match["n_frames"], dtype=bool)
    for first, last in match["halves"]:
        mask[first:last + 1] = True
    return mask


def attack_sign(match):
    """+1 on frames where the home side attacks towards +x, -1 otherwise; +1 outside the halves.

    Decided per half from where the home side's deepest player (the goalkeeper) lives.
    """
    import warnings
    sign = np.ones(match["n_frames"], dtype=np.int8)
    home = np.flatnonzero(match["team"] == HOME)
    for first, last in match["halves"]:
        block = match["px"][first:last + 1:25, home]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            deepest = np.nanmedian(np.nanmin(block, axis=1))
            highest = np.nanmedian(np.nanmax(block, axis=1))
        sign[first:last + 1] = 1 if deepest < PITCH[0] - highest else -1
    return sign


# -- Smart Tagging -------------------------------------------------------------------------------

def tagging_features(match):
    """Per-frame columns from the Smart Tagging intervals, or None for a match without tags.

    One signed column per phase code: +1 while an instance tagged for the home side is open, -1
    for the away side. Which codes mean "has the ball" and which mean "is defending" is left for
    the classifier to learn. Set pieces add an open/closed flag, a signed team column and the time
    to the nearest set-piece start, because a set piece marks the end of a stoppage.
    """
    tags, n = match["tags"], match["n_frames"]
    if not tags:
        return None
    codes = sorted({tag["code"] for tag in tags if tag["code"] != SET_PIECES})
    columns = {code: np.zeros(n, dtype=np.float32) for code in codes}
    sp_open = np.zeros(n, dtype=np.float32)
    sp_team = np.zeros(n, dtype=np.float32)
    starts = []
    for tag in tags:
        a, b = max(tag["start"], 0), min(tag["end"], n)
        if b <= a:
            continue
        signed = 1.0 if tag["team"] == HOME else -1.0 if tag["team"] == AWAY else 0.0
        if tag["code"] == SET_PIECES:
            sp_open[a:b] = 1.0
            sp_team[a:b] += signed
            starts.append(a)
        else:
            columns[tag["code"]][a:b] += signed

    index = np.arange(n)
    starts = np.array(sorted(starts))
    position = np.searchsorted(starts, index)
    after = index - starts[np.clip(position - 1, 0, len(starts) - 1)]
    after = np.where(position > 0, after, 10 ** 6)
    before = starts[np.clip(position, 0, len(starts) - 1)] - index
    before = np.where(position < len(starts), before, 10 ** 6)
    limit = 60 * FPS
    names = ["st_" + code.lower().replace(" ", "_") for code in codes]
    team_block = np.column_stack([columns[code] for code in codes])
    other = np.column_stack([sp_open, np.clip(after, 0, limit), np.clip(before, 0, limit)])
    return {"team_signed": team_block, "team_names": names, "sp_team": sp_team,
            "neutral": other.astype(np.float32),
            "neutral_names": ["sp_open", "sp_since_start", "sp_until_start"]}


def kickoff_frames(match):
    """[(tag, frame)] for every tagged kickoff: the frame the ball is put in play.

    The kickoff tag is a window of a few seconds around the kick. The tracker usually picks the
    ball up on the centre spot at the kick itself, so the first frame inside the window with the
    ball within 2 m of the spot is taken as the kick. Where the ball is not seen there, the
    median offset of the kickoffs where it was is used instead.
    """
    n = match["n_frames"]
    kickoffs = [tag for tag in match["tags"]
                if tag["code"] == SET_PIECES and "kick off" in tag["type"]]
    bx, by = match["bx"], match["by"]
    offsets = []
    for tag in kickoffs:
        a, b = max(tag["start"], 0), min(tag["end"], n)
        with np.errstate(invalid="ignore"):
            near = np.hypot(bx[a:b] - PITCH[0] / 2.0, by[a:b] - PITCH[1] / 2.0) < 2.0
        hits = np.flatnonzero(near)
        offsets.append(int(hits[0]) if len(hits) else None)
    seen = [offset for offset in offsets if offset is not None]
    out = []
    for tag, offset in zip(kickoffs, offsets):
        if offset is None:
            offset = int(np.median(seen)) if seen else (2 * (tag["end"] - tag["start"])) // 3
        out.append((tag, tag["start"] + offset))
    return out


def forced_dead(match, params=PARAMS):
    """Frames that are dead by construction, read off the automatic kickoff tags.

    A half is not in play before its kickoff, and a kickoff after a goal is preceded by at least
    half a minute of celebration.
    """
    forced = np.zeros(match["n_frames"], dtype=bool)
    for tag, kick in kickoff_frames(match):
        if "start" in tag["type"]:
            for first, last in match["halves"]:
                if first < kick <= last:
                    forced[first:kick] = True
        else:
            forced[max(tag["start"] - params["after_goal_dead"], 0):kick] = True
    return forced


# -- layer 1: ball evidence ----------------------------------------------------------------------

def ball_evidence(match, params=PARAMS):
    """Per-frame evidence from the tracked ball. Zero wherever the ball is not tracked."""
    ball = F.ball_features(match["bx"], match["by"])
    d_home, d_away = F.nearest_by_team(match["bx"], match["by"], match["px"], match["py"],
                                       match["team"])
    visible = ball["visible"]

    nearest = np.minimum(d_home, d_away)
    weight = np.clip((params["near_zero"] - nearest)
                     / (params["near_zero"] - params["near_full"]), 0.0, 1.0)
    team = np.clip((d_away - d_home) / params["gap_scale"], -1.0, 1.0) * weight
    team = np.where(visible & np.isfinite(team), team, 0.0)

    # Off the pitch is dead. Lying still is usually a restart being set up. On the pitch and
    # moving is mild evidence of play.
    outside = visible & (ball["outside"] > params["outside_margin"])
    rest = visible & ball["at_rest"] & ~outside
    moving = visible & ~outside & ~rest

    dead = np.zeros(len(visible))
    dead[outside] = params["w_outside"]
    dead[rest] = params["w_rest"]
    dead[moving] = params["w_moving"]
    return {"ball": ball, "d_home": d_home, "d_away": d_away, "team": team, "dead": dead,
            "outside": outside, "rest": rest, "moving": moving, "visible": visible}


def anchors(match, evidence, params=PARAMS):
    """High-precision pseudo-labels read off the ball alone.

    dead   the ball is clearly off the pitch, or lying still
    live   the ball is tracked well inside the pitch, has been struck recently, and nothing dead
           is within three seconds either side
    home / away   live frames where a ball-only decode is settled on one side for two seconds
                  either side
    """
    n = match["n_frames"]
    ball = evidence["ball"]
    visible = evidence["visible"]
    dead = visible & ((ball["outside"] > 1.0) | ball["at_rest"])

    def near(mask, half):
        return F.box_mean(mask.astype(np.float64), half) > 0

    speed = np.where(visible, ball["speed"], 0.0)
    struck = near(speed >= 8.0, 2 * FPS)
    tracked_around = F.box_mean(visible.astype(np.float64), FPS) >= 0.999
    live = (visible & (ball["inside"] >= 1.0) & ~ball["at_rest"] & struck & tracked_around
            & ~near(dead, 3 * FPS))

    # Ball-only team decode: two states, sticky, evidence only where a player is on the ball.
    scores = np.column_stack([0.5 * evidence["team"], -0.5 * evidence["team"]])
    penalty = np.array([[0.0, params["switch_team"]], [params["switch_team"], 0.0]])
    side = np.full(n, -1, dtype=np.int8)
    for first, last in match["halves"]:
        side[first:last + 1] = hmm.viterbi(scores[first:last + 1], penalty)
    changed = np.zeros(n, dtype=bool)
    changed[1:] = side[1:] != side[:-1]
    settled = live & ~near(changed, 2 * FPS)
    return {"dead": dead, "live": live, "home": settled & (side == HOME),
            "away": settled & (side == AWAY), "ball_side": side}


# -- layer 2: features for the classifiers -------------------------------------------------------

def player_design(match, with_tags=True):
    """Collective player features seen from the home side, and the twin seen from the away side.

    ``with_tags`` appends the Smart Tagging columns; models meant to transfer between matches
    leave them out, because other matches have none.
    """
    sign = attack_sign(match)
    tagging = tagging_features(match) if with_tags else None

    def design(team, direction, tag_sign):
        base, names = F.player_features(match["px"], match["py"], team, direction)
        matrix, names = F.with_context(base, names)
        if tagging is not None:
            signed = tagging["team_signed"] * tag_sign
            sp_team = (tagging["sp_team"] * tag_sign)[:, None]
            matrix = np.hstack([matrix, signed, sp_team, tagging["neutral"]]).astype(np.float32)
            names = names + tagging["team_names"] + ["sp_team"] + tagging["neutral_names"]
        return matrix, names

    X, names = design(match["team"], sign, 1.0)
    X_twin, _ = design(1 - match["team"], -sign, -1.0)
    return X, X_twin, names


def owner_design(match, evidence):
    """Ball-local features of who is on the ball, and their mirrored twin."""
    return F.owner_features(evidence["ball"], evidence["d_home"], evidence["d_away"], match)


# -- layer 2: fitting ----------------------------------------------------------------------------

def fit(X, y, params=PARAMS, X_twin=None, class_weight=None, sample_weight=None):
    """A gradient-boosted classifier. With ``X_twin`` (the same frames seen from the other side,
    whose label is the opposite) it is trained on both views, which makes it symmetric."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    if X_twin is not None:
        X = np.vstack([X, X_twin])
        y = np.concatenate([y, 1 - y])
        if sample_weight is not None:
            sample_weight = np.concatenate([sample_weight, sample_weight])
    model = HistGradientBoostingClassifier(
        max_iter=params["boost_iters"], learning_rate=0.06,
        max_leaf_nodes=params["boost_leaves"], min_samples_leaf=200,
        l2_regularization=1.0, max_features=0.5, early_stopping=False,
        class_weight=class_weight, random_state=params["seed"])
    return model.fit(X, y, sample_weight=sample_weight)


def predict(model, X, X_twin=None):
    """P(positive). With a twin the two views are averaged, so P(home) + P(away) = 1 exactly."""
    probability = model.predict_proba(X)[:, 1]
    if X_twin is not None:
        probability = 0.5 * (probability + 1.0 - model.predict_proba(X_twin)[:, 1])
    return probability


def _fold_ids(match, n_folds):
    """Contiguous time folds inside each half; -1 outside the graded blocks."""
    fold = np.full(match["n_frames"], -1, dtype=np.int16)
    for half, (first, last) in enumerate(match["halves"]):
        edges = np.linspace(first, last + 1, n_folds + 1).astype(int)
        for k in range(n_folds):
            fold[edges[k]:edges[k + 1]] = half * n_folds + k
    return fold


def out_of_fold(match, X, y, train, params=PARAMS, X_twin=None, class_weight=None):
    """Out-of-fold P(positive) for every graded frame of one match: no frame is scored by a
    model that saw it or anything within five seconds of its fold."""
    n = match["n_frames"]
    fold = _fold_ids(match, params["n_folds"])
    out = np.full(n, np.nan)
    stride = np.zeros(n, dtype=bool)
    stride[::params["train_stride"]] = True
    for k in np.unique(fold[fold >= 0]):
        held = fold == k
        guard = F.box_mean(held.astype(np.float64), 5 * FPS) > 0
        rows = train & stride & (fold >= 0) & ~guard
        model = fit(X[rows], y[rows], params, None if X_twin is None else X_twin[rows],
                    class_weight)
        out[held] = predict(model, X[held], None if X_twin is None else X_twin[held])
    return out


def self_trained_sources(match, X, X_twin, is_dead, dead_train, is_home, team_train,
                         params=PARAMS):
    """Out-of-fold P(dead) and P(home | alive) from the players, taught by pseudo-labels.

    The dead/alive classifier is class-balanced, so its output is a likelihood ratio rather than
    a posterior under the pseudo-labels' own class mix."""
    p_dead = out_of_fold(match, X, is_dead.astype(np.int8), dead_train, params,
                         class_weight="balanced")
    p_home = out_of_fold(match, X, is_home.astype(np.int8), team_train, params, X_twin=X_twin)
    return {"dead_self": p_dead, "team_self": p_home}


# -- layer 3: decode -----------------------------------------------------------------------------

def _logit(p, clip):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-4, 1 - 1e-4)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def frame_scores(evidence, sources, params=PARAMS, hide=None):
    """(N, 3) log-scores for home / away / dead.

    ``team`` is the log-odds of home against away and ``dead`` the log-odds of dead against alive.
    The side the team evidence favours scores 0 and the other side pays for it, so how *sure* the
    evidence is about the team never leaks into the separate question of whether the ball is in
    play at all.

    ``sources`` maps a source name to a per-frame probability; absent sources are skipped:
      dead_self, dead_transfer        P(dead) from the players
      team_self, team_transfer        P(home | alive) from the players
      owner_transfer                  P(home | alive) from the ball's neighbourhood
      all_transfer                    P(home | alive) from the ball and the players together
      refined_transfer                P(home | alive) from the sequence model over all of these

    ``hide`` masks the ball evidence on some frames, which is how the gap filling can be checked
    without any ground truth.
    """
    clip = params["clip_logit"]
    team = params["w_rule_team"] * evidence["team"]
    dead = evidence["dead"].copy()
    seen = evidence["visible"]
    if hide is not None:
        team = np.where(hide, 0.0, team)
        dead[hide] = 0.0
        seen = seen & ~hide
    dead = dead + np.where(seen, 0.0, params["w_missing"])

    for name in ("self", "transfer"):
        p = sources.get("team_" + name)
        if p is not None:
            weight = np.where(seen, params["w_team_{}_visible".format(name)],
                              params["w_team_{}_hidden".format(name)])
            team = team + weight * _logit(p, clip)
        p = sources.get("dead_" + name)
        if p is not None:
            dead = dead + params["w_dead_" + name] * _logit(p, clip)
    if hide is None:  # these two read the ball, so they cannot be used with the ball masked
        for name in ("owner", "all", "refined"):
            p = sources.get(name + "_transfer")
            if p is not None:
                team = team + params["w_{}_transfer".format(name)] * _logit(p, clip)

    return np.column_stack([-np.maximum(-team, 0.0) - 0.5 * dead,
                            -np.maximum(team, 0.0) - 0.5 * dead,
                            0.5 * dead])


def touch_mask(match, reach):
    """True within ``reach`` frames of a detected ball touch, at either time scale."""
    key = "_touch_near_{}".format(reach)
    if key not in match:
        ball = F.ball_features(match["bx"], match["by"])
        marks = np.zeros(match["n_frames"], dtype=bool)
        for span in (3, 7):
            frames, _ = F.ball_touches(ball, span=span)
            marks[frames] = True
        match[key] = F.box_mean(marks.astype(np.float64), reach) > 0
    return match[key]


def decode(match, scores, params=PARAMS, forced=None):
    """Label codes for every frame: H / A / D inside the graded halves, X outside.

    ``forced`` marks frames that must be dead whatever the evidence says. With the anchor factors
    set, a change of possession costs ``switch_team`` times the factor for that frame; a cost
    above twice ``switch_dead`` is in effect capped there, because the decode can then pass
    through a one-frame stoppage, which is absorbed afterwards."""
    t, d = params["switch_team"], params["switch_dead"]
    if 2 * d <= t:
        raise ValueError("switch_dead must exceed half of switch_team, or a change of possession "
                         "is cheaper through a one-frame stoppage than directly")
    if forced is not None and forced.any():
        scores = scores.copy()
        scores[forced, :2] = -1e6
    anchored = not (params["anchor_near"] == params["anchor_far"] == params["anchor_unseen"] == 1.0)
    if anchored:
        seen = np.isfinite(np.asarray(match["bx"], dtype=np.float64))
        swap = t * np.where(seen, np.where(touch_mask(match, params["anchor_reach"]),
                                           params["anchor_near"], params["anchor_far"]),
                            params["anchor_unseen"])
    penalty = np.array([[0.0, t, d], [t, 0.0, d], [d, d, 0.0]])
    out = np.full(match["n_frames"], 3, dtype=np.int8)
    for first, last in match["halves"]:
        if anchored:
            path = hmm.viterbi_switching(scores[first:last + 1], swap[first:last + 1], d)
        else:
            path = hmm.viterbi(scores[first:last + 1], penalty)
        out[first:last + 1] = absorb_short_stoppages(path, params["min_dead"])
    return out


def absorb_short_stoppages(path, min_dead):
    """Remove dead-ball runs shorter than ``min_dead`` frames; no real stoppage is that short.

    Between two spells of the same side the gap is given to that side, so the possession is not
    cut in two. Between different sides it is split down the middle.
    """
    path = path.copy()
    starts, stops = F.runs(path == D)
    for start, stop in zip(starts, stops):
        if stop - start >= min_dead or stop == len(path):
            continue
        if start == 0:
            path[start:stop] = path[stop]
            continue
        before, after = path[start - 1], path[stop]
        if before == after:
            path[start:stop] = before
        else:
            middle = (start + stop) // 2
            path[start:middle] = before
            path[middle:stop] = after
    return path


def interior(labels, margin):
    """True where a frame is at least ``margin`` frames away from any change of label."""
    changed = np.zeros(len(labels), dtype=bool)
    changed[1:] = labels[1:] != labels[:-1]
    return ~(F.box_mean(changed.astype(np.float64), margin) > 0)


# -- the pipeline --------------------------------------------------------------------------------

def run(match, params=PARAMS, log=None, transferred=None, self_train=True):
    """Decode one match. Returns label codes for every frame and the intermediate evidence.

    ``transferred``  per-frame probabilities from models trained on other matches (transfer.py)
    ``self_train``   also fit the match's own classifiers on its ball-only pseudo-labels
    """
    say = log or (lambda *_: None)
    evidence = ball_evidence(match, params)
    forced = forced_dead(match, params)
    sources = dict(transferred or {})
    result = {"evidence": evidence, "forced": forced, "sources": sources}

    if self_train:
        anchor = anchors(match, evidence, params)
        say("ball evidence and anchors ready")
        X, X_twin, names = player_design(match, with_tags=True)
        say("player features ready: {} columns".format(len(names)))

        # Round 0: classifiers taught by the ball-only anchors.
        sources.update(self_trained_sources(
            match, X, X_twin, anchor["dead"], anchor["dead"] | anchor["live"],
            anchor["home"], anchor["home"] | anchor["away"], params))
        labels = decode(match, frame_scores(evidence, sources, params), params, forced)
        say("round 0 decoded")

        # Self-training: the decode labels every graded frame, including the ones with no ball,
        # so the classifiers can learn what a stoppage or a possession looks like there too. Only
        # frames well inside a decoded segment are used; the ball-only anchors stay authoritative.
        graded = graded_mask(match)
        for round_index in range(params["self_training_rounds"]):
            settled = interior(labels, params["self_training_margin"]) & graded
            is_dead = ((labels == D) & ~anchor["live"]) | anchor["dead"]
            dead_train = settled | anchor["dead"] | anchor["live"]
            team_train = settled & (labels != D) & ~anchor["dead"]
            sources.update(self_trained_sources(match, X, X_twin, is_dead, dead_train,
                                                labels == H, team_train, params))
            labels = decode(match, frame_scores(evidence, sources, params), params, forced)
            say("round {} decoded".format(round_index + 1))
        result.update(anchors=anchor, feature_names=names)
    else:
        labels = decode(match, frame_scores(evidence, sources, params), params, forced)

    result["labels"] = labels
    return result
