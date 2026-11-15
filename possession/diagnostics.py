"""Label-free checks of a decode. None of these uses any ground truth.

They answer "is this output plausible, and does the gap filling hold up?" from the provided
automatic data alone, which is what the fusion weights were checked against.
"""

import numpy as np

from . import features as F
from . import model
from .io import FPS

H, A, D = model.H, model.A, model.D


def _segments(match, labels):
    """[(start, stop, code)] constant runs, per half."""
    out = []
    for first, last in match["halves"]:
        window = labels[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        out += [(first + int(bounds[i]), first + int(bounds[i + 1]), int(window[bounds[i]]))
                for i in range(len(bounds) - 1)]
    return out


def summary(match, labels):
    """Counts a football person can sanity-check: possessions, stoppages, share of dead ball."""
    segments = _segments(match, labels)
    spells = np.array([stop - start for start, stop, code in segments if code != D])
    graded = labels[model.graded_mask(match)]
    alive = max(int(np.sum(graded != D)), 1)
    return {
        "possessions": int(len(spells)),
        "mean_possession_s": float(spells.mean() / FPS),
        "median_possession_s": float(np.median(spells) / FPS),
        "stoppages": int(sum(1 for _, _, code in segments if code == D)),
        "dead_share": float(np.mean(graded == D)),
        "home_share_of_play": float(np.sum(graded == H) / alive),
    }


def set_piece_agreement(match, labels, before=10 * FPS, after=20 * FPS):
    """Does a decoded restart fall near each tagged set piece, with the tagged side on the ball?

    The tags are automatic and noisy, so this is a plausibility check, not a score.
    """
    restarts = np.array([stop for _, stop, code in _segments(match, labels) if code == D])
    pieces = [tag for tag in match["tags"] if tag["code"] == model.SET_PIECES
              and "kick off" not in tag["type"]]
    found = sided = with_side = 0
    for tag in pieces:
        near = restarts[(restarts >= tag["start"] - before) & (restarts <= tag["start"] + after)]
        if not len(near):
            continue
        found += 1
        if tag["team"] >= 0:
            with_side += 1
            restart = int(near[np.argmin(np.abs(near - tag["start"]))])
            if restart < len(labels) and labels[restart] == tag["team"]:
                sided += 1
    return {"set_pieces": len(pieces), "restart_found": found / max(len(pieces), 1),
            "tagged_side_restarts": sided / max(with_side, 1), "with_side": with_side}


def masked_ball_consistency(match, result, params=model.PARAMS, share=0.3, repeats=3, seed=0):
    """Hide the ball where it IS tracked and see whether the decode changes.

    Stretches with the same length distribution as the real ball-less gaps are hidden until
    ``share`` of the ball-visible graded frames are covered. The decode without the ball on those
    frames is compared with the decode that had it, on the hidden frames only.
    """
    evidence, sources, forced = result["evidence"], result["sources"], result["forced"]
    reference = result["labels"]
    graded = model.graded_mask(match)
    visible = evidence["visible"] & graded
    gap_starts, gap_stops = F.runs(~evidence["visible"] & graded)
    gap_lengths = gap_stops - gap_starts
    run_starts, run_stops = F.runs(visible)
    rng = np.random.RandomState(seed)

    dead_agree, team_agree = [], []
    for _ in range(repeats):
        hide = np.zeros(len(visible), dtype=bool)
        target = share * visible.sum()
        for _ in range(20000):
            if hide.sum() >= target:
                break
            k = rng.randint(len(run_starts))
            length = int(gap_lengths[rng.randint(len(gap_lengths))])
            if run_stops[k] - run_starts[k] <= length + 100:
                continue
            start = rng.randint(run_starts[k] + 50, run_stops[k] - length - 50)
            if hide[start - 50:start + length + 50].any():
                continue
            hide[start:start + length] = True
        decoded = model.decode(match, model.frame_scores(evidence, sources, params, hide=hide),
                               params, forced)
        dead_agree.append(np.mean((decoded[hide] == D) == (reference[hide] == D)))
        both_alive = hide & (decoded != D) & (reference != D)
        team_agree.append(np.mean(decoded[both_alive] == reference[both_alive]))
    return {"hidden_share": share, "dead_alive_agreement": float(np.mean(dead_agree)),
            "team_agreement": float(np.mean(team_agree))}


def _auc(score, positive):
    from scipy.stats import rankdata
    ranks = rankdata(score)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def transfer_checks(match, result, params=model.PARAMS):
    """Do models trained on other matches behave on this one? Compared with pseudo-labels read
    off this match's ball alone (model.anchors), which they never saw."""
    sources, evidence = result["sources"], result["evidence"]
    anchor = model.anchors(match, evidence, dict(params, switch_team=16.0))
    graded = model.graded_mask(match)
    out = {}
    rows = (anchor["dead"] | anchor["live"]) & graded
    out["dead_auc"] = _auc(sources["dead_transfer"][rows], anchor["dead"][rows])
    rows = (anchor["home"] | anchor["away"]) & graded
    for name in ("owner_transfer", "all_transfer"):
        if name in sources:
            out[name] = float(np.mean((sources[name][rows] > 0.5) == anchor["home"][rows]))
    return out


def report(match, result, params=model.PARAMS, log=print, masked=True):
    labels = result["labels"]
    s = summary(match, labels)
    log("possessions {possessions}, mean {mean_possession_s:.1f} s, median "
        "{median_possession_s:.1f} s; stoppages {stoppages}; dead share {dead_share:.3f}; "
        "home share of play {home_share_of_play:.3f}".format(**s))
    p = set_piece_agreement(match, labels)
    log("tagged set pieces {set_pieces}: decoded restart nearby {restart_found:.2f}; tagged side "
        "restarts {tagged_side_restarts:.2f} (of {with_side})".format(**p))
    if masked:
        # Only meaningful when no source reads the ball: those cannot be masked after the fact.
        m = masked_ball_consistency(match, result, params)
        log("masked ball ({hidden_share:.0%} of visible frames hidden): dead/alive agreement "
            "{dead_alive_agreement:.3f}, team agreement {team_agreement:.3f}".format(**m))
    if "dead_transfer" in result["sources"]:
        t = transfer_checks(match, result, params)
        log("transferred models against this match's ball-only anchors: dead/alive AUC "
            "{:.3f}; side agreement {}".format(t["dead_auc"], ", ".join(
                "{} {:.3f}".format(k.split("_")[0], v) for k, v in t.items() if k != "dead_auc")))
    visible = result["evidence"]["visible"]
    graded = model.graded_mask(match)
    log("dead share where the ball is tracked {:.3f}, where it is not {:.3f}".format(
        float(np.mean(labels[graded & visible] == D)),
        float(np.mean(labels[graded & ~visible] == D))))
