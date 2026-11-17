"""Two extra sources of evidence about who has the ball, added after the first submission.

Ball sensor
    On a frame without a tracked ball the 22 players still react to it: pressers converge on the
    carrier, the team in possession spreads. A regressor trained on the sample matches' true ball
    predicts where the ball is from identity-free player features alone (about 7 m median error
    on a match it never saw, against 18 m for the players' centroid). On ball-less frames the
    predicted ball gives "pseudo-ball" features (each side's nearest player, players within 3 and
    6 m), appended to the player features.

Facing
    Body orientation is not in the data. For the two players of each side nearest the ball it is
    guessed: the direction of running when moving fast, towards the ball when standing. With it
    come how each of them moves relative to the ball (closing speed, moving with the ball,
    relative speed). Appended to the ball-neighbourhood features.

Both are built home view and away view, so the models stay exactly symmetric between the sides.
"""

import warnings

import numpy as np

from . import features as F
from .io import FPS, HOME, AWAY, PITCH

L2, W2 = PITCH[0] / 2.0, PITCH[1] / 2.0
REG_STRIDE = 10


def _nan(function, values, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(values, axis=axis)


# -- ball sensor: regressor inputs ---------------------------------------------------------------

def _convergence(X, Y, VX, VY, S, min_speed=1.5):
    """Least-squares point the moving players' heading lines point towards (ridge to centroid)."""
    ok = (S > min_speed) & ~np.isnan(X) & ~np.isnan(VX)
    w = np.where(ok, S, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx, dy = np.where(ok, VX / S, 0.0), np.where(ok, VY / S, 0.0)
    x0, y0 = np.nan_to_num(X), np.nan_to_num(Y)
    a11 = (w * (1 - dx * dx)).sum(1)
    a12 = (w * (-dx * dy)).sum(1)
    a22 = (w * (1 - dy * dy)).sum(1)
    b1 = (w * ((1 - dx * dx) * x0 - dx * dy * y0)).sum(1)
    b2 = (w * (-dx * dy * x0 + (1 - dy * dy) * y0)).sum(1)
    lam = 0.1 * w.sum(1) + 1e-3
    cx = np.nan_to_num(_nan(np.nanmean, X))
    cy = np.nan_to_num(_nan(np.nanmean, Y))
    a11, a22 = a11 + lam, a22 + lam
    b1, b2 = b1 + lam * cx, b2 + lam * cy
    det = a11 * a22 - a12 * a12
    return (np.clip((a22 * b1 - a12 * b2) / det, -65, 65),
            np.clip((a11 * b2 - a12 * b1) / det, -45, 45))


def _regressor_inputs(match, velocity, first, sign):
    """Identity-free player features in coordinates where side ``first`` attacks towards +x."""
    px, py, team = match["px"], match["py"], match["team"]
    vx, vy, speed = velocity
    s = sign.astype(np.float64)[:, None]
    X = (px.astype(np.float64) - L2) * s
    Y = py.astype(np.float64) - W2
    VX = vx.astype(np.float64) * s
    VY = vy.astype(np.float64)
    S = speed.astype(np.float64)
    a, b = np.flatnonzero(team == first), np.flatnonzero(team != first)
    cols = []
    for c in (a, b):
        x, y = X[:, c], Y[:, c]
        cols += [_nan(np.nanmean, x), _nan(np.nanmean, y), _nan(np.nanstd, x), _nan(np.nanstd, y),
                 _nan(np.nanmean, VX[:, c]), _nan(np.nanmean, VY[:, c]), _nan(np.nanmean, S[:, c])]
        xs = np.sort(np.where(np.isnan(x), np.inf, x), axis=1)
        xs = np.where(np.isinf(xs), np.nan, xs)
        cols += [xs[:, k] if k < xs.shape[1] else np.full(len(xs), np.nan) for k in range(11)]
        ys = np.sort(np.where(np.isnan(y), np.inf, y), axis=1)
        ys = np.where(np.isinf(ys), np.nan, ys)
        cols += [ys[:, k] if k < ys.shape[1] else np.full(len(ys), np.nan) for k in (1, 3, 5, 7, 9)]
        cols += list(_convergence(x, y, VX[:, c], VY[:, c], S[:, c]))
    cols += list(_convergence(X, Y, VX, VY, S))
    w = np.where(np.isnan(S) | np.isnan(X), 0.0, S ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols += [(w * np.nan_to_num(X)).sum(1) / w.sum(1), (w * np.nan_to_num(Y)).sum(1) / w.sum(1)]
    order = np.argsort(np.where(np.isnan(S), -np.inf, -S), axis=1)[:, :4]      # the 4 fastest
    rows = np.arange(len(X))[:, None]
    for M in (X, Y, S):
        picked = M[rows, order]
        cols += [picked[:, k] for k in range(4)]
    # The closest opposing pair, and the players most pressed by opponents within 6 m.
    d = np.hypot(X[:, a][:, :, None] - X[:, b][:, None, :],
                 Y[:, a][:, :, None] - Y[:, b][:, None, :]).astype(np.float32)
    dfill = np.where(np.isnan(d), np.inf, d)
    del d
    flat = dfill.reshape(len(X), -1)
    k1 = np.argmin(flat, axis=1)
    ia, ib = np.unravel_index(k1, (len(a), len(b)))
    r = np.arange(len(X))
    dmin = flat[r, k1]
    ok = np.isfinite(dmin)
    cols += [np.where(ok, 0.5 * (X[r, a[ia]] + X[r, b[ib]]), np.nan),
             np.where(ok, 0.5 * (Y[r, a[ia]] + Y[r, b[ib]]), np.nan),
             np.where(ok, dmin, np.nan)]
    pressed = ((dfill < 6.0).sum(2), (dfill < 6.0).sum(1))
    del dfill, flat
    for c, pr in zip((a, b), pressed):
        top = np.argsort(-pr, axis=1, kind="stable")[:, :2]
        for k in range(2):
            cols += [X[r, c[top[:, k]]], Y[r, c[top[:, k]]], pr[r, top[:, k]].astype(np.float64)]
    out = np.column_stack(cols).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def regressor_design(match):
    """(home view, away view, attack sign) of the regressor inputs for one match."""
    from .model import attack_sign
    velocity = F.player_velocities(match["px"], match["py"])
    sign = attack_sign(match)
    return (_regressor_inputs(match, velocity, HOME, sign),
            _regressor_inputs(match, velocity, AWAY, -sign), sign)


def regressor_rows(match, true_bx, true_by, graded):
    """Training rows of one labelled match: inputs, and the true ball in the home-attacks-+x frame."""
    home, away, sign = regressor_design(match)
    tx = (np.asarray(true_bx, dtype=np.float64) - L2) * sign
    ty = np.asarray(true_by, dtype=np.float64) - W2
    ok = graded & ~np.isnan(tx) & ~np.isnan(ty)
    index = np.flatnonzero(ok)[::REG_STRIDE]
    return home[index], away[index], tx[index], ty[index]


def fit_ball(row_sets, seed=0):
    """Two boosted regressors (x, y) of the ball position, trained on both views of each frame."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.vstack([r[0] for r in row_sets] + [r[1] for r in row_sets])
    tx = np.concatenate([r[2] for r in row_sets] + [-r[2] for r in row_sets])
    ty = np.concatenate([r[3] for r in row_sets] + [r[3] for r in row_sets])
    keep = np.array([len(np.unique(c[~np.isnan(c)])) >= 2 for c in X.T])
    X = X[:, keep]
    regs = []
    for y in (tx, ty):
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                            min_samples_leaf=100, l2_regularization=1.0,
                                            random_state=seed)
        regs.append(reg.fit(X, y))
    return keep, regs[0], regs[1]


def predict_ball(regressor, match):
    """Predicted ball (x, y in pitch metres), its position along the home attacking direction, and
    how much the two views disagree (a rough uncertainty)."""
    keep, rx, ry = regressor
    home, away, sign = regressor_design(match)
    home, away = home[:, keep], away[:, keep]
    x1, x2 = rx.predict(home), -rx.predict(away)
    y1, y2 = ry.predict(home), ry.predict(away)
    x, y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return {"x": x * sign + L2, "y": y + W2, "along": x, "spread": np.hypot(x1 - x2, y1 - y2)}


def pseudo_ball_features(match, evidence, ball):
    """Possession features from the predicted ball, on frames without a tracked ball (NaN where
    the ball is tracked). Returns (X, X_twin, names)."""
    px, py, team = match["px"], match["py"], match["team"]
    hidden = ~evidence["visible"]
    per = {}
    for code in (HOME, AWAY):
        cols = np.flatnonzero(team == code)
        with np.errstate(invalid="ignore"):
            d = np.hypot(px[:, cols] - ball["x"][:, None], py[:, cols] - ball["y"][:, None])
        d = np.where(np.isnan(d), np.inf, d)
        near = d.min(1)
        near[~np.isfinite(near)] = np.nan
        per[code] = {"near": near, "n3": (d < 3).sum(1).astype(np.float64),
                     "n6": (d < 6).sum(1).astype(np.float64)}

    def side(own, opp, xsign):
        o, p = per[own], per[opp]
        columns = {"own_near": o["near"], "opp_near": p["near"], "diff_near": o["near"] - p["near"],
                   "own_n3": o["n3"], "opp_n3": p["n3"], "own_n6": o["n6"], "opp_n6": p["n6"],
                   "diff_n6": o["n6"] - p["n6"], "along": ball["along"] * xsign,
                   "spread": ball["spread"]}
        out = {}
        for key, values in columns.items():
            values = np.where(hidden, values, np.nan)
            out[key] = values
            if key in ("diff_near", "own_near", "opp_near", "diff_n6", "along", "spread"):
                out[key + "_m12"] = F.box_mean(values, 12)
                out[key + "_m37"] = F.box_mean(values, 37)
        return out

    home, away = side(HOME, AWAY, 1.0), side(AWAY, HOME, -1.0)
    keys = sorted(home)
    names = ["pb_" + k for k in keys]
    return (np.column_stack([home[k] for k in keys]).astype(np.float32),
            np.column_stack([away[k] for k in keys]).astype(np.float32), names)


# -- facing --------------------------------------------------------------------------------------

def _unit(x, y):
    n = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return x / n, y / n


def facing_features(match, evidence):
    """For each side's two players nearest the ball: a guessed facing and how they move relative
    to the ball. NaN without a tracked ball. Returns (X, X_twin, names)."""
    from .model import attack_sign
    px, py, bx, by = match["px"], match["py"], match["bx"], match["by"]
    vx, vy, speed = F.player_velocities(px, py)
    bhx, bhy = _unit(evidence["ball"]["vx"], evidence["ball"]["vy"])
    bvx, bvy = evidence["ball"]["vx"], evidence["ball"]["vy"]
    sign = attack_sign(match).astype(np.float64)
    rows = np.arange(match["n_frames"])

    def side(code):
        cols = np.flatnonzero(match["team"] == code)
        attack = sign if code == HOME else -sign
        dx, dy = bx[:, None] - px[:, cols], by[:, None] - py[:, cols]
        order = np.argsort(np.where(np.isnan(dx), np.inf, np.hypot(dx, dy)), axis=1)[:, :2]
        out = {}
        for k in range(2):
            c = cols[order[:, k]]
            pvx, pvy, ps = vx[rows, c], vy[rows, c], speed[rows, c]
            tx, ty = _unit(dx[rows, order[:, k]], dy[rows, order[:, k]])
            hx, hy = _unit(pvx, pvy)
            w = np.clip((np.nan_to_num(ps) - 1.0) / 3.0, 0.0, 1.0)
            fx, fy = _unit(w * np.nan_to_num(hx) + (1 - w) * tx, w * np.nan_to_num(hy) + (1 - w) * ty)
            tag = "n{}_".format(k + 1)
            out[tag + "face_goal"] = fx * attack
            out[tag + "face_ball_run"] = hx * tx + hy * ty
            out[tag + "close_speed"] = pvx * tx + pvy * ty
            out[tag + "with_ball"] = hx * bhx + hy * bhy
            out[tag + "rel_speed"] = np.hypot(pvx - bvx, pvy - bvy)
            out[tag + "back_to_play"] = -(fx * attack)
        for key in list(out):
            out[key] = np.where(evidence["visible"], out[key], np.nan)
            out[key + "_m12"] = F.box_mean(out[key], 12)
        return out

    home, away = side(HOME), side(AWAY)
    keys = sorted(home)

    def stack(a, b):
        return np.column_stack([a[k] for k in keys] + [b[k] for k in keys]
                               + [a[k] - b[k] for k in keys]).astype(np.float32)

    names = (["face_own_" + k for k in keys] + ["face_opp_" + k for k in keys]
             + ["face_diff_" + k for k in keys])
    return stack(home, away), stack(away, home), names
