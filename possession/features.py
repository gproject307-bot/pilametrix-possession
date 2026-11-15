"""Per-frame features computed from the automatic tracking.

Nothing here knows about labels. All distances are metres, all speeds m/s, all windows frames at
25 fps. Player identities do not exist in the data, so every player feature is collective.
"""

import numpy as np

from .io import FPS, HOME, AWAY, PITCH


def runs(mask):
    """Half-open [start, stop) index pairs of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def run_length_of(mask):
    """For every frame, the length of the True run it belongs to (0 where False)."""
    out = np.zeros(len(mask), dtype=np.int32)
    for start, stop in zip(*runs(mask)):
        out[start:stop] = stop - start
    return out


def box_mean(values, half):
    """Centred moving average over +/- half frames that ignores NaN. NaN where the window is empty."""
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0), dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half)
    high = np.minimum(len(values), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def shift(values, k, fill=np.nan):
    """values[t + k] aligned at t (k may be negative), padded with ``fill``."""
    out = np.full(len(values), fill, dtype=np.float64)
    if k > 0:
        out[:-k] = values[k:]
    elif k < 0:
        out[-k:] = values[:k]
    else:
        out[:] = values
    return out


# -- ball ------------------------------------------------------------------------------------------

def ball_features(bx, by):
    """Kinematics and status of the tracked ball.

    speed     m/s from a centred difference over +/- 2 frames (one-sided at the ends of a track)
    outside   metres beyond the nearest pitch line, 0 when inside
    inside    metres to the nearest pitch line when inside, 0 when outside
    at_rest   True where the ball stays within 0.5 m of itself for a second either side
    """
    n = len(bx)
    visible = ~np.isnan(bx)
    x = bx.astype(np.float64)
    y = by.astype(np.float64)

    speed = np.full(n, np.nan)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    for back, ahead in ((2, 2), (1, 1), (0, 2), (2, 0), (0, 1), (1, 0)):
        xa, ya = shift(x, ahead), shift(y, ahead)
        xb, yb = shift(x, -back), shift(y, -back)
        dt = (back + ahead) / float(FPS)
        ok = np.isnan(speed) & visible & ~np.isnan(xa) & ~np.isnan(xb)
        vx[ok] = (xa[ok] - xb[ok]) / dt
        vy[ok] = (ya[ok] - yb[ok]) / dt
        speed[ok] = np.hypot(vx[ok], vy[ok])

    length, width = PITCH
    outside = np.maximum.reduce([-x, x - length, -y, y - width, np.zeros(n)])
    inside = np.maximum(np.minimum.reduce([x, length - x, y, width - y]), 0.0)
    outside[~visible] = np.nan
    inside[~visible] = np.nan

    # At rest: every observation within +/- 1 s is within 0.5 m of the current one, and the track
    # exists for at least a second around the frame.
    half = FPS
    spread = np.zeros(n)
    seen = np.zeros(n, dtype=np.int32)
    for k in range(-half, half + 1, 3):
        xs, ys = shift(x, k), shift(y, k)
        ok = ~np.isnan(xs)
        distance = np.where(ok, np.hypot(xs - x, ys - y), 0.0)
        spread = np.maximum(spread, distance)
        seen += ok
    at_rest = visible & (spread < 0.5) & (seen >= 12)

    return {"visible": visible, "speed": speed, "vx": vx, "vy": vy,
            "outside": outside, "inside": inside, "at_rest": at_rest}


def nearest_by_team(bx, by, px, py, team):
    """Distance from the ball to the nearest tracked player of each side. NaN without a ball."""
    out = {}
    with np.errstate(invalid="ignore"):
        for side in (HOME, AWAY):
            columns = np.flatnonzero(team == side)
            distance = np.hypot(px[:, columns] - bx[:, None], py[:, columns] - by[:, None])
            distance = np.where(np.isnan(distance), np.inf, distance)
            nearest = distance.min(axis=1)
            nearest[np.isnan(bx)] = np.nan
            out[side] = nearest
    return out[HOME], out[AWAY]


# -- players ---------------------------------------------------------------------------------------

def _nan_stat(values, function):
    with np.errstate(invalid="ignore", all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return function(values, axis=1)


def player_velocities(px, py, step=5):
    """Per-slot velocity from positions over +/- ``step`` frames, and its magnitude.

    Speeds are derived rather than read from the file so that every match, whatever its source,
    is described the same way. Track slots change identity, so differences larger than a sprint
    are discarded rather than trusted.
    """
    dt = 2.0 * step / FPS
    vx = np.full(px.shape, np.nan, dtype=np.float32)
    vy = np.full(py.shape, np.nan, dtype=np.float32)
    vx[step:-step] = (px[2 * step:] - px[:-2 * step]) / dt
    vy[step:-step] = (py[2 * step:] - py[:-2 * step]) / dt
    with np.errstate(invalid="ignore"):
        speed = np.hypot(vx, vy)
        jump = speed > 12.0
    vx[jump] = np.nan
    vy[jump] = np.nan
    speed[jump] = np.nan
    return vx, vy, speed


def player_features(px, py, team, attack_sign):
    """Collective, identity-free descriptors of the two teams, one row per frame.

    ``attack_sign`` is +1 on frames where the home side attacks towards +x, -1 otherwise, so
    every longitudinal quantity can be expressed in the home side's attacking direction.

    Returns (matrix, names). Rows with no tracked players are NaN.
    """
    columns = {}

    present = ~np.isnan(px)
    vx, vy, speed = player_velocities(px, py)
    columns["n_players"] = present.sum(axis=1).astype(np.float64)
    columns["speed_mean"] = _nan_stat(speed, np.nanmean)
    columns["speed_median"] = _nan_stat(speed, np.nanmedian)
    columns["speed_p80"] = _nan_stat(speed, lambda v, axis: np.nanpercentile(v, 80, axis=axis))
    columns["speed_max"] = _nan_stat(speed, np.nanmax)
    with np.errstate(invalid="ignore"):
        count = np.maximum(columns["n_players"], 1.0)
        columns["frac_walk"] = np.nansum(speed < 1.5, axis=1) / count
        columns["frac_run"] = np.nansum(speed > 4.0, axis=1) / count

    # Players beyond the lines: a throw-in or corner taker stands outside the pitch, and nobody
    # does while the ball is in play.
    beyond = np.maximum.reduce([-px, px - PITCH[0], -py, py - PITCH[1]])
    with np.errstate(invalid="ignore"):
        columns["n_outside"] = np.nansum(beyond > 0.3, axis=1).astype(np.float64)
    columns["max_outside"] = np.maximum(_nan_stat(beyond, np.nanmax), -5.0)

    sign = attack_sign.astype(np.float64)
    for side, tag in ((HOME, "h"), (AWAY, "a")):
        cols = np.flatnonzero(team == side)
        x, y = px[:, cols].astype(np.float64), py[:, cols].astype(np.float64)
        cx = _nan_stat(x, np.nanmean)
        cy = _nan_stat(y, np.nanmean)
        columns[tag + "_cx"] = (cx - PITCH[0] / 2.0) * sign
        columns[tag + "_cy"] = cy - PITCH[1] / 2.0
        columns[tag + "_sx"] = _nan_stat(x, np.nanstd)
        columns[tag + "_sy"] = _nan_stat(y, np.nanstd)
        columns[tag + "_len"] = _nan_stat(x, np.nanmax) - _nan_stat(x, np.nanmin)
        columns[tag + "_wid"] = _nan_stat(y, np.nanmax) - _nan_stat(y, np.nanmin)
        columns[tag + "_vx"] = _nan_stat(vx[:, cols], np.nanmean) * sign
        columns[tag + "_vy_abs"] = np.abs(_nan_stat(vy[:, cols], np.nanmean))
        columns[tag + "_speed"] = _nan_stat(speed[:, cols], np.nanmean)
        columns[tag + "_speed_p80"] = _nan_stat(
            speed[:, cols], lambda v, axis: np.nanpercentile(v, 80, axis=axis))
        # Deepest and most advanced outfield lines in the home attacking direction.
        along = (x - PITCH[0] / 2.0) * sign[:, None]
        ordered = np.sort(np.where(np.isnan(along), np.inf, along), axis=1)
        columns[tag + "_back2"] = np.where(np.isinf(ordered[:, 1]), np.nan, ordered[:, 1])
        ordered = -np.sort(np.where(np.isnan(along), np.inf, -along), axis=1)
        columns[tag + "_front2"] = np.where(np.isinf(ordered[:, 1]), np.nan, ordered[:, 1])

    for name in ("cx", "sx", "sy", "len", "wid", "vx", "speed", "speed_p80"):
        columns["d_" + name] = columns["h_" + name] - columns["a_" + name]

    names = sorted(columns)
    matrix = np.column_stack([columns[name] for name in names])
    matrix[columns["n_players"] == 0] = np.nan
    return matrix, names


def with_context(matrix, names, halves=(12, 37, 75), lags=(50, 125)):
    """Add temporal context: moving averages over several windows and lagged / led differences.

    A frame on its own says little about a possession; a few seconds either side says much more.
    """
    blocks = [matrix]
    out_names = list(names)
    for half in halves:
        smooth = np.column_stack([box_mean(matrix[:, j], half) for j in range(matrix.shape[1])])
        blocks.append(smooth)
        out_names += ["{}_m{}".format(name, half) for name in names]
    base = np.column_stack([box_mean(matrix[:, j], 12) for j in range(matrix.shape[1])])
    for lag in lags:
        for direction, tag in ((-1, "past"), (1, "next")):
            moved = np.column_stack([shift(base[:, j], direction * lag)
                                     for j in range(base.shape[1])])
            blocks.append(moved - base)
            out_names += ["{}_{}{}".format(name, tag, lag) for name in names]
    return np.hstack(blocks).astype(np.float32), out_names


# -- who is on the ball ----------------------------------------------------------------------------

def _since_last(mask, cap):
    """Frames since ``mask`` was last True (0 on a True frame), capped."""
    index = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, index, -10 ** 9))
    return np.minimum(index - last, cap).astype(np.float64)


def _until_next(mask, cap):
    return _since_last(mask[::-1], cap)[::-1]


def _rolling_min(values, window, future):
    """Minimum over the ``window`` frames before (or after) each frame, itself included.
    NaN counts as missing; NaN where the whole window is missing."""
    from scipy.ndimage import minimum_filter1d
    filled = np.where(np.isnan(values), np.inf, values)
    origin = -(window // 2) if future else window // 2
    out = minimum_filter1d(filled, size=2 * (window // 2) + 1, origin=origin, mode="constant",
                           cval=np.inf)
    return np.where(np.isinf(out), np.nan, out)


def ball_touches(ball, threshold=4.0, span=3):
    """Frames where the ball's velocity changes abruptly: somebody played it.

    The change is measured across +/- ``span`` frames and a touch is a local maximum of at least
    ``threshold`` m/s. It survives slowly varying position error, which absolute distances to
    players do not. A tracker's filter spreads a touch over several frames, so the features use
    a short and a long span. Returns (frames, strength per frame).
    """
    from scipy.ndimage import maximum_filter1d
    vx, vy = ball["vx"], ball["vy"]
    change = np.hypot(shift(vx, span) - shift(vx, -span), shift(vy, span) - shift(vy, -span))
    strength = np.nan_to_num(change, nan=0.0)
    reach = span + 1
    peak = (strength >= threshold) & (strength == maximum_filter1d(strength, size=2 * reach + 1))
    # A plateau would mark several frames; keep the first of each run.
    peak &= ~shift(peak.astype(np.float64), -1, 0.0).astype(bool)
    return np.flatnonzero(peak), strength


def touch_points(ball, bx, by, touches, span):
    """Where the ball was actually played at each touch.

    A tracker's filter rounds off the corner the ball makes when it is redirected, so the tracked
    ball never quite reaches the player who played it. The straight paths before and after the
    touch are not rounded off: their crossing is where the ball changed direction. Falls back to
    the tracked position where the two paths are nearly parallel or the crossing is implausibly
    far away.
    """
    n = len(bx)
    lead = 2 * span
    x = bx[touches].astype(np.float64)
    y = by[touches].astype(np.float64)
    before = np.clip(touches - lead, 0, n - 1)
    after = np.clip(touches + lead, 0, n - 1)
    x1, y1 = bx[before].astype(np.float64), by[before].astype(np.float64)
    x2, y2 = bx[after].astype(np.float64), by[after].astype(np.float64)
    vx1, vy1 = ball["vx"][before], ball["vy"][before]
    vx2, vy2 = ball["vx"][after], ball["vy"][after]
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = vx1 * vy2 - vy1 * vx2
        # p1 + a v1 = p2 - b v2
        a = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / cross
        b = (vx1 * (y2 - y1) - vy1 * (x2 - x1)) / cross
        horizon = 1.5 * lead / FPS
        turning = np.abs(cross) > 0.2 * np.hypot(vx1, vy1) * np.hypot(vx2, vy2)
        ok = (np.isfinite(a) & np.isfinite(b) & turning
              & (a >= 0) & (a <= horizon) & (b >= 0) & (b <= horizon))
    x[ok] = x1[ok] + a[ok] * vx1[ok]
    y[ok] = y1[ok] + a[ok] * vy1[ok]
    return x, y


def owner_features(ball, d_home, d_away, match=None, contact=1.5, cap=250):
    """Ball-local evidence of who owns the ball, built to be mirrored.

    Returns (X, X_twin, names): ``X_twin`` is the same frames with the two sides swapped, so a
    model trained on both (with the label flipped for the twin) cannot favour a side.

    Per side: distance from the ball to its nearest player now and at several lags, the closest
    it got over the last and next 1, 2 and 5 seconds, how long since and until it was last on the
    ball (within ``contact`` metres), how long since it was on the ball *alone*, how long that
    last spell on the ball lasted, and how much nearer it was than the other side at the last two
    and next two touches of the ball. Shared: ball speed now and around, the timing and strength
    of those touches, and how long the ball has been and will stay tracked.
    """
    visible = ball["visible"]
    speed = np.where(visible, ball["speed"], np.nan)
    shared = {"speed": speed}
    for half in (6, 12, 25):
        shared["speed_m{}".format(half)] = box_mean(speed, half)
    smooth = box_mean(speed, 6)
    for lag in (12, 25):
        shared["speed_next{}".format(lag)] = shift(smooth, lag) - smooth
        shared["speed_past{}".format(lag)] = shift(smooth, -lag) - smooth
    shared["tracked_for"] = _since_last(~visible, cap)
    shared["tracked_until"] = _until_next(~visible, cap)

    def side(d, other):
        d = np.where(visible, d, np.nan)
        other = np.where(visible, other, np.nan)
        out = {}
        for lag in (-50, -25, -12, -6, 0, 6, 12, 25, 50):
            out["d{:+d}".format(lag)] = shift(d, lag)
        for window in (25, 50, 125):
            out["dmin_past{}".format(window)] = _rolling_min(d, window, future=False)
            out["dmin_next{}".format(window)] = _rolling_min(d, window, future=True)
        with np.errstate(invalid="ignore"):
            on = np.nan_to_num(d, nan=np.inf) < contact
            alone = on & (np.nan_to_num(other, nan=np.inf) > d + 1.0)
        out["since_on"] = _since_last(on, cap)
        out["until_on"] = _until_next(on, cap)
        out["since_alone"] = _since_last(alone, cap)
        out["until_alone"] = _until_next(alone, cap)
        # Length of the spell on the ball that is running or last ended, and of the next one.
        length = run_length_of(on).astype(np.float64)
        index = np.arange(len(on))
        last = np.maximum.accumulate(np.where(on, index, -1))
        out["last_spell"] = np.where(last >= 0, length[np.maximum(last, 0)], 0.0)
        nxt = np.minimum.accumulate(np.where(on, index, len(on))[::-1])[::-1]
        out["next_spell"] = np.where(nxt < len(on), length[np.minimum(nxt, len(on) - 1)], 0.0)
        return out

    home, away = side(d_home, d_away), side(d_away, d_home)

    # Touches: for the two before and the two after each frame, how long ago / ahead, how hard,
    # how close the nearest player was, and which side was nearer (the one that played it).
    n = len(visible)
    with np.errstate(invalid="ignore"):
        advantage = np.clip(np.where(visible, d_away - d_home, np.nan), -10.0, 10.0)
        closest = np.clip(np.where(visible, np.minimum(d_home, d_away), np.nan), 0.0, 10.0)
    # The advantage around a touch is steadier than at its single frame.
    advantage_near = box_mean(advantage, 3)
    for span in (3, 7):
        tag = "touch{}".format(span)
        touches, strength = ball_touches(ball, span=span)
        for lag in (-2 * span, -span, 0, span, 2 * span):
            shared["{}{:+d}".format(tag, lag)] = shift(strength, lag)
        if match is not None and len(touches):
            # Distances from the reconstructed point of contact, not from the rounded-off track.
            tx, ty = touch_points(ball, match["bx"], match["by"], touches, span)
            sides = []
            for side_code in (HOME, AWAY):
                cols = np.flatnonzero(match["team"] == side_code)
                with np.errstate(invalid="ignore"):
                    d = np.hypot(match["px"][touches][:, cols] - tx[:, None],
                                 match["py"][touches][:, cols] - ty[:, None])
                sides.append(np.where(np.isnan(d), np.inf, d).min(axis=1))
            with np.errstate(invalid="ignore"):
                point_adv = np.full(n, np.nan)
                point_adv[touches] = np.clip(sides[1] - sides[0], -10.0, 10.0)
                point_closest = np.full(n, np.nan)
                point_closest[touches] = np.clip(np.minimum(sides[0], sides[1]), 0.0, 10.0)
        else:
            point_adv = point_closest = None
        position = np.searchsorted(touches, np.arange(n), side="right") - 1
        for name, offset in (("last1", 0), ("last2", -1), ("next1", 1), ("next2", 2)):
            k = position + offset
            ok = (k >= 0) & (k < len(touches))
            frame = (touches[np.clip(k, 0, max(len(touches) - 1, 0))] if len(touches)
                     else np.zeros(n, int))
            gap = np.abs(np.arange(n) - frame)
            ok &= gap <= cap
            shared["{}_{}_gap".format(tag, name)] = np.where(ok, gap, cap).astype(np.float64)
            shared["{}_{}_strength".format(tag, name)] = np.where(ok, strength[frame], np.nan)
            shared["{}_{}_closest".format(tag, name)] = np.where(ok, closest[frame], np.nan)
            sources = [("adv", advantage), ("advnear", advantage_near)]
            if point_adv is not None:
                sources.append(("advpoint", point_adv))
                shared["{}_{}_closestpoint".format(tag, name)] = np.where(
                    ok, point_closest[frame], np.nan)
            for label, source in sources:
                adv = np.where(ok, source[frame], np.nan)
                home["{}_{}_{}".format(tag, name, label)] = adv
                away["{}_{}_{}".format(tag, name, label)] = -adv

    keys = sorted(home)
    shared_keys = sorted(shared)

    def stack(first, second):
        columns = [first[k] for k in keys] + [second[k] for k in keys]
        columns += [first[k] - second[k] for k in keys]
        columns += [shared[k] for k in shared_keys]
        return np.column_stack(columns).astype(np.float32)

    names = (["own_" + k for k in keys] + ["opp_" + k for k in keys]
             + ["diff_" + k for k in keys] + shared_keys)
    return stack(home, away), stack(away, home), names
