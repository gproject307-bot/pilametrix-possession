"""Viterbi decoding of a small hidden-state sequence with per-frame scores."""

import numpy as np


def viterbi(scores, penalty, start=None):
    """Most likely state path.

    scores   (N, K) per-frame log-scores, higher is better
    penalty  (K, K) cost subtracted for moving from state i to state j (zeros on the diagonal)
    start    optional (K,) log-scores for the first frame
    """
    n, k = scores.shape
    back = np.zeros((n, k), dtype=np.int8)
    current = scores[0].astype(np.float64).copy()
    if start is not None:
        current += start
    for t in range(1, n):
        candidates = current[:, None] - penalty  # (from, to)
        best = candidates.argmax(axis=0)
        back[t] = best
        current = candidates[best, np.arange(k)] + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(current.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_switching(scores, swap_cost, other_cost):
    """Most likely path over home / away / dead where the cost of a home <-> away change varies by
    frame (``swap_cost[t]`` for a change that lands on frame t) and every change into or out of
    dead costs ``other_cost``."""
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    current = scores[0].astype(np.float64).copy()
    penalty = np.array([[0.0, 0.0, other_cost], [0.0, 0.0, other_cost],
                        [other_cost, other_cost, 0.0]])
    states = np.arange(3)
    for t in range(1, n):
        penalty[0, 1] = penalty[1, 0] = swap_cost[t]
        candidates = current[:, None] - penalty
        best = candidates.argmax(axis=0)
        back[t] = best
        current = candidates[best, states] + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(current.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
