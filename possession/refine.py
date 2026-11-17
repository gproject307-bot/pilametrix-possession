"""A sequence model that refines the question "which side has the ball" over time.

The frame models (transfer.py) judge every frame from hand-built summaries of its surroundings.
This model reads their output as a time series, together with a few raw signals (how much nearer
one side is to the ball, ball speed, ball touches and which side was nearer to the reconstructed
point of each touch). It is a small temporal convolutional network with dilated layers: 5 s of
context either side by default, about 40 s in the first submission.

It answers only the ownership question, on live frames. Whether the ball is in play is left to
the dead/alive evidence, because the sample matches never show a ball during a stoppage and a
sequence model would learn that as a shortcut.

Two properties hold by construction:

- swapping the two sides flips the sign of its output exactly, so it cannot favour a side;
- training weights each frame by the length of the possession it belongs to (see
  transfer.spell_weights), because the ranking metric counts possessions, not frames.
"""

import numpy as np

from . import features as F
from . import model
from . import transfer


def channels(match, sources, evidence):
    """(anti, sym): input channels that flip sign when the sides are swapped, and ones that do
    not. Each is (channels, frames), float32, roughly unit scale."""
    visible = evidence["visible"]
    ball = evidence["ball"]
    n = match["n_frames"]
    clip = 6.0
    with np.errstate(invalid="ignore"):
        advantage = np.where(visible, np.clip(evidence["d_away"] - evidence["d_home"], -10, 10),
                             0.0) / 5.0
        closest = np.where(visible, np.clip(np.minimum(evidence["d_home"], evidence["d_away"]),
                                            0, 10), 0.0) / 5.0
        speed = np.where(visible, ball["speed"], 0.0) / 10.0
    anti = [model._logit(sources["owner_transfer"], clip) / 3.0 * visible,
            model._logit(sources["all_transfer"], clip) / 3.0,
            model._logit(sources["team_transfer"], clip) / 3.0,
            np.nan_to_num(advantage)]
    sym = [model._logit(sources["dead_transfer"], clip) / 3.0, visible.astype(np.float64),
           np.nan_to_num(speed), np.nan_to_num(closest)]
    for span in (3, 7):
        touches, strength = F.ball_touches(ball, span=span)
        sym.append(np.clip(strength, 0, 30) / 10.0)
        point = np.zeros(n)
        mark = np.zeros(n)
        if len(touches):
            tx, ty = F.touch_points(ball, match["bx"], match["by"], touches, span)
            nearest = []
            for side in (0, 1):
                cols = np.flatnonzero(match["team"] == side)
                with np.errstate(invalid="ignore"):
                    d = np.hypot(match["px"][touches][:, cols] - tx[:, None],
                                 match["py"][touches][:, cols] - ty[:, None])
                nearest.append(np.where(np.isnan(d), np.inf, d).min(axis=1))
            with np.errstate(invalid="ignore"):
                point[touches] = np.nan_to_num(np.clip(nearest[1] - nearest[0], -10, 10) / 5.0)
            mark[touches] = 1.0
        anti.append(point)
        sym.append(mark)
    return np.stack(anti).astype(np.float32), np.stack(sym).astype(np.float32)


def sequences(match, sources, evidence, params=model.PARAMS):
    """One training sequence per half of a labelled match: inputs, target (1 = home) and
    per-frame weight (0 on dead frames)."""
    anti, sym = channels(match, sources, evidence)
    labels = match["labels"]
    weight = transfer.spell_weights(match, params["refine_spell_weight"])
    out = []
    for first, last in match["halves"]:
        window = slice(first, last + 1)
        alive = (labels[window] != model.D).astype(np.float32)
        out.append((anti[:, window], sym[:, window],
                    (labels[window] == model.H).astype(np.float32),
                    alive * weight[window].astype(np.float32)))
    return out


def dilations(layers, reach=None):
    """Doubling dilations 1, 2, 4, ...: ``layers`` of them, or, with ``reach`` (frames each side),
    as many as fit, plus one last layer that makes the reach exact (kernel 3 adds d each side)."""
    if not reach:
        return [2 ** i for i in range(layers)]
    out, total, d = [], 0, 1
    while total + d <= reach:
        out.append(d)
        total += d
        d *= 2
    if reach > total:
        out.append(reach - total)
    return out


def _network(c_in, width, layers, dropout, reach=None):
    import torch
    import torch.nn as nn
    steps = dilations(layers, reach)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(c_in, width, 1)
            self.conv = nn.ModuleList([nn.Conv1d(width, width, 3, padding=d, dilation=d)
                                       for d in steps])
            self.mix = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in steps])
            self.drop = nn.Dropout(dropout)
            self.out = nn.Conv1d(width, 1, 1)

        def forward(self, x):
            h = self.inp(x)
            for conv, mix in zip(self.conv, self.mix):
                h = h + self.drop(mix(torch.relu(conv(h))))
            return self.out(h).squeeze(1)

    return Network()


def _repeatable():
    """oneDNN's CPU convolutions are not bit-for-bit repeatable when the machine is busy, which
    would break the guarantee that a re-run reproduces the submission exactly. PyTorch's own
    implementation is, at the same speed for a network this size."""
    import torch
    torch.backends.mkldnn.enabled = False


def _odd(net, anti, sym):
    """A logit that changes sign exactly when the two sides are swapped."""
    import torch
    return 0.5 * (net(torch.cat([anti, sym], 1)) - net(torch.cat([-anti, sym], 1)))


def fit(sequence_list, params=model.PARAMS, log=None):
    """Train ``len(params["refine_seeds"])`` networks on the given sequences."""
    import torch
    import torch.nn.functional as functional

    say = log or (lambda *_: None)
    _repeatable()
    device = params["refine_device"]
    crop, batch = params["refine_crop"], params["refine_batch"]
    c_in = sequence_list[0][0].shape[0] + sequence_list[0][1].shape[0]
    frames = sum(s[0].shape[1] for s in sequence_list)
    steps = params["refine_epochs"] * max(1, frames // (crop * batch))
    tensors = [tuple(torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in s)
               for s in sequence_list]
    nets = []
    for seed in params["refine_seeds"]:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        net = _network(c_in, params["refine_width"], params["refine_layers"],
                       params["refine_dropout"], params.get("refine_reach")).to(device)
        optimiser = torch.optim.AdamW(net.parameters(), lr=params["refine_lr"], weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=params["refine_lr"],
                                                       total_steps=steps)
        net.train()
        for _ in range(steps):
            a_list, s_list, y_list, w_list = [], [], [], []
            for _ in range(batch):
                anti, sym, target, weight = tensors[rng.randint(len(tensors))]
                start = rng.randint(0, anti.shape[1] - crop)
                flip = rng.rand() < 0.5   # the same stretch seen from the other side
                a_list.append(-anti[:, start:start + crop] if flip else anti[:, start:start + crop])
                s_list.append(sym[:, start:start + crop])
                y = target[start:start + crop]
                y_list.append(1.0 - y if flip else y)
                w_list.append(weight[start:start + crop])
            weights = torch.stack(w_list)
            # One view per window is enough in training (each window is flipped at random);
            # the exact antisymmetry is imposed when the model is applied.
            logit = net(torch.cat([torch.stack(a_list), torch.stack(s_list)], 1))
            loss = (functional.binary_cross_entropy_with_logits(
                logit, torch.stack(y_list), reduction="none") * weights).sum() / weights.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
        net.eval()
        nets.append(net)
        say("sequence model {} of {} trained".format(len(nets), len(params["refine_seeds"])))
    return nets


def apply(nets, match, sources, evidence, params=model.PARAMS):
    """P(home in possession) for every frame, averaged over the trained networks."""
    import torch

    _repeatable()
    device = params["refine_device"]
    anti, sym = channels(match, sources, evidence)
    out = np.full(match["n_frames"], 0.5)
    with torch.no_grad():
        for first, last in match["halves"]:
            a = torch.from_numpy(np.ascontiguousarray(anti[:, first:last + 1])).to(device)[None]
            s = torch.from_numpy(np.ascontiguousarray(sym[:, first:last + 1])).to(device)[None]
            logit = sum(_odd(net, a, s)[0] for net in nets) / len(nets)
            out[first:last + 1] = torch.sigmoid(logit).cpu().numpy()
    return out
