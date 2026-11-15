# Team possession from automatic tracking — technical document

More than a Hack 2027, Phase 1. Team: `<team_name>`.

## 1. Approach

The output is one of `H`, `A`, `D` per graded frame, decoded as a sequence, not frame by frame.

**Evidence, per frame.**

- *Ball rules* (where the ball is tracked, 59% of graded frames). A ball more than 0.5 m beyond a
  pitch line is strong evidence of a dead ball; a ball lying still for two seconds is moderate
  evidence of one; a ball tracked on the pitch and moving is mild evidence of play. A frame with
  no ball at all is mild evidence of a dead ball.
- *Dead/alive model.* A gradient-boosted classifier that reads the collective behaviour of the
  players: speed statistics (mean, median, 80th percentile, share walking, share running), players
  standing beyond the pitch lines, team shapes. It is the main witness on the 41% of graded frames
  with no ball, which come in about 150 long stretches (median 12 s).
- *Ball model.* A classifier for which side owns the ball, from the ball's neighbourhood: distance
  to each side's nearest player now and at lags up to 2 s, the closest each side got over the
  last and next 1, 2 and 5 s, how long since and until each side was on the ball, and the ball's
  *touches* (abrupt changes of its velocity, detected at two time scales): for the two touches
  before and the two after each frame, how long ago, how hard, and which side was nearer.
  "Nearer" is measured from the tracked ball and also from the **reconstructed point of
  contact**: the tracker's filter rounds off the corner the ball makes when it is redirected, so
  the tracked ball never quite reaches the player who played it, but the straight paths before
  and after the touch still cross where it was played.
- *Player model* and *combined model.* The same ownership question answered from the players'
  collective features alone (both teams' centroid, length, width, spread, back and front lines
  and longitudinal velocity in the home side's attacking direction), and from ball and player
  features together.
- *Sequence model.* A small temporal convolutional network (ten dilated layers, about 40 s of
  context either side) that reads the three ownership models' outputs as time series, together
  with which side is nearer the ball, the ball's speed, and the touches with the side nearer
  each reconstructed point of contact. It learns the shape of a possession as a whole, of which
  a frame model only ever sees one summary. It answers the ownership question only, and it is
  trained with each frame weighted by the length of its possession, because the ranking metric
  counts possessions, not frames.

Player and team features are also given as moving averages over 1, 3 and 6 s and as differences
against 2 s and 5 s earlier and later. There are no player identities in the data, so every
feature is a set statistic. Every ownership model is exactly symmetric between the sides (the
frame models are trained on both sides' views and average their two predictions; the sequence
model's output is built to change sign when the sides are swapped), so none can learn that one
club usually has the ball.

**Decode.** A three-state Viterbi pass (home, away, dead) per half. The team log-odds are a
weighted sum of the sequence model's (0.5), the combined model's (0.25) and the ball model's
(0.25) log-odds, plus the player model's where the ball is not tracked; the dead log-odds are the
ball rules plus the weighted log-odds of the dead/alive model. The side the team evidence
favours scores zero and the other pays, so confidence about *who* never leaks into *whether the
ball is in play*. Changing state has a fixed cost, which is what turns evidence into intervals.
Stoppages shorter than 2 s are absorbed into the surrounding possession. Frames outside the two
halves are `X`.

Two things are imposed rather than inferred, both from the automatic Smart Tagging kickoff tags.
The tag is a window of a few seconds around the kick; the kick itself is taken as the first frame
in that window where the ball is tracked within 2 m of the centre spot (the tracker usually
re-acquires the ball right there), or the median offset of the other kickoffs when it is not. A
half is dead until its kickoff, and the 30 s before an after-goal kickoff are dead.

**Where the models learn.** Not on the challenge match. They are trained on three *other*
matches with real labels (section 3), and applied to the challenge match unchanged.

## 2. How the method was chosen: a benchmark on other matches

The challenge gives five minutes of ground truth and forbids tuning on it, so methods were
compared on other matches where the whole answer is known: Metrica Sports' three public sample
matches (corrected tracking plus event data). `benchmark.py` reproduces everything in this
section.

*Ground truth* is derived mechanically, the way the challenge describes its own: dead where the
corrected tracking has no ball (it is blank exactly from the ball going out, a foul or a goal
until the restart), otherwise the side that owns the ball, which changes at the first on-ball
event of the other side (recovery, pass, carry, shot or set piece). The three matches come out
at 32–41% dead ball and 330–390 possessions each, with a median possession of 5 s.

*The clean tracking is roughened until it behaves like the automatic tracking.* The ball track
is smoothed over 7 frames (a tracker's filter), ball and players get slowly drifting position
offsets (0.7 m and 0.3 m), the ball is lost shortly before most stoppages and for long stretches
of open play until 23% of ball-less frames are live, and each player vanishes 7% of the time.
The three noise numbers were set by matching statistics that can be measured on the challenge
match without any label:

| statistic, live play | challenge match | sample, clean | first noise model | final noise model |
|---|---|---|---|---|
| detectable ball touches per minute | 13.9 | 30.4 | 37.5 | 14.8 |
| ball to nearest player at a touch, median (m) | 1.83 | 0.17 | 1.19 | 1.82 |
| … with a slow ball (m) | 0.96 | 0.05 | 1.18 | 1.05 |
| … with a fast ball (m) | 4.18 | 3.33 | 3.51 | 4.00 |
| ball acceleration, median (m/s²) | 7.1 | 0.8 | 85 | 3.1 |

The first noise model (frame-to-frame jitter of 0.8 m) matched the overall distance
distribution but made the ball ten times too jerky and invented touches, so models trained on it
learned from a ball track unlike the real one. The benchmark is harder under the final noise
model (the same method scores about 0.03 lower) and its numbers are the honest ones.

*Protocol.* Leave one match out: every model is trained on the other two matches and scored on
the third with the official metric code.

| method | macroF1 | spellF1 | per match |
|---|---|---|---|
| reference baseline, vote over ±25 frames | 0.765 | 0.238 | 0.256 0.242 0.217 |
| reference baseline, vote over ±50 frames | 0.774 | 0.288 | 0.314 0.295 0.255 |
| reference baseline, vote over ±125 frames | 0.736 | 0.279 | 0.281 0.294 0.262 |
| nearest-player rule + dead/alive model | 0.807 | 0.411 | 0.432 0.426 0.376 |
| … + player model on ball-less frames | 0.834 | 0.465 | 0.478 0.481 0.435 |
| ball model + player model on ball-less frames | 0.866 | 0.535 | 0.548 0.540 0.517 |
| combined model | 0.876 | 0.599 | 0.603 0.596 0.599 |
| ball model + combined model (previous submission) | 0.883 | 0.636 | 0.639 0.638 0.631 |
| sequence model alone | 0.891 | 0.673 | 0.681 0.687 0.651 |
| **sequence + ball + combined model (submitted)** | **0.886** | **0.653** | 0.652 0.665 0.641 |

The sequence-model rows were scored with networks trained on a GPU, for speed; the submission's
are trained on the CPU, which is what makes it reproducible to the bit (section 7).

The baseline behaves on these matches as it does on the challenge match (the published full-match
scores are 0.200, 0.243 and 0.224 for the same three windows), which is some evidence that the
benchmark is about as hard as the real thing.

*Tuning.* The fusion weights and switch costs were chosen by coordinate ascent on the mean
spellF1 of the three held-out matches, on a coarse grid, ignoring gains below 0.002. The result
is flat around the optimum. The one deliberate departure: the dead/alive model's weight is 0.25
where the search preferred 0.15 (a 0.004 difference), because the sample matches never show a
ball during a stoppage and the challenge match does, so the benchmark cannot see what a low
weight costs there; on the challenge match the lower weight found a restart near four fewer of
the 53 tagged set pieces.

*Where the benchmark loses points.* Given the true dead/alive mask the frame models would
score 0.70 instead of 0.64; given the true side on every live frame, 0.84. So most of the loss
is in who has the ball, and it is concentrated in short possessions: at IoU 0.5, recall is about
0.85 for possessions over 8 s, 0.66 for 4–8 s, 0.34 for 2–4 s, 0.16 for 1–2 s and 0.04 under
1 s. Possessions under 2 s are 23% of all possessions.

*Why short possessions are hard, measured.* On the clean sample tracking, 93% of changes of
possession coincide with a detectable touch and the right side is the nearer one at that touch
83% of the time. On tracking roughened to match the challenge data, 65% coincide with a
detectable touch and the nearer side is right 70% of the time from the tracked ball, 76% from
the reconstructed point of contact. For possessions under two seconds the last-touch rule is
then barely better than a coin, whatever is built on top.

*How far short possessions can be pushed.* The sequence model alone finds nearly twice as many
1-2 s possessions as the frame models (30% against 16% at IoU 0.5) and scores 0.673. It pays
with false splits, 2.6 opposite-side spells per ten minutes of 20 s+ possession against 0.1,
and the benchmark price of those is already in its score. What decided against it is a check
that needs no labels: relative to the frame models it adds 22% more possessions per minute of
play on the benchmark but 32% on the challenge match, so on this match a good part of its
extra possessions are not the short ones it finds elsewhere. The strongest blends showed the
same gap more mildly (+9% against +15%). The submitted mix is the benchmark-best setting whose
extra possessions agree between the benchmark and the challenge match: the sequence model at
half weight and a stricter change cost, +5% on both. It is better on all three benchmark
matches (0.653 against 0.636), finds a few more short possessions (25% of those under 4 s,
against 22%) and rarely splits long ones (0.3 per ten minutes). The finding stands: at this
tracking quality, possessions under two seconds mostly cannot be recovered.

*Ship rule, and the decision.* A candidate was to replace the earlier submission only if it
raised the benchmark mean, made no held-out match worse beyond noise, and did not split long
possessions the way the sequence model alone does. The submitted method met all three: mean
0.653 against 0.636; each match better (+0.013, +0.027, +0.010); 0.3 false splits per ten
minutes of 20 s+ possession, against 0.1 for the earlier submission and 2.6 for the sequence
model alone. It changes 1.7% of the challenge match's graded frames. The earlier submission is
`run.py --method frame`.

*Tried on the benchmark and dropped:* making a change of possession cheaper near ball touches
and dearer elsewhere (within 0.005 either way, for the frame models and the sequence model
alike, and it did not reduce false splits); weighting frames by possession length in the frame
models (+0.007, and not on every match); adding the nearest-player rule on top of the final mix
(no gain); a wider or longer-trained sequence model (no gain).

## 3. Training data

| | |
|---|---|
| dataset | Metrica Sports sample data, https://github.com/metrica-sports/sample-data |
| what | 3 anonymised matches: corrected 25 fps tracking of players and ball, and event data |
| amount | 429,923 frames (about 4 h 47 min), 1,072 possessions |
| who labelled | Metrica Sports produced the tracking and the events. We labelled nothing: the per-frame labels are derived from those files by the code in `possession/external.py` |
| relation to the challenge match | none: other, anonymised games |

Each match is used in two noise realisations. No other dataset, no pre-trained model, and no
human-produced data about the challenge match (events, corrected tracking, possession figures)
was used.

**What the challenge match itself contributed.** Its automatic tracking was used, without
labels, to calibrate the noise model (the table in section 2). Its **public five-minute ground
truth** was used to check the pipeline, and this is everything it was used for: confirming the
frame alignment and the meaning of the labels; reading how particular situations are labelled
(a dead ball starts when the ball crosses the line, a restart is the ball being put back in
play); and finding defects in an earlier version, fixed by general rules (the second half was
decoded as in play during the pre-kickoff warm-up, now handled with the kickoff tags). The
scorer was run on the window for candidate versions during development, and the particular
possessions it showed being missed (two under 1.5 s, a late change, an early dead-ball call)
prompted the investigation into touches, tracking noise and short possessions; the fixes were
then built and selected on the benchmark and on label-free checks of the challenge match
(section 2). No model was trained on the window and no parameter of the submitted method was
chosen on it.

**Does it transfer to the challenge match?** Checked without labels:

| check on the challenge match | value | on the benchmark |
|---|---|---|
| dead/alive model against ball-only anchors (ball off the pitch or at rest, against ball in open play), AUC | 0.95 | 0.93–0.97 against truth |
| ownership models agree with a ball-only nearest-player decode where it is settled | 0.93 | — |
| ball model and player-only model agree, ball-visible live frames | 0.86 | 0.86–0.91 |
| share of graded frames decoded as dead ball | 48.0% (20% where the ball is tracked, 88% where it is not) | 32–41% true |
| possessions / stoppages | 265 (mean 12.5 s, median 7.5 s) / 96 | 254 decoded per match, 357 true / 112 true |
| possessions per minute of live play | 4.8 | 4.4 decoded, 6.0 true |
| extra possessions per minute of play, against the previous submission | +5% | +5% |
| home share of time in play | 62.1% | — |
| tagged set pieces with a decoded restart within −10 s / +20 s | 91% of 53; the tagged side has the ball after it in 75% of the 24 that carry a side | — |
| second-half kickoff | frame 113,742, the documented one | — |
| reproducibility | two independent runs side by side, byte-identical CSV | |

**Public five-minute window** (pipeline check only; 17 possessions, so it cannot rank anything):

| on the public window | macroF1 | spellF1 | possessions decoded (17 real) |
|---|---|---|---|
| reference baseline, vote over ±50 frames | 0.621 | 0.326 | 26 |
| earlier self-contained method | 0.822 | 0.538 | 12 |
| frame models trained under the first noise model | 0.916 | 0.680 | 13 |
| previous submission (frame models) | 0.921 | 0.667 | 16 |
| **submitted method** | 0.906 | 0.691 | 16 |
| sequence model alone (not submitted) | 0.906 | 0.530 | 23 |

The scorer's bootstrap interval for the submitted spellF1 is 0.52 to 0.85, so the model rows
cannot be told apart here, and none of these choices was made on this window. The benchmark
figure (0.65 on whole matches) is the better guide to what to expect.

## 4. Assets used, and dropped

| asset | used | how |
|---|---|---|
| tactical ATD (players and ball) | yes | all features; noise calibration |
| tactical Smart Tagging | kickoffs only | forces dead ball before each kickoff |
| tactical homography | no | not needed: all reasoning is in pitch coordinates |
| tactical video | no | no ball re-detection was attempted |
| broadcast video, ATD, homography | no | not synchronised to the tactical timeline |

Player speeds are derived from positions rather than read from the ATD's speed channel, so that
every match is described the same way (the two agree with correlation 0.97).

*An earlier, self-contained method* is kept in the repository (`run.py --method self`): the
nearest-player rule, plus dead/alive and team classifiers taught only by pseudo-labels read off
this match's own ball-visible frames, with the Smart Tagging phase codes as extra features and
one round of self-training. It needs no other match. It cannot be scored on the benchmark (the
sample matches never show a ball during a stoppage, so its dead-ball anchors do not exist
there); its closest benchmark equivalent is the fifth row of the table. It was superseded
because the learned ownership models are far better than the nearest-player rule on every
held-out match.

## 5. Possession rules implemented

- **Three states only.** Every graded frame is home, away or dead.
- **Ownership is what the models learned from the sample matches' event data**: the side of the
  last on-ball action. A loose or contested ball stays with the side that last played it.
- **A change of possession needs sustained evidence.** The decode pays a fixed cost to change
  side, equivalent to about a quarter of a second of confident contrary evidence, so a single
  ambiguous frame cannot flip it.
- **Dead ball.** Starts when the ball is tracked beyond a pitch line, or when the players'
  collective behaviour says play has stopped (the main cue when the ball is not tracked, and
  the only one for fouls and offsides). It ends when that evidence turns.
- **After a stoppage** the side in possession is whichever the evidence in the following frames
  supports.
- **Kickoffs.** Dead until the kick, located as described in section 1.
- **No stoppage shorter than 2 s.**
- **No abstention inside the halves.** `X` is written only outside the graded blocks; inside
  them the decode's "keep the previous state" is what the scorer would do with an `X`, and it
  costs nothing on macroF1.

## 6. Limitations

- **Short possessions are mostly missed**, and at this tracking quality mostly cannot be
  found: see the measurement in section 2.
- **The benchmark's noise is simulated.** It is calibrated on five label-free statistics of the
  challenge data, but real tracking errors (a ball in the air projected onto the ground, team
  mis-assignment, identity switches) are not a filter plus drift. The transfer checks in
  section 3 are reassuring; they are not proof.
- **The sample matches never show a ball during a stoppage; the challenge data does**, about a
  fifth of dead time. The benchmark therefore cannot tune the ball rules, which were set by
  reasoning, and restart timing is softer in that situation.
- **Stoppages with the players still moving** (a quick throw-in) do not look dead in the
  players' behaviour, and with the ball untracked there is nothing else to see them with.
- **Only three training matches**, all from one provider and unknown competitions.
- **Smart Tagging is barely used**, because the training matches have none.

With more time: re-detect the ball in the tactical video on the ball-less stretches and at
touches, which is where the information for short possessions would have to come from; align the
broadcast tracking to the tactical timeline and use its ball; simulate carried and out-of-play
balls in the benchmark so the dead-ball rules can be tuned too.

## 7. Reproduction

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/metrica-sports/sample-data
python run.py --data /path/to/data --samples sample-data/data --out out/<team_name>.csv
```

Python 3.12.10, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, torch 2.13.0 (CPU), Windows 11
x86-64. Fixed seeds, CPU only; about 75 minutes on a laptop, most of it training the frame
models four times (once on all three sample matches, then once without each, to give the
sequence model honest inputs to learn from). PyTorch's oneDNN convolutions are switched off:
they are not bit-for-bit repeatable when the machine is busy, and the contest requires a re-run
to reproduce the file exactly. `run.py --method frame` reproduces the previous submission, without
the sequence model, in about 10 minutes. See the repository README for the data layout and for
reproducing the benchmark.
