# Technical document: team possession from automatic tracking

More than a Hack 2027, Phase 1. Team: Pilametrix.

## 1. Approach

Each graded frame gets one of `H`, `A`, `D`. We compute per-frame evidence from several models
and then decode each half as a sequence with Viterbi, so the output is a set of intervals and
not independent frame calls.

### 1.1 Per-frame evidence

**Ball rules** (ball tracked, 59% of graded frames)

- ball more than 0.5 m outside a pitch line: strong evidence of dead ball
- ball not moving for 2 s: moderate evidence of dead ball
- ball on the pitch and moving: mild evidence of live play
- no ball in the frame: mild evidence of dead ball

**Dead/alive model.** Gradient-boosted classifier on the players' collective behaviour: speed
statistics (mean, median, 80th percentile, share walking, share running), players standing
outside the pitch lines, team shapes. This model does most of the work on the 41% of graded
frames with no ball. Those frames come in about 150 long stretches (median 12 s). It only gets
player features, and nothing that tells it whether the ball is tracked (see 2.6).

**Ball model.** Classifier for which side has the ball, using features around the ball:

- distance to each side's nearest player, now and at lags up to 2 s
- closest distance each side got over the last and next 1, 2 and 5 s
- time since and until each side was on the ball
- ball touches: sudden changes in ball velocity, detected at two time scales. For the two
  touches before and the two after each frame we use how long ago, how hard, and which side
  was nearer.

"Nearer" is measured two ways: from the tracked ball and from a reconstructed contact point.
The tracker's filter smooths the corner in the ball's path when it is redirected, so the
tracked ball does not reach the player who played it. We take the straight paths before and
after the touch and use their intersection as the contact point.

**Facing.** The data has no body orientation. For the two players of each side nearest the
ball we estimate it (running direction when moving fast, towards the ball when standing) and
add how each of them moves relative to the ball: closing speed, moving with the ball, relative
speed. These features go into the ball model.

**Ball sensor.** Players react to the ball even when it is not tracked: the pressing side
converges on the carrier and the side in possession spreads out. Two gradient-boosted
regressors (x and y), trained on the true ball position in the sample matches, predict the
ball position from player features alone. On a held-out match the median error is about 7 m
(18 m if the players' centroid is used as the guess). On frames with no ball, the predicted
ball gives "pseudo-ball" features (each side's nearest player to it, players within 3 and
6 m). These go into the player model.

**Player model and combined model.** The same question (which side has the ball) answered
from the players' collective features: both teams' centroid, length, width, spread, back and
front lines, and longitudinal velocity in the home side's attacking direction, plus the
pseudo-ball features. The combined model uses ball and player features together.

**Sequence model.** A small temporal convolutional network (TCN) with seven dilated layers
and 5 s of context each side. Its inputs are the outputs of the three ownership models as
time series, plus which side is nearer the ball, ball speed, and the touches with the side
nearer each contact point. It only answers the ownership question. In training, frames are
weighted according to the length of their possession so that long possessions do not outvote
short ones, because spellF1 counts possessions and not frames.

Other details:

- Player and team features are also given as moving averages over 1, 3 and 6 s, and as
  differences against 2 s and 5 s earlier and later.
- There are no player identities in the data, so all player features are set statistics.
- All ownership models are symmetric between the two sides. The frame models are trained on
  both sides' views and the two predictions are averaged. The sequence model's output flips
  sign when the sides are swapped. So no model can learn that one club usually has the ball.

### 1.2 Decoding

Three-state Viterbi (home, away, dead), run per half.

- Team log-odds: weighted sum of the sequence model (1.0), the combined model (0.25) and the
  ball model (0.25), plus the player model where the ball is not tracked.
- Dead log-odds: the ball rules plus the weighted log-odds of the dead/alive model.
- The side favoured by the team evidence scores zero and the other side pays the difference.
  This keeps the confidence about which team has the ball out of the dead/alive decision.
- Changing state has a fixed cost.
- Stoppages shorter than 2 s are merged into the surrounding possession.
- Frames outside the two halves are `X`.

Two things are set by rule from the Smart Tagging kickoff tags instead of being predicted.
The tag is a window of a few seconds around the kick. We take the kick as the first frame in
that window where the ball is tracked within 2 m of the centre spot (the tracker usually
picks the ball up there), or the median offset of the other kickoffs if there is no such
frame. Each half is dead until its kickoff, and the 30 s before a kickoff that follows a goal
are dead.

### 1.3 Training

No model is trained on the challenge match. All models are trained on three other matches
with real labels (section 3) and applied to the challenge match unchanged.

## 2. Evaluation and model selection

The challenge provides five minutes of ground truth and does not allow tuning on it. We
compared methods on other matches where the full answer is known: the three public Metrica
Sports sample matches (corrected tracking plus event data). `benchmark.py` reproduces the
main table in 2.3.

### 2.1 Ground truth for the sample matches

Derived by code, the same way the challenge describes its own labels:

- dead where the corrected tracking has no ball (it is blank from the moment the ball goes
  out, a foul or a goal until the restart)
- otherwise the side that owns the ball, which changes at the first on-ball event of the
  other side (recovery, pass, carry, shot or set piece)

The three matches have 32-41% dead ball and 330-390 possessions each. The median possession
is 5 s.

### 2.2 Degrading the clean tracking

The sample tracking is much cleaner than the challenge tracking, so we degrade it before
training and scoring:

- ball track smoothed over 7 frames (like a tracker's filter)
- slowly drifting position offsets on the ball (0.7 m) and the players (0.3 m)
- ball dropped shortly before most stoppages, and for long stretches of open play, until 23%
  of ball-less frames are live
- each player missing 7% of the time

The three noise parameters were set by matching statistics that can be measured on the
challenge match without labels:

| statistic, live play | challenge match | sample, clean | first noise model | final noise model |
|---|---|---|---|---|
| detectable ball touches per minute | 13.9 | 30.4 | 37.5 | 14.8 |
| ball to nearest player at a touch, median (m) | 1.83 | 0.17 | 1.19 | 1.82 |
| same, slow ball (m) | 0.96 | 0.05 | 1.18 | 1.05 |
| same, fast ball (m) | 4.18 | 3.33 | 3.51 | 4.00 |
| ball acceleration, median (m/s^2) | 7.1 | 0.8 | 85 | 3.1 |

Our first noise model (0.8 m frame-to-frame jitter) matched the overall distance distribution
but made the ball about ten times too jerky and created false touches. Models trained on it
were learning from a ball track that does not look like the real one. The benchmark is harder
with the final noise model (the same method scores about 0.03 lower). All numbers below use
the final noise model.

### 2.3 Protocol and results

Leave-one-match-out: every model is trained on two matches and scored on the third with the
official metric code. The ball sensor is held out the same way, one level deeper: the sensor
applied to a training match was fitted on the other training match only. So no model is
trained on features from a sensor that has seen that match's ball.

| method | macroF1 | spellF1 | per match |
|---|---|---|---|
| reference baseline, vote over +/-25 frames | 0.765 | 0.238 | 0.256 0.242 0.217 |
| reference baseline, vote over +/-50 frames | 0.774 | 0.288 | 0.314 0.295 0.255 |
| reference baseline, vote over +/-125 frames | 0.736 | 0.279 | 0.281 0.294 0.262 |
| nearest-player rule + dead/alive model | 0.807 | 0.411 | 0.432 0.426 0.376 |
| same + player model on ball-less frames | 0.833 | 0.466 | 0.479 0.482 0.437 |
| ball model + player model on ball-less frames | 0.869 | 0.553 | 0.576 0.575 0.509 |
| combined model | 0.878 | 0.613 | 0.619 0.608 0.611 |
| ball model + combined model, no sequence model | 0.884 | 0.636 | 0.641 0.638 0.629 |
| sequence model alone | 0.893 | 0.690 | 0.715 0.685 0.669 |
| our first submission (no ball sensor, no facing, 40 s of context) | 0.886 | 0.653 | 0.652 0.665 0.641 |
| **sequence + ball + combined model (submitted)** | **0.888** | **0.674** | 0.689 0.674 0.658 |

Notes:

- The sequence-model rows were scored with networks trained on a GPU to save time. The
  submission's networks are trained on the CPU so that the run is reproducible to the bit
  (section 7).
- The first submission is `run.py --method first`.
- The baseline scores about the same on these matches as on the challenge match (published
  full-match scores for the same three windows: 0.200, 0.243, 0.224). This suggests the
  benchmark is about as hard as the real task.

### 2.4 Changes since the first submission

- ball sensor
- facing features
- sequence model context reduced from 40 s to 5 s. Window lengths from 5 to 45 s score within
  0.005 of each other, so we kept the shortest.

With the first submission's decoder unchanged, these raise the benchmark from 0.653 to 0.662.
The sequence model alone goes from 0.673 to 0.690. Since the sequence model got better, we
raised its weight in the decoder from 0.5 to 1.0, which gives 0.674. That is the only decoder
change.

### 2.5 Selection rule

A candidate replaces the earlier submission only if all of these hold:

1. the benchmark mean goes up
2. no held-out match gets worse
3. it does not split long possessions the way the sequence model alone does
4. it adds possessions at the same rate on the challenge match as on the benchmark (this
   check needs no labels)

We scored 60 decoder settings on the new evidence (switch cost 8 to 16, sequence weight 0.5
to 1.5, with and without the combined model and the nearest-player rule):

| decoder on the new evidence | spellF1 | per match | false splits per 10 min of 20 s+ possession | extra possessions per minute against the first submission: benchmark / challenge match |
|---|---|---|---|---|
| first submission, for reference | 0.653 | 0.652 0.665 0.641 | 0.3 | - |
| benchmark-tuned (sequence weight 1.5, switch cost 8, nearest-player rule 0.5) | 0.683 | 0.712 0.676 0.659 | 1.6 | +21% / +24% |
| sequence model alone | 0.690 | 0.715 0.685 0.669 | 1.3 | - |
| **submitted (sequence weight 1.0, switch cost 12)** | **0.674** | 0.689 0.674 0.658 | 0.6 | +8% / +10% |

The benchmark-tuned setting fails condition 3. It gets its score by cutting long possessions
about five times as often as the first submission, and that is the behaviour that over-split
the challenge match in earlier versions.

The unconstrained optimum (0.698) also lowers the dead/alive model's weight from 0.25 to
0.15. We kept 0.25, as in the first submission. The sample matches never show a ball during a
stoppage and the challenge match does, so the benchmark cannot measure what a low weight
costs there. On the challenge match the lower weight finds a restart near 87% of the tagged
set pieces, against 91% with 0.25.

The submitted setting is the best on the benchmark among those that pass all four conditions:

- mean 0.674 against 0.653
- every match better (+0.037, +0.009, +0.017)
- 5 false splits in 89 minutes of long possession, against 3
- extra possessions agree between the benchmark and the challenge match (+8% / +10%)

It changes 1.4% of the challenge match's graded frames compared to the first submission.

### 2.6 A leak we found and fixed

The pseudo-ball features are empty exactly on the frames where the ball is tracked. In an
earlier version they were also fed to the dead/alive model, and it learned "ball tracked
means live". That is true in the sample matches (the corrected tracking never shows a ball
during a stoppage) but not in the challenge match.

The benchmark did not catch it (it scored 0.68 to 0.71). A label-free check on the challenge
match did: the share of ball-tracked frames decoded as dead dropped from 20% to 6%. The
dead/alive model now gets player features only. We retrained the affected models, and all
numbers in this document are from after the fix.

### 2.7 Error analysis

Measured on the unconstrained optimum (0.698):

- with the true dead/alive mask it would score 0.77
- with the true side on every live frame it would score 0.88

So most of the loss is in who has the ball, mainly in short possessions. Recall at IoU 0.5 by
possession length:

| possession length | submitted | first submission |
|---|---|---|
| over 8 s | 0.87 | 0.86 |
| 4-8 s | 0.72 | 0.68 |
| 2-4 s | 0.44 | 0.38 |
| 1-2 s | 0.28 | 0.19 |
| under 1 s | 0.04 | 0.04 |

Possessions under 2 s are 23% of all possessions.

Why short possessions are hard:

- On the clean sample tracking, 93% of possession changes coincide with a detectable touch,
  and the nearer side at that touch is the right one 83% of the time.
- On tracking degraded to match the challenge data, 65% coincide with a detectable touch. The
  nearer side is right 70% of the time from the tracked ball and 76% from the reconstructed
  contact point.

For possessions under 2 s, a last-touch rule is then only slightly better than chance.

### 2.8 Tried and dropped

All of these were scored on the benchmark the same way, after the first submission unless
noted.

Decoders:

- semi-Markov (segment) decoder with a duration prior: 0.66-0.69 against 0.68 for Viterbi on
  the same evidence. The prior made it worse.
- fusion weights learned as a conditional random field: 0.47, it under-segments
- minimum-Bayes-risk decoding for the metric: no gain
- cap on stoppage length: worse. Real stoppages have a median of 13.5 s and one in ten is
  longer than 47 s.
- snapping possession changes to the nearest touch: +0.001
- shifting boundaries: no gain, they are already within one frame of the truth in the median

Sequence model:

- two and three stages, a smoothing loss, raw features as extra input channels, more seeds,
  more epochs, dropout, width, weight decay, input noise, ensembles of window lengths
- bidirectional GRU, TCN followed by a GRU, transformer encoder
- None gave a consistent gain over the single-stage TCN. Differences were within 0.01 and
  changed sign with the decoder setting. The recurrent and attention models were 0.006 to
  0.04 worse.

Dead/alive:

- a sequence model on the dead/alive output
- the ball sensor's predicted ball as extra features on every frame
- Both: AUC +0.001 to +0.004, no measurable gain in spellF1.

Features:

- the moment both teams reverse direction: +0.009 in a quick test, not on every match
- set-piece shapes: +0.011, on one match only
- a touch detector that fits the ball's velocity over 12 frames each side: finds fewer
  possession changes than ours on degraded tracking
- crediting a touch to the side the ball moves away from: the right side at a possession
  change goes from 72% to 73-74%. Too small to justify a retrain before the deadline.

From before the first submission:

- cheaper possession change near ball touches
- weighting frames by possession length in the frame models
- a wider or longer-trained sequence model

## 3. Training data

| | |
|---|---|
| dataset | Metrica Sports sample data, https://github.com/metrica-sports/sample-data |
| what | 3 anonymised matches: corrected 25 fps tracking of players and ball, and event data |
| amount | 429,923 frames (about 4 h 47 min), 1,072 possessions |
| who labelled | Metrica Sports produced the tracking and the events. We labelled nothing: the per-frame labels are derived from those files by the code in `possession/external.py`. The ball sensor's target is the ball position in the same corrected tracking |
| relation to the challenge match | none: other, anonymised games |

Each match is used in two noise realisations. We used no other dataset, no pre-trained model,
and no human-produced data about the challenge match (events, corrected tracking, possession
figures).

### 3.1 How the challenge match was used

Its automatic tracking was used without labels for two things: calibrating the noise model
(table in 2.2) and the label-free checks in 3.2.

Its public five-minute ground truth was used to check the pipeline only:

- confirming the frame alignment and the meaning of the labels
- reading how particular situations are labelled (a dead ball starts when the ball crosses
  the line, a restart is the ball being put back in play)
- finding bugs in an earlier version, which we fixed with general rules (the second half was
  decoded as live during the pre-kickoff warm-up; this is now handled with the kickoff tags)

We ran the scorer on the window for candidate versions during development. The possessions it
showed as missed (two under 1.5 s, a late change, an early dead-ball call) are what made us
look into touches, tracking noise and short possessions. The fixes were then built and
selected on the benchmark and on label-free checks of the challenge match (section 2). No
model was trained on the window and no parameter of the submitted method was chosen on it.

### 3.2 Transfer checks on the challenge match (no labels)

| check on the challenge match | value | on the benchmark |
|---|---|---|
| dead/alive model against ball-only anchors (ball off the pitch or at rest, against ball in open play), AUC | 0.95 | 0.93-0.97 against truth |
| ownership models agree with a ball-only nearest-player decode where it is settled | 0.92 (ball model), 0.93 (combined) | - |
| share of graded frames decoded as dead ball | 48.3% (21% where the ball is tracked, 88% where it is not) | 32-41% true |
| possessions / stoppages | 291 (mean 11.3 s, median 6.4 s) / 96 | 274 decoded per match, 357 true / 112 true |
| possessions per minute of live play | 5.3 | 4.8 decoded, 6.0 true |
| extra possessions per minute of play, against the first submission | +10% | +8% |
| home share of time in play | 62.0% | - |
| tagged set pieces with a decoded restart within -10 s / +20 s | 91% of 53; the tagged side has the ball after it in 75% of the 24 that carry a side | - |
| second-half kickoff | frame 113,742, the documented one | - |
| reproducibility | a second full run with no cached model outputs gave a byte-identical CSV | |

### 3.3 Public five-minute window

Pipeline check only. The window has 17 possessions, so it cannot rank methods.

| on the public window | macroF1 | spellF1 | possessions decoded (17 real) |
|---|---|---|---|
| reference baseline, vote over +/-50 frames | 0.621 | 0.326 | 26 |
| earlier self-contained method | 0.822 | 0.538 | 12 |
| our first submission | 0.906 | 0.691 | 16 |
| benchmark-tuned decoder on the new evidence (not submitted) | 0.905 | 0.611 | 19 |
| **submitted method** | 0.915 | 0.655 | 16 |

The submitted method matches 11 of the 17 possessions at IoU 0.5 and the first submission
matches 12. One possession moves spellF1 by about 0.04 here, and the scorer's bootstrap
interval for the first submission's 0.691 is 0.52 to 0.85. The window cannot separate these
rows and we did not use it to choose: the submitted method was fixed by the selection rule in
2.5 before this row was computed. The benchmark figure (0.67 on whole matches) is a better
estimate of what to expect.

## 4. Assets used and dropped

| asset | used | how |
|---|---|---|
| tactical ATD (players and ball) | yes | all features; noise calibration |
| tactical Smart Tagging | kickoffs only | forces dead ball before each kickoff |
| tactical homography | no | not needed, everything is done in pitch coordinates |
| tactical video | no | we did not attempt ball re-detection |
| broadcast video, ATD, homography | no | not synchronised to the tactical timeline |

Player speeds are computed from positions instead of read from the ATD speed channel, so that
all matches are processed the same way (the two agree with correlation 0.97).

An earlier self-contained method is still in the repository (`run.py --method self`): the
nearest-player rule plus dead/alive and team classifiers trained only on pseudo-labels from
this match's own ball-visible frames, with the Smart Tagging phase codes as extra features
and one round of self-training. It needs no other match. It cannot be scored on the
benchmark, because the sample matches never show a ball during a stoppage and so its
dead-ball anchors do not exist there. Its closest benchmark equivalent is the fifth row of
the table in 2.3. We replaced it because the learned ownership models are much better than
the nearest-player rule on every held-out match.

## 5. Possession rules implemented

- **Three states.** Every graded frame is home, away or dead.
- **Ownership** is what the models learned from the sample matches' event data: the side of
  the last on-ball action. A loose or contested ball stays with the side that last played it.
- **Change of possession.** The decoder pays a fixed cost to change side, equivalent to about
  four frames of confident contrary evidence, so one ambiguous frame does not flip it.
- **Dead ball.** Starts when the ball is tracked outside a pitch line, or when the players'
  collective behaviour says play has stopped (the main cue when the ball is not tracked, and
  the only one for fouls and offsides). It ends when that evidence turns.
- **After a stoppage** the side in possession is the one the evidence in the following frames
  supports.
- **Kickoffs.** Dead until the kick, located as described in 1.2.
- **No stoppage shorter than 2 s.**
- **No abstention inside the halves.** `X` is written only outside the graded blocks. Inside
  them, keeping the previous state is what the scorer would do with an `X` anyway, and it
  costs nothing on macroF1.

## 6. Limitations

- **Short possessions are mostly missed.** At this tracking quality most of them cannot be
  recovered (see 2.7). Even a perfect answer for who has the ball on every live frame would
  leave the benchmark below 0.9.
- **The benchmark noise is simulated.** It is calibrated on five label-free statistics of the
  challenge data, but real tracking errors (a ball in the air projected onto the ground, team
  mis-assignment, identity switches) are more than a filter plus drift. The transfer checks
  in 3.2 look fine but they are not proof. The leak in 2.6 is an example of something the
  benchmark alone did not catch.
- **The sample matches never show a ball during a stoppage and the challenge data does**, for
  about a fifth of dead time. So the benchmark cannot tune the ball rules. They were set by
  hand, and restart timing is less precise in that situation.
- **Stoppages where the players keep moving** (a quick throw-in) do not look dead from the
  players' behaviour. With the ball untracked there is no other signal for them.
- **Only three training matches**, all from one provider and from unknown competitions. All
  the sequence architectures we tried score within a few hundredths of each other, so we
  think the limit is the amount of labelled data and not the model.
- **The ball sensor is coarse** (7 m median error). It tells which group of players the ball
  is among, not who has it.
- **Smart Tagging is barely used**, because the training matches have none.

With more time we would:

- train on more labelled matches (public tracking datasets with per-frame possession exist
  and were not used here)
- re-detect the ball in the tactical video on the ball-less stretches and at touches, which
  is where the information for short possessions would have to come from
- credit touches to the side the ball moves away from
- align the broadcast tracking to the tactical timeline and use its ball
- simulate carried and out-of-play balls in the benchmark so the dead-ball rules can be tuned
  too

## 7. Reproduction

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/metrica-sports/sample-data
python run.py --data /path/to/data --samples sample-data/data --out out/Pilametrix.csv
```

Environment: Python 3.12.10, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, torch 2.13.0
(CPU), Windows 11 x86-64.

- Fixed seeds, CPU only.
- Run time is 70 to 100 minutes on a laptop. Most of it is training the frame models four
  times: once on all three sample matches, then once without each, so that the sequence model
  is trained on held-out predictions.
- PyTorch's oneDNN convolutions are switched off. They are not bit-for-bit repeatable when
  the machine is busy, and the contest requires a re-run to reproduce the file exactly.
- A second full run with no cached model outputs gave a byte-identical file (sha256
  `55fff164b19bf68efe8ba37c508ba146db1b133a4b7a250ceb05a750bd92360c`).
- `run.py --method first` reproduces our first submission (no ball sensor, no facing, 40 s of
  context) byte for byte (sha256
  `3bf5c3e83bb5c3c79614ffed7a01b36b0178c21f43617d88e52bd599ecf641c2`).

See the repository README for the data layout and for reproducing the benchmark.
