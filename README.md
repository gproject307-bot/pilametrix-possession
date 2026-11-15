# Team possession from automatic tracking

Entry for **More than a Hack 2027, Phase 1: Ball Possession**
([challenge repository](https://github.com/metrica-sports/more-than-a-hack-2027)).

For every frame of the tactical video it outputs `H` (home in possession), `A` (away in
possession), `D` (dead ball) or `X` (outside the two halves).

The method and how it was chosen are described in
[docs/technical_document.md](docs/technical_document.md). In short: gradient-boosted models read
the ball's neighbourhood and the collective behaviour of the 22 players and say, per frame, whether
the ball is in play and which side has it; a small temporal convolutional network refines the
"which side" answer over time; a three-state Viterbi decode turns that evidence into possessions. The models are trained on **other matches** with real labels (Metrica Sports' public
sample data), roughened until they behave like the automatic tracking. Several methods were
compared on those matches with the official scorer, and the chosen one was applied to the
challenge match unchanged.

## Reproduce the submission

Requires Python 3.12, the challenge data bundle, and the public sample data. Neither dataset is
included here.

```bash
pip install -r requirements.txt
```

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data
```

```bash
python run.py --data /path/to/data --samples sample-data/data --out out/<team_name>.csv
```

`--data` is the folder that contains `tactical/`, laid out as in the challenge repository's
`DATA.md`. Only two of its files are read:

```
tactical/BAR-ATH_tactical_atd.txt   (+ BAR-ATH_tactical_atd.xml)
tactical/BAR-ATH_tactical_smart_tagging.xml
```

The run takes 75 to 100 minutes on a laptop CPU and is deterministic: fixed seeds, pinned
dependencies, everything on the CPU. `--method frame` leaves out the sequence model (the previous
submission, about 10 minutes). Validate the result with the challenge repository's own checker:

```bash
python /path/to/more-than-a-hack-2027/evaluation/validate_submission.py out/<team_name>.csv
```

## Reproduce the comparison of methods

```bash
python benchmark.py --samples sample-data/data --scorer /path/to/more-than-a-hack-2027/evaluation --tune
```

Scores every method on the three sample matches (each model is trained on the other two) with
the official metric code, then searches the fusion weights and switch costs. This is where the
parameters in `possession/model.py` come from; the challenge match plays no part in it. Add
`--device cuda` to train the sequence model on a GPU there (faster, not bit-for-bit repeatable).

## Watch a replay

A 2D replay in the browser: the players, the ball, our possession call and, where it is known,
the truth, with a strip of what the models said and a list of every stretch where our call
differs from the truth.

```bash
python run.py --data /path/to/data --samples sample-data/data --evidence out/evidence.npz
```

```bash
python replay.py --data /path/to/data --labels out/submission.csv --truth /path/to/more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv --evidence out/evidence.npz --open
```

`--open` serves the replays on `http://127.0.0.1:8765` and opens the browser; `--serve` reopens
what was already built. `python replay.py --sample 1 --samples sample-data/data` builds the same
view for a public sample match, where the truth covers the whole match (run `benchmark.py` first
for its predictions). The address keeps the moment you are looking at, for example
`#frame=12345&window=30&theme=dark`. Everything stays on your computer: what the replay writes
for the challenge match is match data and must not be uploaded or shared.

## Layout

```
run.py                     produce the submission
benchmark.py               compare and tune methods on other matches with known answers
replay.py                  build and serve a replay of a match (viewer/index.html is the page)
possession/io.py           read the challenge tracking and Smart Tagging, write the CSV
possession/external.py     read the public sample matches, derive their labels, roughen them
possession/features.py     ball kinematics and touches, ball-to-player distances, team features
possession/model.py        ball evidence, classifiers, fusion and decode
possession/transfer.py     train on other matches, apply to the match being decoded
possession/refine.py       the sequence model that refines who has the ball
possession/hmm.py          Viterbi
possession/diagnostics.py  label-free checks (run.py --diagnostics)
tests/                     unit tests that need no match data
docs/                      technical document
```

```bash
python -m unittest discover -s tests
```

## Data and rules

- No match data is in this repository; `data/`, `cache/` and `out/` are git-ignored.
- No hand labels were made by us, on any match. Training labels come from the event data and
  corrected tracking that Metrica Sports published for three other, anonymised matches
  (https://github.com/metrica-sports/sample-data).
- Nothing about the challenge match other than the provided automatic data was used. Its public
  five-minute ground truth was used only to check that the pipeline is wired correctly, not to
  train or to choose parameters.
