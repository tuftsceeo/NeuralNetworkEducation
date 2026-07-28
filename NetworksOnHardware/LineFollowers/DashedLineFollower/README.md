# Dashed Line Follower

An line edge-following LEGO robot ( color sensor + double mtoro)
that stays on a line even when the line is dashed (has gaps in it), by
training an LSTM to hold a memory of "true" line position
through the gaps instead of reacting to them as if they were real drifts.

## The problem

The robot follows the *edge* of the line, not the center: the sensor reads
`0` (fully dark) when it's drifted onto the line, `100` (fully light) when
it's drifted off onto the background, and ~`50` when it's straddling the
edge correctly. A simple proportional controller (see `baselinefollower.py`)
maps this reading directly to differential motor speeds and works fine on a
solid line.

The trouble is a gap in the tape reads identically to "drifted far off
onto the background" — both saturate the sensor at `100`. A purely reactive
controller can't tell "there's a gap here" apart from "I've veered way off
the line," so it swerves at every dash. An LSTM can tell the difference,
because its hidden/cell state remembers whether it was gently approaching
the edge or genuinely veering off right before the saturated reading hit.

## Files

- **`baselinefollower.py`** — simple proportional edge-follower (no network). Was just used to show the alternative.
- **`datacollection.py`** — runs the same proportional controller as
  `baselinefollower.py` on a solid line, but also logs `(timestep, sensor
  reading, motor speeds)` for every step and dumps it to the `line_run_1`
  JSON file. This recorded run is the training data for both LSTM
  scripts below.
- **`line_run_1`** — recorded sensor + motor data from a real run over a
  solid line (no gaps). Used as ground truth: gaps are synthetically
  stamped into a copy of this data (`make_dashed`) to simulate what the same
  run would look like over a dashed line, while the original readings serve
  as the training target.
- **`DashedLineFollower_LSTM_1.py`** — trains the LSTM to predict the *true
  (un-gapped) sensor reading* at each timestep, then converts that predicted
  reading into motor speeds using the same affine mapping as the baseline
  controller.
- **`DashedLineFollower_LSTM_2.py`** — trains the LSTM directly against the
  recorded motor speeds (two output heads, one per motor) instead of an
  intermediate sensor estimate, so it learns the gap-robust control policy
  end-to-end.

Both LSTM scripts implement a **from-scratch, hidden-size-1 scalar LSTM**
(`ScalarLSTMCell`) with hand-written forward gates and full truncated BPTT,
so every weight and gradient is a single traceable number — meant as a
teaching tool for how gates and backprop-through-time actually work, not as
a production model.

## Usage

1. Run `datacollection.py` with a bot following a line to get data (or just use line_run_1)
2. Run `DashedLineFollower_LSTM_1.py` or `DashedLineFollower_LSTM_2.py` to:
   - synthetically dash the recorded run (`make_dashed`),
   - train the scalar LSTM via BPTT to bridge the gaps,
   - then drive the real robot live using the trained cell's persistent
     hidden/cell state, which carries the line-position estimate through
     each dash instead of reacting to the saturated reading.

Update the card_serial values as needed to connect hardware

## Something to watch for

During training/inference, watch the forget gate `f_t` during a gap: values
pulled toward `1` mean the cell state (and the memory of the true, solid-line
reading) survives the saturated input instead of being overwritten by it —
this is the mechanism that lets the robot coast through a dash rather than
swerving at it.

## Other Notes

This bot does poorly with long dashes and sharp turns, as it has no way to differentiate a hard turn from a dash. Also, it does much better with turns away from the line than towards the line, as it knows it should always turn when it sees the line, but it does not always know to turn when it sees white.
