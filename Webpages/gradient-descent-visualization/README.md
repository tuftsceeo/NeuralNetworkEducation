# Gradient Descent Visualizer

An interactive, browser-based tool that teaches **gradient descent** by making every intermediate calculation visible. It trains a single linear neuron (`ŷ = w·x + b`) on a user-editable 2-point dataset, and walks the learner step-by-step through one full training loop: forward pass → error computation → gradient computation → parameter update → new network.

## Features

- **Step-by-step training loop** — Step (›), Run Epoch (››), Back a Step (‹), Back an Epoch (‹‹), Play/Pause (▶/❚❚) to auto-run epochs, and Reset (⟲), so learners can move forward and backward through the math at their own pace.
- **Live flowchart panel** — shows Error → Gradients (`dE/dw`, `dE/db`) → Updates (`Δw`, `Δb`) → new `w`/`b`, with each row progressively highlighting as it's computed. Can be hidden/shown.
- **Editable dataset** — two data points (`x`, `y` for each) can be edited directly, or randomized with one click, which also reinitializes `w`/`b`.
- **Error function selector** — choose Mean Squared Error, Mean Absolute Error, or write a **custom error expression** (e.g. `abs(pred - y)**1.5`), safely evaluated with a numeric-gradient fallback when no closed-form gradient is available.
- **Adjustable learning rate** input.
- **Live network diagram** — shows the current `x → w·x + b → ŷ` computation, with hover tooltips explaining each part, plus a per-point forward-pass breakdown for both data points.
- **Plots (via Plotly.js)**:
  - Loss vs. `w` and Loss vs. `b` — parabola/slice views with the current gradient direction annotated.
  - Network Prediction — the fitted line against the two data points.
  - Loss vs. epoch — training progress over time.
- **Animated feedback** — updated `w`/`b` values visually "fly" from the flowchart into the live network equation when an epoch completes.

## Files

| File | Purpose |
|---|---|
| `index.html` | Page structure/layout; loads Plotly.js and PyScript, then `main.py`. |
| `main.py` | Entry point (`import app`). |
| `app.py` | Controller — state machine for stepping/epochs, Play mode, undo/redo history, event wiring, animations. |
| `state.py` | `TrainingState` — holds dataset, `w`/`b`, epoch/step index, history. |
| `math_core.py` | Pure math — error functions and gradients, sandboxed custom-error evaluation, forward pass, update rule. |
| `dom.py` | DOM element lookups and Python→JS conversion helpers for Plotly. |
| `flowchart.py` | Builds and updates the flowchart/forward-pass diagram DOM. |
| `plots.py` | Plotly chart construction and updates (prediction, loss history, loss slices). |
| `styles.css` | Visual design system — palette, layout, node styling, animations, responsive layout. |
