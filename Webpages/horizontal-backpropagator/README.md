# Horizontal Backpropagator

An interactive, browser-based tool that teaches **backpropagation in a multi-layer feed-forward network**. Unlike the companion [gradient-descent-visualization](../gradient-descent-visualization) tool (which trains a single neuron), this page lets learners build a chain of single-neuron layers (`x → n1 → a1 → n2 → a2 → ... → ŷ → L`) drawn left-to-right, and watch backpropagation animate one node at a time — a "reveal" of the chain-rule product at each weight node and each activation node, walking right to left from the loss back to the input.

## Features

- **Live horizontal network diagram** — neuron boxes, activation boxes, weight badges (rendered as algebraic equations, e.g. `5.00x + 1.20`), and a loss node, connected by arrows.
- **Animated backward pass** — gradient arrows and chain-rule formula labels accumulate step by step as backprop walks from the loss back to the input, with node highlighting/pulsing to show what's currently active.
- **Editable architecture**:
  - "+ Layer" button to add a neuron, each with its own activation function dropdown (None, ReLU, Leaky ReLU, Sigmoid, Tanh, Softplus).
  - Per-layer "×" button to remove a layer.
  - "Biases" toggle to include/exclude bias terms.
  - "🎲 Randomize weights" button.
  - "?" help button opens a popover reference explaining each activation function.
- **Adjustable learning rate** input.
- **Step-by-step training controls** — Back Epoch (‹‹), Back Step (‹), Step (›), Run Epoch (››), Reset (⟲), and Play/Pause (▶/❚❚) for auto-running epochs, with full undo/redo via snapshots.
- **Editable training dataset** — add, edit, or remove `(x, y)` points in a live table.
- **Plots (via Plotly.js)**:
  - Current fit — data scatter plus the network's prediction curve.
  - Loss vs. epoch.

## Files

| File | Purpose |
|---|---|
| `index.html` | Page structure/layout; loads Plotly.js and PyScript, then `main.py`. |
| `main.py` | Boots the app — seeds a default layer/dataset and wires all event listeners. |
| `state.py` | Shared mutable module state: layers, dataset, epoch/step index, learning rate, backward-pass plan, undo history. |
| `activations.py` | Pure math — activation functions and their derivatives. |
| `network_model.py` | Topology (add/remove layer), forward pass, builds the batch-averaged backward "reveal plan", applies weight updates, snapshot/restore for stepping back. |
| `training.py` | Step state machine — forward pass, loss→ŷ boundary, per-layer reveal substeps, epoch completion; also turbo epoch execution for Play mode and backward step/epoch via snapshot replay. |
| `diagram_render.py` | Builds/updates the diagram DOM — weight badges, delta arrows, gradient arrows and formula labels, node highlighting, activation-help popover. |
| `dataset_ui.py` | Renders and binds the editable data table. |
| `plots.py` | Plotly wrappers for the fit curve and loss trace. |
| `ui_refresh.py` | Aggregator functions that re-render dependent views on topology/weight/dataset changes (kept separate from `main.py` to avoid PyScript re-running `main.py` on a stray import). |
| `styles.css` | Card-based light UI theme, loading splash, one-screen layout with scroll fallback. |