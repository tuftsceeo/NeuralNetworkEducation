# Neural Network Trainer

A hands-on webpage for training a small neural network on data manually inputed or collected from LEGO CS + AI hardware. Users build the structure of their neural network, put in a data set, and train it via gradient descent while watching live loss and fit-quality plots.

## Features

- **Devices card** — connect/disconnect LEGO hardware over BLE; connected devices appear as removable chips.
- **Live network diagram** — add/remove Input values, hidden Layers (each with neurons sharing an activation function), and Outputs; zoom controls; SVG arrows connect nodes and can display live numeric values via a Debug toggle.
- **Activation functions per layer** — None, ReLU, Sigmoid, Tanh, Softplus, or Custom (piecewise expression editor), with a help popup.
- **Randomize weights** button to set new weights and biases for the network
- **Play/Stop** to continuously turn the harware input into hardware outputs via the neural network.
- **Training data collection** — a table of datapoint points; "Collect" samples connected hardware every 0.1s, "Clear" resets the dataset, rows are individually deletable, points can also be added manually, and a crop slider lets you visually exclude a time window of points from training without deleting them.
- **Training controls** — "Step" (train 1 epoch), "Train 50 epochs", an adjustable learning-rate input, and two live Plotly charts:
  - **Fit** — dataset points vs. live network output, color-coded per output.
  - **Loss** — loss vs. epoch.

## Files

| Path | Purpose |
|---|---|
| `index.html` | Page skeleton/styling/fonts; loads PyScript core, Plotly (CDN), and boots `main.py`. |
| `main.py` | Wires all event handlers, defines the Play/Stop loop, and boots the initial 1-input/1-layer/1-output network. |
| `Device.py` | LEGO hardware abstraction via the `legoeducation` package plus BLE glue and WASM/single-thread compatibility patches. |
| `glue/state.py` | Single shared mutable state module — topology, training data, zoom, plots. |
| `glue/sync.py` | Keeps the DOM in sync with state after topology changes. |
| `glue/bindings.py` | Event listener wiring for network items. |
| `network/network_model.py` | Topology lookups, forward propagation, and normalized-space gradient-descent training with weight/bias clipping. |
| `network/network_actions.py` | CRUD for inputs/neurons/layers/outputs (data + DOM). |
| `network/activations.py` | Activation math (ReLU/Sigmoid/Tanh/Softplus, custom expression evaluator) and derivatives for training. |
| `network/activation_editor.py` | UI logic for the activation selector and custom piecewise-function editor. |
| `misc_ui/templates.py` | HTML string builders for network diagram items. |
| `misc_ui/arrows.py` | SVG arrow/fan drawing between diagram nodes, with debug value labels. |
| `misc_ui/ui_chrome.py` | Zoom controls and resize-observer-based layout upkeep. |
| `misc_ui/dataset_ui.py` | Dataset table rendering, crop sliders, add/remove points, hardware data-collection loop. |
| `plots/plot_utils.py` | Shared Python→JS conversion helper for Plotly. |
| `plots/live_plot.py` | Scrolling per-input/output live sensor plots. |
| `plots/fit_plot.py` | Dataset vs. live network output scatter, per output. |
| `plots/loss_plot.py` | Loss-vs-epoch line chart. |
| `plots/scatter_plot.py` | Persistent test-point scatter plot. |
| `JS/ble.js` | Web Bluetooth wrapper class exposed to Python as a PyScript JS module. |
| `JS/plotly.js` | Stub exposing the CDN-loaded global `Plotly` as a PyScript JS module. |
| `styles.css` | Visual styling for cards, diagram, plots, and popovers. |

## Deployment

Hosted at [pyscript.com/@tuftsceeo/network-trainer](https://pyscript.com/@tuftsceeo/network-trainer). See the [repo-level PyScript deployment notes](../../README.md#how-the-webpages-apps-deploy-to-pyscriptcom) for how this actually syncs with GitHub — short version: every `.py`/`.js` file here is fetched live from `raw.githubusercontent.com`/jsdelivr on `main`, so pushing to `main` is enough to update the live app. The one exception is `pyscript.toml` itself: it lives only in the pyscript.com project's own storage and has to be hand-edited there whenever a file is added, removed, or renamed. The copy in this directory is a reference mirror, not the live one — keep it in sync by hand.
