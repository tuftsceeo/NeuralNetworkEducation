# NeuralNetworkEducation

Materials for teaching neural networks with LEGO CS+AI hardware.

- `Activities/` — lesson materials and hackathon activities.
- `NetworksOnHardware/` — Python examples that run neural networks directly on SPIKE hubs.
- `Webpages/` — browser-based PyScript apps (network builder/trainer, visualizations) that connect to hardware over Web Bluetooth. Each is hosted as its own project on pyscript.com.

## How the `Webpages/*` apps deploy to pyscript.com

Each app is hosted at `pyscript.com/@tuftsceeo/<app-name>` (e.g. [network-trainer](https://pyscript.com/@tuftsceeo/network-trainer)). **pyscript.com does not sync with GitHub automatically.** Only two files actually live in a pyscript.com project's own storage:

- `index.html`
- `pyscript.toml`

Every other file — every `.py` module and the JS modules — is listed inside `pyscript.toml` as a **URL**, not a local path, and gets fetched fresh over HTTP each time the page loads:

- Python files, via `[files]`: `https://raw.githubusercontent.com/tuftsceeo/NeuralNetworkEducation/refs/heads/main/Webpages/<app>/...`
- JS modules, via `[js_modules.main]`: `https://cdn.jsdelivr.net/gh/tuftsceeo/NeuralNetworkEducation@main/Webpages/<app>/...`

So for any `.py`/`.js` change: **commit and push to `main`, and the live app picks it up on next load** — no copy-pasting into the pyscript.com editor needed.

Two things this does *not* cover:

1. **`pyscript.toml` itself is not fetched from GitHub.** It can't be — PyScript needs to read it before it knows what else to fetch, so it has to live in the pyscript.com project's own storage. If you add, remove, or rename a file under `Webpages/<app>/`, you must open that project on pyscript.com and hand-edit its `pyscript.toml` to add/remove the corresponding line. The copy of `pyscript.toml` committed in this repo (e.g. `Webpages/network-trainer/pyscript.toml`) is a **mirror for reference**, not the live one — keep it in sync by hand whenever the live one changes, or it'll drift out of date (as `network-trainer`'s had, before being corrected).
2. **Caching lag.** `raw.githubusercontent.com` sits behind a short-lived CDN cache (~5 min), so `.py` changes usually show up within a few minutes of a push — hard-refresh the pyscript.com page to be sure you're not looking at a stale copy. jsdelivr's `@main` branch alias caches much longer (their docs cite up to ~24h unless purged) — if a JS change doesn't show up promptly, that's why.

`index.html` and `styles.css` are otherwise pushed/pulled the same way, by hand, since they aren't listed in `[files]`/`[js_modules.main]` and aren't fetched from GitHub either — `index.html` lives directly in the pyscript.com project (it's the one that references `pyscript.toml` and `main.py`), and `styles.css` is linked from it as a plain relative path, so it also has to be uploaded to the pyscript.com project directly rather than fetched from GitHub.

`horizontal-backpropagator`, `gradient-descent-visualization`, and `network-builder` use the same URL-fetch pattern in their committed `pyscript.toml` files, but those still point at the repo's old name/layout (`tuftsceeo/spike-neural-nets`, `TeachingNNs/...`) — worth checking whether their *live* pyscript.com projects were manually repointed the way `network-trainer`'s was, next time one of them needs a change.
