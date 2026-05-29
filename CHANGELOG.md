# Changelog

All notable changes to mlgidLAB are recorded here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); `aN` suffixes are alpha
pre-releases.

## 0.1.0a1 — first alpha (2026-05-29)

First public alpha of **mlgidLAB**, a desktop GUI for the
[mlgidBASE](https://github.com/mlgid-project/mlgidBASE) GIWAXS analysis
pipeline. It wraps the full `pygid → mlgidDETECT → pygidFIT → mlgidMATCH`
workflow — raw detector frames can be converted to NeXus, then detected,
fitted, matched, reviewed, and edited, all in one PySide6 window. The
upstream algorithms are unchanged; mlgidLAB adds visual control.

This is an **alpha** for evaluation inside the research group: the core
detect → fit → match → edit loop works end-to-end, but expect rough
edges. Please report issues (see *Known limitations* below).

### Highlights

**View & navigate**
- Open NeXus or raw HDF5 by content auto-detection (File → Open,
  drag-and-drop, or Open recent). Multiple files open at once.
- Cartesian (q_xy / q_z) ↔ polar image toggle; raw files show the
  detector image in pixels. Frame slider + Play, colormap, histogram
  levels, log/linear contrast, live cursor readout.
- Lazy frame loading with bounded per-frame LRU caches and a
  background prefetcher, so multi-GB stacks open on a laptop.

**Convert (raw → NeXus)**
- Multi-file batch conversion; GID / Transmission geometry and all
  four pygid conversion types.
- PONI + mask + angle-of-incidence inputs, with an embedded pyFAI
  calibration dialog (Create… buttons) seeded from the active scan.
- Separate-files or separate-datasets output, append-vs-overwrite.

**Detect / fit / match**
- Run Detection, Fitting, Matching (or the full pipeline) from the
  Pipeline dock; every mlgidBASE parameter is exposed.
- Matching from a preprocessed CIF pickle, raw `.cif` files, or a CIF
  folder; per-entry experimental parameters from instrument metadata.
- Run scopes: active entry / all entries / active frame / all frames.

**Edit peaks**
- Manual peaks via `Ctrl+Alt`-drag; commit with Add-to-detected or
  Add-to-fitted (2D pygidfit, or 1D scipy fallback) with a live cyan
  preview box and on-image ROI resize.
- **Multi-select** detected *or* fitted peaks with `Ctrl+click`
  (kind-aware) and `Ctrl+A` (all of the current kind on the frame).
- **Copy/paste** detected peaks with `Ctrl+C` / `Ctrl+V`, and
  **paste to a frame range** (`Ctrl+Shift+V`, e.g. `0-34,37`).
- **Batch 2D fit** ("Fit selected (2D)") over every selected detected
  peak, with a cancellable progress dialog.
- **Bulk delete** a multi-selection (`Delete`); single and bulk delete
  are both **undoable** (`Ctrl+Z` / `Ctrl+Shift+Z`), including the
  fitted 2D-shape parameters.

**Inspect & export**
- Peaks dock: sortable Detected / Fitted / Matched tables with
  bidirectional click-sync to the image; Display dock overlay toggles
  + score / probability / CIF-name filters.
- Profiles dock: live radial + angular Gaussian fits of the selected
  peak.
- Tools → Export figure… (matplotlib via mlgidbase), Export peaks as
  CSV…, Clear / Reset peaks at frame / entry / all scopes.
- Help → Controls & shortcuts (F1), About, Copy diagnostics.

### Install

Requires **Python ≥ 3.11** (Linux, macOS, Windows).

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a1"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a1"

mlgidlab        # launch
```

The `[pipeline]` extra pins the verified-good backend set:
`mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`. The GUI runs without them in
view-only mode (Run buttons disabled). See
[`docs/getting_started.md`](docs/getting_started.md) for per-OS install
detail and a first-file walkthrough, and
[`docs/backend_compatibility.md`](docs/backend_compatibility.md) for the
backend version policy.

### Known limitations

- **Alpha**: APIs, file layout, and UI may change before a stable
  release. Back up data before editing.
- The `[pipeline]` extra installs the heavy ML stack (torch + the
  in-house mlgid packages); first install is large.
- `mlgidMATCH` is GPL-3.0 while mlgidlab is MIT — it is an optional,
  separately-installed dependency the GUI only calls (aggregation, not
  a derived work).
- Some error paths in the calibration dialog and background workers
  log rather than surface to the UI; if an operation seems to do
  nothing, check the Logs dock.
- mlgidlab's fitted-peak add/delete touches `fitted_peaks` only, not
  the paired `fitted_peaks_errors`; re-run Matching after editing
  fitted peaks (matched indices reference the fitted-row ordering).

### License & contact

MIT — see [`LICENSE`](LICENSE). Maintainer: Nico Lerch
(<nico.lerch@uni-tuebingen.de>); issues via
https://github.com/mlgid-project/mlgidLAB/issues.
