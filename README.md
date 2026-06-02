# mlgidLAB

A desktop GUI for the
[mlgidBASE](https://github.com/mlgid-project/mlgidBASE) GIWAXS analysis
pipeline. It drives the full
`pygid → mlgidDETECT → pygidFIT → mlgidMATCH` workflow in one PySide6
window: convert raw detector images to NeXus, then **detect / fit /
match**, **review and edit** peaks, and **export** — same algorithms,
visual control.

> **Alpha (`v0.1.0a2`).** Pre-release for evaluation in the research
> group. The detect → fit → match → edit loop works end-to-end; expect
> rough edges and report issues. See [`CHANGELOG.md`](CHANGELOG.md) for
> highlights.

<!-- TODO: add a screenshot here, e.g. docs/img/overview.png -->

## Install & launch

Requires **Python ≥ 3.11** (Linux / macOS / Windows). A fresh conda
environment is recommended — the `[pipeline]` extra pulls a large ML
stack (PyTorch + the in-house mlgid packages), so keep it isolated.

```bash
# create + activate an isolated environment
conda create -n mlgidlab python=3.12 -y
conda activate mlgidlab

# full pipeline (detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a2"
mlgidlab
```

No conda? Any Python ≥ 3.11 virtual environment works
(`python -m venv mlgidlab && source mlgidlab/bin/activate`).

Drop `[pipeline]` for view-only mode (browse + edit existing NeXus
results, in-GUI pyFAI calibration; Run buttons disabled). Full per-OS
install, a first-file walkthrough, and shortcuts are in
**[docs/getting_started.md](docs/getting_started.md)**.

## What you can do

- **Open** NeXus or raw HDF5 by auto-detection; multiple files at once.
- **View** in Cartesian (q_xy / q_z) or polar; frame playback, colormap,
  log/linear contrast, live cursor readout; multi-GB stacks load lazily.
- **Convert** raw → NeXus (batch), with an embedded pyFAI calibration
  dialog for PONI + mask.
- **Detect / fit / match** from the Pipeline dock (every mlgidBASE
  parameter exposed), at frame / entry / all-entries scope.
- **Edit peaks**: manual boxes, Add-to-fitted (2D pygidfit or 1D);
  multi-select detected *or* fitted (`Ctrl+click`, `Ctrl+A`);
  copy/paste detected (`Ctrl+C` / `Ctrl+V`, `Ctrl+Shift+V` for a frame
  range); batch 2D fit; bulk delete; full undo/redo (`Ctrl+Z` /
  `Ctrl+Shift+Z`).
- **Inspect**: sortable Detected / Fitted / Matched tables with
  click-sync to the image; live radial + angular profile fits; overlay
  filters.
- **Export** figures (matplotlib) and peaks as CSV.

## Documentation

- **[Getting started](docs/getting_started.md)** — install, first run,
  shortcuts, troubleshooting.
- **[Backend compatibility](docs/backend_compatibility.md)** — pipeline
  version policy + how to verify a backend bump.
- **[Changelog](CHANGELOG.md)** — release highlights.

(For Contributor-facing architecture / module reference docs please contact me.)

## License

MIT — see [`LICENSE`](LICENSE). The optional `[pipeline]` extra pulls in
GPL-3.0 `mlgidMATCH` as a separate, optional dependency the GUI calls
(aggregation, not a derived work).

## Contact

Nico Lerch — <nico.lerch@uni-tuebingen.de>. Issues and feedback via
[GitHub issues](https://github.com/mlgid-project/mlgidLAB/issues).
