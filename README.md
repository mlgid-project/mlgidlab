# mlgidBASE_GUI

A desktop GUI for the [mlgidBASE](https://gitlab.com/mlgid/mlgidbase) GIWAXS
analysis pipeline. Wraps the four-stage `pygid → mlgidDETECT → pygidFIT →
mlgidMATCH` workflow in a PySide6 interface that lets you label, edit, and
review peaks on top of the raw and polar images, then commit changes back to
the NeXus file.

## What it does

- **Browse NeXus files.** silx-powered HDF5 tree on the left; multiple files
  can be opened in the same window and switched by clicking into the tree.
- **View frames in real or polar coordinates.** pyqtgraph image canvas with
  a frame slider, overlay toggles, configurable colormap (default: `magma`),
  and a Data tab for inspecting any HDF5 dataset directly.
- **Label peaks manually.** `Ctrl+Alt`-drag a polar rectangle to mark a
  candidate peak. Manual peaks are kept in memory until you commit them to
  the file as detected or fitted entries.
- **Edit and delete file-resident peaks.** Click any detected, fitted, or
  matched peak to select it; drag the ROI edges to resize (detected and
  fitted only); press `Delete` to cascade-remove via `mlgidBASE.delete_peak`.
  Geometry edits are written straight to HDF5 in place.
- **Inspect 1D radial / azimuthal profiles.** The bottom dock shows the
  radial and angular cross-sections of the selected box, with live Gaussian
  fits that follow ROI drags. Manual peaks get a real bounded refit;
  detected/fitted/matched peaks render the synthetic Gaussian implied by
  their stored width.
- **Commit fits back to the file.** "Add to detected" appends a row to
  `detected_peaks` using the box geometry; "Add to fitted" appends a row to
  `fitted_peaks` using the 1D Gaussian fit parameters (radial border = FWHM,
  azimuthal border = 2 × FWHM).
- **Run the pipeline stages.** The Pipeline dock exposes Detection, Fitting,
  and Matching as buttons. Matching takes a preprocessed CIF pickle, a
  peaks-type (segments / rings), a threshold, and a device (cpu / cuda).
  Per-run logs stream into the dock; the GUI is gated `busy` while a run is
  in flight.
- **Undo / redo.** `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) walk an
  Action-protocol history covering manual add, manual remove, manual
  geometry edits, and detected/fitted geometry edits. Pipeline operations
  that reshuffle peak ids clear the history.
- **Per-file save.** Edits land on a per-session temp copy; the original is
  only touched when you choose Save (overwrite) or Save As. Each open file
  tracks its own dirty flag and prompts on close.

## Install

This project uses Python ≥ 3.11. From a clone:

```bash
pip install -e .
# Optional — pulls torch and the mlgid analysis stack:
pip install -e ".[pipeline]"
```

The `pipeline` extra is what enables the Detection / Fitting / Matching
buttons. Without it the GUI still runs in view-only mode.

Runtime dependencies: `PySide6`, `silx[full]`, `h5py`, `numpy`, `scipy`,
`pyqtgraph`, `qdarkstyle`. See `pyproject.toml` for pinned minimums.

## Run

```bash
mlgidbase-gui
# or
python -m mlgidbase_gui
```

A NeXus file produced by mlgidBASE is expected — see `example/BA2PbI4.h5`
for a working sample, and `example/prepr_cifs.pickle` for the matching
input.

## UI layout

- **Left dock — File browser.** silx HDF5 tree. Multi-select Open
  (`Ctrl+O`) appends every chosen file. Clicking a node from a different
  file makes it the active session.
- **Center — Image / Data tabs.** Image tab is the pyqtgraph viewer with
  Cartesian / polar mode toggle, frame slider, and colormap controls.
  Data tab is silx's `DataViewerFrame` — auto-renders whatever HDF5
  dataset is selected in the tree.
- **Right docks — Display / Pipeline (tabbed).**
  - *Display*: entry selector, overlay visibility checkboxes (manual,
    detected, fitted, matched-by-structure), and the **Selected peak**
    panel with detected / fitted parameter readouts plus add / delete
    actions.
  - *Pipeline*: Detection / Fitting / Matching launchers and the run log.
- **Bottom dock — Profiles.** Radial and angular cross-sections of the
  selected peak with live Gaussian fits.
- **View menu.** Toggles for every dock (mirrors the right-click title-bar
  menu).
- **Edit menu.** Undo / Redo entries with the standard shortcuts.
- **Tools menu.** Bulk operations that don't fit the per-peak ROI
  workflow. Today: clear-all of one peak kind (manual / detected /
  fitted / matched) for the active entry. Clearing fitted also clears
  matched, since matched_* references fitted ids.

## Repository layout

```
mlgidbase_gui/
  main_window.py     QMainWindow: menus, docks, session/pipeline plumbing
  image_viewer.py    pyqtgraph viewer, ROI, overlays, undo stack
  profile_viewer.py  1D radial + angular profile widget with live fits
  parameter_panel.py "Selected peak" readout + commit/delete buttons
  pipeline_panel.py  Detection / Fitting / Matching launcher
  pipeline.py        Lazy mlgidbase wrappers (no Qt)
  file_model.py      h5py reads + targeted in-place writes
  fit.py             1D Gaussian fitting helpers
  polar.py           Cartesian↔polar transform cache
  session.py         Per-file working copy (temp dir + dirty flag)
  workers.py         QThread workers for open + pipeline runs
  theme.py           qdarkstyle + pyqtgraph color overrides
example/             Sample NeXus file + matching CIF pickle
```

## Editing model

- Opening a file copies it into a session-local temp directory; all edits
  target the temp file. The original is only overwritten on **Save**.
- Geometry edits (box drag) on detected / fitted peaks open the temp file
  `r+` and rewrite the matching row by `id` — silx is detached for the
  write and reattached afterward.
- Pipeline runs (Detection / Fitting / Matching, `add_peak`, `delete_peak`)
  go through the same detach/reattach dance and run on a worker thread so
  the UI stays responsive.
- The undo stack uses an Action protocol (`ManualAddAction`,
  `ManualRemoveAction`, `ManualGeomAction`, `FileGeomAction`). Pipeline
  ops that reshuffle ids invalidate the stack and clear it on completion.

## Tools menu — future ideas

The Tools menu is the home for batch / cross-cutting operations. Likely
additions, roughly in order of effort:

- **Re-fit all detected peaks.** One-click sweep that runs the radial /
  angular Gaussian fit on every detected_peaks row and appends the
  results to fitted_peaks.
- **Run full pipeline.** Detection → Fitting → Matching as a single
  chained job, scoped by the same Active-entry / All-entries dropdown.
- **Apply intensity threshold.** Bulk-delete peaks whose amplitude is
  below a user-set value (per kind), useful for pruning low-confidence
  detections after a noisy run.
- **Copy peaks across frames.** Take the current frame's peaks of one
  kind and replicate to a frame range — useful when a series shares
  one diffraction pattern.
- **Symmetrize peaks.** Mirror peaks across q_z = 0 (or any user-chosen
  axis) — shorthand for the common GIWAXS symmetry assumption.
- **Export peaks to CSV.** Per-frame or stacked CSV of detected /
  fitted / matched tables for downstream analysis in pandas / Excel.
- **Export image as PNG.** Snapshot the current Cartesian or polar
  view at full resolution with overlays included.
- **Frame statistics.** Per-frame histogram of peak counts (detected /
  fitted / matched), fit-quality score distributions, ring vs segment
  ratios — useful as a glance health check for a series.
- **Diff against baseline.** Compare two open files and highlight
  per-frame differences in detected / fitted peaks; helpful when
  tweaking detection configs.
- **Reset analysis.** Single-click "Clear detected + fitted +
  matched" — the common "I want to redo from scratch" combo.
- **Recompute polar transform.** Force a refresh of the cached polar
  stack (e.g. after editing pygid metadata externally).
- **Save snapshot.** Snapshot the temp file alongside an annotation
  message so the user can step back to known-good states without
  Save-As-ing the original.

These are deliberately quick, self-contained operations; anything that
requires its own dock or custom widgets should live elsewhere.

## Status

Pre-release (`0.0.1`). Intended for interactive review of mlgidBASE outputs
and small per-frame corrections; bulk edits should still go through the
pipeline scripts.
