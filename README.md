# mlgidLAB

A desktop GUI for the [mlgidBASE](https://gitlab.com/mlgid/mlgidbase) GIWAXS
analysis pipeline. Drives the full
`pygid → mlgidDETECT → pygidFIT → mlgidMATCH` workflow — raw detector
images can be **converted to NeXus**, then **detected / fitted / matched**,
**reviewed and edited**, all in one PySide6 window.

---

## Quick start

1. **Install.** In a terminal with a working Python ≥ 3.11 environment:
   ```bash
   pip install -e .
   pip install -e ".[pipeline]"   # adds raw conversion + ML pipeline
   ```

2. **Launch:**
   ```bash
   mlgidlab
   ```

3. **Open files** with **File → Open…**, drag-and-drop onto the window,
   or pick from **File → Open recent**. The GUI auto-detects each file
   as either a converted NeXus file or raw detector data — you don't
   have to pick the right entry point. Multiple files can be open at
   once; click into a different file's subtree to switch the active
   session.

The GUI auto-switches between modes: when a raw file is active you
see the **Conversion** panel on the right; when a NeXus file is active
you see the **Pipeline** panel (detection / fitting / matching). Other
docks (image viewer, file tree, profiles, logs) stay the same.

A working sample is bundled at `example/BA2PbI4.h5` plus
`example/prepr_cifs.pickle` for matching.

---

## What you can do

### Open files
- **File → Open…** picks any HDF5 file; classification by content
  routes each one to the NeXus or raw flow automatically.
- **Drag-and-drop** files anywhere on the main window for the same
  routing. Multi-drop is honoured; raw files in one drop bundle into
  a single shared session.
- **File → Open recent** keeps the last 10 NeXus + raw paths via
  `QSettings`. Raw rows are prefixed with `[raw]`; tooltip shows the
  full path. Missing files are filtered out automatically.
- The **file browser dock** distinguishes raw and NeXus files at a
  glance via different standard icons (drive vs. file glyph).

### View
- Browse multiple HDF5 files in the left tree; click to switch between
  them.
- Toggle the central image between **polar** and **Cartesian
  (q_xy / q_z)**; for raw files the same canvas shows the detector
  image in pixels.
- Scrub through frames with the slider, change colormap, adjust
  intensity levels with the histogram next to the image. The
  frame slider, prev / next buttons, and Play toggle live on the
  image viewer's toolbar (so they remain reachable from any
  right-dock tab).
- **Frame keyboard shortcuts:** ← / → (and J / K) step prev / next;
  Home / End jump to first / last. Text-input widgets keep their
  caret nav for the same keys — the shortcuts only fire when focus
  isn't on a QSpinBox / QLineEdit.
- **Log / linear contrast toggle** on the image toolbar — useful for
  GIWAXS data with wide dynamic range. Coordinate axes and overlays
  are unaffected.
- **View → Reset layout** snaps every dock back to its cold-start
  position after the user has drag-rearranged things.
- Inspect any HDF5 dataset directly under the **Data** tab.
- The **status bar** along the bottom shows the active file (with `*`
  dirty marker), entry, current frame, pipeline state, and a live
  cursor readout. q-mode shows `q_xy / q_z` (cartesian) or `r / θ`
  (polar) plus intensity; raw mode shows `row / col / I`. Toggle the
  cursor segment via **View → Show cursor readout**.

### Convert (raw → NeXus)
- Multi-file batch conversion in one run.
- Choose **GID** or **Transmission** geometry and the conversion type
  (`det2q_gid`, `det2q`, `det2pol_gid`, `det2pol`).
- Provide a **PONI** calibration file, an optional **mask**
  (`.npy` / `.tif` / `.edf`), and the **angle of incidence**.
- **Create… buttons** next to the PONI and Mask fields open the
  embedded pyFAI calibration dialog so you can produce both files
  without leaving mlgidLAB. The dialog seeds the experiment image
  from the active raw scan (per-pixel mean across all frames so
  faint outer rings come out of the noise) and carries existing
  PONI / mask paths from the Conversion fields into the dialog so
  you can refine in place. Two prominent "Add … to conversion"
  buttons push saved paths back into the QLineEdits without
  closing the dialog.
- Edit **sample metadata** (YAML editor) and add **experimental
  metadata** manually or by picking a dataset from the raw HDF5 tree.
- Choose where the output goes:
  - *Separate files* — one converted file per raw input.
  - *Separate datasets* — every scan in one file as `entry_0000`,
    `entry_0001`, …
- Optionally name the output file. Re-converting into an existing
  file with **Overwrite existing file** unchecked **appends** new
  entries (`entry_0002`, `entry_0003`, …) instead of replacing the
  old ones.
- The freshly converted file is opened automatically when the run
  finishes.

### Detect / fit / match
- Run **Detection**, **Fitting**, or **Matching** from the Pipeline
  dock — every parameter the underlying mlgidBASE method takes is
  exposed as a named field (model type, clustering distances,
  `θ_fixed`, multiprocessing, matching thresholds, device,
  peaks-type, …).
- **Run full pipeline** button at the bottom of the Pipeline dock
  chains Detection → Fitting → Matching back-to-back using the
  current section kwargs. Disabled until the active matching source
  has a path; mid-stage errors are logged and the chain continues.
- Matching accepts a preprocessed CIF pickle, raw `.cif` files, or
  a folder of CIFs. The active source is picked via radio buttons
  next to each input row — the inactive row is greyed out so there
  is no ambiguity. Experimental parameters are derived per entry
  from the active NeXus file's instrument metadata (multi-energy
  files just work).
- Run-scope dropdowns let any stage operate on the active entry,
  all entries, the active frame, or all frames — "All entries"
  expands into one queued command per entry with per-entry log
  lines and per-entry error recovery.

### Edit peaks
- **Manual peaks:** `Ctrl+Alt`-drag a polar rectangle to label a
  candidate. One manual box per frame; drawing a new box replaces
  the existing one (single undo entry). `Esc` dismisses an
  in-progress box. Click any peak (manual / detected / fitted /
  matched) to select it; drag the ROI edges to resize (manual +
  detected + fitted); press `Delete` to remove. Deletion is
  scoped to the kind you selected — deleting a detected peak
  leaves its fitted / matched derivatives alone, and deleting a
  fitted peak invalidates only its dependent matched solutions.
  Geometry edits write straight into the NeXus file.
- Commit a manual or detected peak with **Add to detected** (uses
  the drawn box) or **Add to fitted**. The parameter panel carries
  a **1D / 2D fit-mode** radio pair that decides what
  Add-to-fitted runs:
  - **2D fit (pygidfit)** — default. Runs the same 2D Gaussian
    fit the pipeline uses for batch fitting, so the saved row is
    identical to what a fresh `Run fitting` would produce on the
    same box.
  - **1D fit (scipy)** — quicker fallback for narrow or unusual
    peaks where pygidfit can't converge; uses the live 1D
    Gaussian fits shown on the profile plots and zero-fills the
    2D shape coefficients.
- Both modes save the same width convention (`2σ` in radial and
  angular), so the saved blue box looks the same for the same
  Gaussian regardless of which mode produced it.
- **Live preview.** While a manual or detected peak is selected, a
  cyan dashed box on the image previews exactly where the next
  Add-to-fitted commit will land. In **1D mode** the pink fit
  curves on the profile plots track what Add-to-fitted (1D) will
  save. In **2D mode** the pink curves are hidden while a manual
  or detected box is selected (pygidfit's projected 1D Gaussian
  doesn't perfectly match the integrated profile, so the cyan
  image-side box is the single preview); clicking Add-to-fitted
  then auto-switches the selection to the new fitted peak, whose
  pink overlay appears immediately — fitted and matched peaks
  always show their overlay regardless of mode. Tick **Save
  fitted as ring** for full-azimuthal peaks: the cyan preview
  switches to a full-quadrant ring and ring storage forces 1D
  (pygidfit doesn't model rings).
- **Undo / redo** with `Ctrl+Z` / `Ctrl+Shift+Z` covers manual
  add / remove / geometry edits, detected / fitted geometry
  edits, and Add-to-fitted commits (undoing one deletes the new
  fitted row and reselects the source peak). Pipeline ops that
  re-index peak ids clear the history.
- The **Display dock** carries a master Matched-peaks toggle that
  cascades to every per-structure row; ticking a single structure
  while the master is off promotes it exclusively.

### Inspect
- The **Peaks** dock has tabbed sortable tables of every peak on the
  current frame — one tab each for **Detected**, **Fitted**, and
  **Matched**. Column headers are click-sortable; sort order
  persists across frame changes. **Bidirectional click-sync** with
  the image viewer: clicking a row selects the peak in the viewer
  and renders the white highlight overlay; clicking a peak in the
  image switches the dock to that peak's kind tab and scrolls the
  row into view. Selecting a matched structure highlights every
  peak that belongs to it at once.
- The **Display** dock carries overlay toggles (Detected / Fitted /
  Matched) plus per-layer filters:
  - **Detected**: a min-score slider under the Detected checkbox
    hides peaks whose model score is below the cutoff. Auto-seeds
    to the frame's lowest score on every frame change.
  - **Matched**: a per-row master cascade plus two filters above
    the rows — a CIF-name substring textbox and a min-probability
    slider (0.00–1.00). The two AND together; the slider auto-
    seeds to the frame's lowest probability so the default shows
    every match.

  All filters are inclusive (`p=1.0` passes when the slider sits at
  1.0). Filtered-out items disappear from both the dock and the
  image overlay without altering the per-structure checkbox state;
  drop the slider/filter to bring them back.
- The **Profiles** dock shows live radial and angular Gaussian fits
  of the selected peak (with linear background). The fit source
  depends on the active Add-to-fitted mode: in **1D mode** the
  pink curves are scipy fits over the user-drawn box; in **2D
  mode** they come from pygidfit's 2D fit, projected onto each
  axis so the curve sits on the integrated data over pygidfit's
  refined box. Either way, the pink curve previews what
  Add-to-fitted will store. The X range pans with the box on ROI
  drag so the borders stay visible. A **Log y** checkbox above
  the plots switches both y-axes to log10 — useful when peak
  amplitudes span multiple orders of magnitude.
- The matched palette uses 10 colours × 4 line styles for
  40 unique pens before any pair repeats — useful when matching
  against folders with many candidate CIFs.
- A shared **Logs** dock collects every pipeline and conversion
  log line as a separate tab.

### Save and export
- Edits land on a per-session temp copy. The original file is only
  touched when you choose **Save** or **Save As**. Each open file
  tracks its own unsaved-changes flag and prompts on close.
- **Tools → Export figure…** opens a non-modal window built around
  `mlgidbase.plot_analysis_results`. Settings column (layer
  toggles, entry / frame, colormap, intensity range, q range, DPI,
  figure size, plus collapsible per-layer styling for Detected /
  Fitted / Matched and a `set_plot_defaults` section) drives a
  matplotlib preview on the right. A **Render preview** button
  below the image redraws on demand; **Save figure** writes the
  rendered PNG. With the Matched layer on, mlgidbase writes one
  PNG per solution (suffix `_sol_NNNN`).
- **Tools → Export peaks as CSV…** writes detected, fitted, or
  matched peaks for the active frame, the active entry (all
  frames), or all entries. Detected/Fitted dump the full structured
  dtype (Fitted joins per-row error fields with `*_err` suffixes);
  Matched emits one row per solution with a `peak_list` cell
  carrying the fitted-peak indices the solution references.
- **Tools → Clear peaks** wipes one layer for the active entry:
  Detected; Fitted and Matched (clearing fitted also clears matched,
  since matched references fitted by id); or Matched only (removes the
  matched solutions, leaving detected and fitted intact).
- **Tools → Clear peaks → Reset all peaks** wipes detected +
  fitted + matched (and all manual peaks) at three scopes —
  Active entry (all frames), All entries, or Active frame
  (greyed out unless the file has more than one frame).

### Help
- **Help → Controls & shortcuts…** (F1) — modal reference for
  every keyboard shortcut, image-viewer mouse interaction, and
  the manual-peak commit workflow.
- **Help → About mlgidLAB…** — modal with a one-line description
  and a version table covering mlgidLAB, Python, OS, PySide6, Qt,
  numpy, h5py, silx, pyFAI, pyqtgraph, matplotlib, and mlgidbase.
- **Help → Copy diagnostics** — writes a plain-text blob to the
  clipboard with three sections (versions, active session
  details, last 50 log lines). Status-bar message confirms.
  Nothing is uploaded; the user pastes into their bug report.

---

## UI layout

```
┌────────────────┬───────────────────────────────────┬──────────────────────┐
│ File browser   │  Image  │  Data                   │  Display             │
│ (silx HDF5     │  ┌────────────────────────────┐   │  Pipeline /          │
│  tree)         │  │ pyqtgraph viewer           │   │     Conversion       │
│                │  │  Cartesian / Polar / Raw   │   │  Logs                │
│                │  │  frame slider + Play       │   │  (tabbed)            │
│                │  │  histogram + colormap      │   │                      │
│                │  └────────────────────────────┘   │                      │
│                ├───────────────────────────────────┤                      │
│                │  Profiles / Peaks (tabbed)        │                      │
├────────────────┴───────────────────────────────────┴──────────────────────┤
│ Status bar:  file  │  entry  │  frame  │  pipeline state  │  cursor       │
└───────────────────────────────────────────────────────────────────────────┘
```

- **Display dock** — entry selector, overlay toggles, Matched-peaks
  master + per-structure cascade + substring filter, Selected-peak
  parameter panel.
- **Pipeline dock** — Detection / Fitting / Matching plus the
  Run-full-pipeline button (visible in NeXus mode).
- **Conversion dock** — pygid raw-data conversion (visible in raw
  mode; tab position swaps with Pipeline per mode).
- **Logs dock** — shared by Pipeline and Conversion.
- **Profiles + Peaks** — share the bottom dock area as tabs. The
  **Profiles** tab carries radial + angular cross-sections of the
  selected peak; the **Peaks** tab is a tabbed sortable table
  (Detected / Fitted / Matched) with bidirectional click-sync to
  the image viewer. Profile is raised by default; one click flips
  to the table.

The **menu bar** runs **File · Edit · Tools · View · Settings · Help**:
- **File** — open, open recent, save, save-as, close, exit.
- **Edit** — undo / redo, Find peak by ID… (Ctrl+F).
- **Tools** — clear-peaks submenu, Export figure…, Export peaks
  as CSV…
- **View** — per-dock visibility toggles, cursor-readout toggle,
  Reset layout, Fullscreen image viewer (F11), Theme submenu
  (Dark / Light, persisted via QSettings).
- **Settings** — Playback settings…
- **Help** — Controls & shortcuts… (F1), About mlgidLAB…,
  Copy diagnostics.

---

## Conversion: detailed reference

The Conversion panel maps directly onto pygid's
`ExpParams` + `CoordMaps` + `Conversion` API:

- **Selection.** A tree of `(file, entry)` pairs with checkboxes;
  one shared *frame mode* (All / Single / List of indices) applies
  to every checked entry.
- **Experimental parameters.** PONI file, mask, angle of incidence;
  optional manual overrides for `centerX`, `centerY`, `SDD`,
  `wavelength`, `fliplr`, `flipud`, `transp`. Empty overrides fall
  through to the PONI value.
- **Conversion config.** Geometry (`GID` / `Transmission`),
  conversion type, orientation flags. **`vert_positive` and
  `hor_positive` default to checked** to match the pygid example
  notebook's recommended workflow — this puts the converted image
  in the upper-right (`+q_xy`, `+q_z`) quadrant. Uncheck either to
  keep the natural negative q range. Per-conversion-type parameter
  set: `(dq, q_xy_range, q_z_range)` for `det2q_gid`,
  `(dq, q_x_range, q_y_range)` for `det2q`,
  `(dang, dq, radial_range, angular_range)` for the polar variants.
- **Metadata.** Load / edit YAML for sample metadata; add
  experimental key/values manually or pick a dataset from the raw
  file's HDF5 tree.
- **Output.** Output directory, mode (separate files / separate
  datasets), optional custom filename, *Overwrite existing file*,
  *Overwrite existing dataset*. With overwrite off, successive runs
  append `entry_NNNN` groups.

Polar and profile views correctly handle converted images that span
any combination of quadrants (positive, negative, mixed, decreasing
axes); the peak-overlay layer covers the full `[-180°, 180°]` range
so peaks outside the upper-right quadrant still render.

---

## For developers

### How edits and runs hit disk

- Opening a NeXus file copies it into a session-local temp copy;
  every edit (peak add / delete / geometry drag, pipeline runs,
  Add-to-fitted commits) targets the temp file. The original is
  only overwritten on **Save**.
- Opening a raw file builds a read-only session that lists the
  raw inputs but never modifies them; conversion writes fresh
  NeXus output.
- Pipeline runs and conversion runs happen on worker threads, so
  the UI stays responsive. Long-running ops show progress in the
  status bar; the on-disk view detaches and re-attaches around
  each write so the file browser doesn't fight the writer.

In-depth notes on the modules, the editing model, the polar
transform, and the silx detach / reattach plumbing live in
`Documentation/` — start with `01_overview.md`.

### Install / extras

| extra        | adds                                  |
|---           |---                                    |
| (none)       | view-only mode + in-GUI pyFAI calib   |
| `[pipeline]` | `mlgidbase`, `pygid`, `PyYAML`        |
| `[dev]`      | `pytest`                              |

Runtime base: `PySide6`, `silx[full]`, `h5py`, `numpy`, `scipy`,
`pyqtgraph`, `qdarkstyle`, `pyFAI` (used by the in-GUI
calibration + mask creation dialog and the figure export
window). See `pyproject.toml` for pinned minimums.

`pygid` and `mlgidbase` are imported lazily — the GUI starts up
without them in view-only mode (open, browse, edit peaks
manually, no Run buttons enabled). Install the `[pipeline]` extra
to enable Conversion, Detection, Fitting, and Matching.

### Status

Pre-release (`0.0.1`). Suitable for interactive review and editing
of mlgidBASE outputs plus single-step raw-data conversions; bulk
processing beyond the GUI's batch should still go through the
pipeline scripts.
