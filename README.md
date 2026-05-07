# mlgidBASE_GUI

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
   mlgidbase-gui
   ```

3. **Pick a workflow** from the **File** menu:
   - *Open NeXus…* — view / edit / analyse an already-converted file.
   - *Open raw data…* — convert one or more raw HDF5 detector files into
     a NeXus file first, then analyse.

The GUI auto-switches between the two modes: when a raw file is active you
see a **Conversion** panel on the right; when a NeXus file is active you
see the **Pipeline** panel (detection / fitting / matching). Other docks
(image viewer, file tree, profiles, logs) stay the same.

A working sample is bundled at `example/BA2PbI4.h5` plus `example/prepr_cifs.pickle`
for matching.

---

## What you can do

### View
- Browse multiple HDF5 files in the left tree; click to switch between them.
- Toggle the central image between **polar** and **Cartesian (q_xy / q_z)**;
  for raw files the same canvas shows the detector image in pixels.
- Scrub through frames with the slider, change colormap, adjust intensity
  levels with the histogram next to the image.
- Inspect any HDF5 dataset directly under the **Data** tab.

### Convert (raw → NeXus)
- Multi-file batch conversion in one run.
- Choose **GID** or **Transmission** geometry and the conversion type
  (`det2q_gid`, `det2q`, `det2pol_gid`, `det2pol`).
- Provide a **PONI** calibration file, an optional **mask**
  (`.npy` / `.tif` / `.edf`), and the **angle of incidence**.
- Edit **sample metadata** (YAML editor) and add **experimental metadata**
  manually or by picking a dataset from the raw HDF5 tree.
- Choose where the output goes:
  - *Separate files* — one converted file per raw input.
  - *Separate datasets* — every scan in one file as `entry_0000`,
    `entry_0001`, …
- Optionally name the output file. Re-converting into an existing file
  with **Overwrite existing file** unchecked **appends** new entries
  (`entry_0002`, `entry_0003`, …) instead of replacing the old ones.
- The freshly converted file is opened automatically when the run finishes.

### Detect / fit / match
- Run **Detection**, **Fitting**, or **Matching** from the Pipeline dock —
  every parameter the underlying mlgidBASE method takes is exposed as a
  named field (model type, clustering distances, `θ_fixed`, multiprocessing,
  matching thresholds, device, peaks-type, …).
- Matching accepts a preprocessed CIF pickle, raw `.cif` files, or a folder
  of CIFs; experimental parameters are derived from the active NeXus file.

### Edit peaks
- **Manual peaks:** `Ctrl+Alt`-drag a polar rectangle to label a candidate.
  Click any peak (manual / detected / fitted / matched) to select it; drag
  the ROI edges to resize (manual + detected only); press `Delete` to
  remove. Geometry edits write straight into the NeXus file.
- Commit a manual peak to the file with **Add to detected** (uses the box)
  or **Add to fitted** (uses the live 1D Gaussian fit; tick *Save fitted as
  ring* for full-azimuthal peaks).
- **Undo / redo** with `Ctrl+Z` / `Ctrl+Shift+Z` covers manual add /
  remove / geometry edits and detected/fitted geometry edits. Pipeline
  ops that re-index peak ids clear the history.

### Inspect
- The **Profiles** dock shows live radial and angular Gaussian fits of the
  selected peak; manual peaks get a real bounded refit, file-resident peaks
  render the Gaussian implied by their stored width.
- A shared **Logs** dock collects every pipeline and conversion log line
  as a separate tab.

### Save
- Edits land on a per-session temp copy. The original file is only touched
  when you choose **Save** or **Save As**. Each open file tracks its own
  unsaved-changes flag and prompts on close.
- **Tools → Export current frame as PNG…** captures the visible image
  with overlays at full resolution.
- **Tools → Clear peaks** wipes any one kind (manual / detected / fitted /
  matched) for the active entry; clearing fitted cascades to matched.

---

## UI layout

```
┌────────────────┬───────────────────────────────────┬──────────────────────┐
│ File browser   │  Image  │  Data                   │  Display             │
│ (silx HDF5     │  ┌────────────────────────────┐   │  Pipeline            │
│  tree)         │  │ pyqtgraph viewer           │   │  Conversion          │
│                │  │  Cartesian / Polar / Raw   │   │  Logs                │
│                │  │  histogram + colormap      │   │  (tabbed)            │
│                │  └────────────────────────────┘   │                      │
│                ├───────────────────────────────────┤                      │
│                │  Profiles (radial + angular)      │                      │
└────────────────┴───────────────────────────────────┴──────────────────────┘
```

- **Display dock** — entry selector, frame slider, overlay toggles,
  Selected-peak panel.
- **Pipeline dock** — Detection / Fitting / Matching (visible in NeXus mode).
- **Conversion dock** — pygid raw-data conversion (visible in raw mode).
- **Logs dock** — shared by Pipeline and Conversion.
- **Profiles dock** — radial + angular cross-sections of the selected peak.

The **View** menu toggles every dock; the **Edit** menu has Undo / Redo;
**Tools** holds bulk operations (clear, PNG export); **File** holds
NeXus / raw open + save.

---

## Conversion: detailed reference

The Conversion panel maps directly onto pygid's
`ExpParams` + `CoordMaps` + `Conversion` API:

- **Selection.** A tree of `(file, entry)` pairs with checkboxes; one shared
  *frame mode* (All / Single / List of indices) applies to every checked
  entry.
- **Experimental parameters.** PONI file, mask, angle of incidence;
  optional manual overrides for `centerX`, `centerY`, `SDD`,
  `wavelength`, `fliplr`, `flipud`, `transp`. Empty overrides fall through
  to the PONI value.
- **Conversion config.** Geometry (`GID` / `Transmission`), conversion type,
  orientation flags. **`vert_positive` and `hor_positive` default to checked**
  to match the pygid example notebook's recommended workflow — this puts
  the converted image in the upper-right (`+q_xy`, `+q_z`) quadrant.
  Uncheck either to keep the natural negative q range.
  Per-conversion-type parameter set: `(dq, q_xy_range, q_z_range)` for
  `det2q_gid`, `(dq, q_x_range, q_y_range)` for `det2q`,
  `(dang, dq, radial_range, angular_range)` for the polar variants.
- **Metadata.** Load / edit YAML for sample metadata; add experimental
  key/values manually or pick a dataset from the raw file's HDF5 tree.
- **Output.** Output directory, mode (separate files / separate datasets),
  optional custom filename, *Overwrite existing file*, *Overwrite existing
  dataset*. With overwrite off, successive runs append `entry_NNNN`
  groups.

Polar and profile views correctly handle converted images that span any
combination of quadrants (positive, negative, mixed, decreasing axes); the
peak-overlay layer covers the full `[-180°, 180°]` range so peaks outside
the upper-right quadrant still render.

---

## For developers

### Repository layout

```
mlgidbase_gui/
  main_window.py     QMainWindow: menus, docks, session / pipeline / conversion plumbing
  image_viewer.py    pyqtgraph viewer, ROI, overlays, undo stack, raw-mode rendering
  profile_viewer.py  1D radial + angular profile widget with live Gaussian fits
  parameter_panel.py "Selected peak" readout + commit / delete buttons
  pipeline_panel.py  Detection / Fitting / Matching launcher (scrollable, multi-expand)
  conversion_panel.py pygid raw → NeXus conversion launcher (scrollable, multi-expand)
  pipeline.py        Lazy mlgidBASE wrappers (no Qt)
  conversion.py      Lazy pygid wrappers (no Qt)
  file_model.py      h5py reads + targeted in-place writes + raw-entry walker
  fit.py             1D Gaussian fitting helpers
  polar.py           Cartesian↔polar transform; handles arbitrary q-axis orientation
  session.py         BaseSession + NexusSession (writable temp copy) + RawSession (read-only batch)
  workers.py         QThread workers for open / pipeline / conversion / CIF parse
  theme.py           qdarkstyle + pyqtgraph color overrides
example/             Sample NeXus file + matching CIF pickle
```

### Editing model

- Opening a NeXus file copies it into a session-local temp directory; all
  edits target the temp file. The original is only overwritten on **Save**.
- Opening a raw file builds a read-only `RawSession` that lists the raw
  inputs but never modifies them; conversion writes a fresh NeXus output.
- Geometry edits (ROI drag) on detected / fitted peaks open the temp file
  `r+` and rewrite the matching row by `id` — silx is detached for the
  write and reattached afterward.
- Pipeline and conversion runs go through the same detach / reattach
  dance and run on worker threads so the UI stays responsive.
- The undo stack uses an `_Action` protocol (`ManualAddAction`,
  `ManualRemoveAction`, `ManualGeomAction`, `FileGeomAction`). Pipeline
  ops that reshuffle ids invalidate the stack and clear it on completion.

### Conversion engine

`conversion.execute(scans, cfg)`:
1. Builds **one** shared `pygid.ExpParams` and `pygid.CoordMaps` per run
   (the roadmap's "global objects").
2. For each `RawScan`, instantiates a `pygid.Conversion` and dispatches
   on `cfg.conv_type` (`det2q_gid` / `det2q` / `det2pol_gid` / `det2pol`).
3. Output paths and `entry_NNNN` group names are pre-planned by
   `_plan_output_paths` and `_next_entry_index` so re-runs append
   instead of overwriting.
4. Sample / experimental metadata are passed through to pygid's
   `ExpMetadata` / `SampleMetadata`.

`pygid` is imported lazily so the GUI runs without it (view-only mode).
Same pattern as `pipeline.py` for `mlgidbase`.

### Install / extras

| extra        | adds                                  |
|---           |---                                    |
| (none)       | view-only mode                        |
| `[pipeline]` | `mlgidbase`, `pygid`, `PyYAML`        |
| `[dev]`      | `pytest`                              |

Runtime base: `PySide6`, `silx[full]`, `h5py`, `numpy`, `scipy`,
`pyqtgraph`, `qdarkstyle`. See `pyproject.toml` for pinned minimums.

### Status

Pre-release (`0.0.1`). Suitable for interactive review and editing of
mlgidBASE outputs plus single-step raw-data conversions; bulk processing
beyond the GUI's batch should still go through the pipeline scripts.
