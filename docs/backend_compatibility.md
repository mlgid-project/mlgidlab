# Backend compatibility — verifying mlgidbase & the pipeline stack

mlgidLAB runs standalone (view + edit existing NeXus results) on its
public-PyPI runtime dependencies alone. Detection, fitting, and
matching require the in-house pipeline stack — installed via the
`[pipeline]` extra. That stack moves faster than the GUI, so this
document is the procedure for confirming a given pipeline release still
drives mlgidLAB the same way, plus the exact API surface the GUI
depends on (the things that break if an upstream signature drifts).

Run this whenever you bump `mlgidbase` (or any backend), before
shipping a release, and as part of the alpha pre-flight.

## Verified-good baseline

All 186 tests pass with this set installed (the backend-dependent
tests un-skip and run, not just the pure-h5py ones):

| Package | Baseline version | Reached how |
|-|-|-|
| `mlgidbase` | 0.1.3 | declared in `[pipeline]` |
| `pygid` | 0.2.10 | declared in `[pipeline]` |
| `pygidfit` | 0.1.3 | declared in `[pipeline]` (GUI imports it directly) |
| `mlgidmatch` | 0.1.3 | declared in `[pipeline]` (GUI imports it directly) |
| `pygidsim` | 0.1.4 | declared in `[pipeline]` (GUI imports it directly) |
| `mlgiddetect` | 0.2.3 | transitive via `mlgidbase` (GUI never imports it) |

**Pinning policy — exact `==`, not floors.** Each in-house backend
release can shift the detection/fitting/matching numerics or the
on-disk schema, so a bump must be a deliberate, test-rechecked change
rather than something pip picks up silently. The `[pipeline]` extra
therefore pins every directly-imported backend to an exact version
(the baseline above). `mlgidbase 0.1.3` itself pins its own backends
with `==` too (`pygid==0.2.10`, `mlgiddetect==0.2.3`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`); `pygidsim==0.1.4` arrives via `mlgidmatch` and
`pygid`. The GUI still declares `pygidfit`/`mlgidmatch`/`pygidsim`
explicitly — rather than leaning on `mlgidbase`'s transitive closure —
because it imports those three *directly*, so a future `mlgidbase`
dropping one must surface as a clear resolver error, not a runtime
`ImportError`. `mlgiddetect` is left transitive (the GUI never imports
it). **To move the baseline up:** bump the pins here and in
`pyproject.toml`, re-run the full suite + the end-to-end demo loop,
then commit.

**Runtime (non-pipeline) deps stay on `>=` floors on purpose.** The GUI
stack — PySide6, pyqtgraph, silx, numpy, etc. — is *not* pinned exact,
so the CI matrix can keep testing forward-compat across Python
3.11-3.14 against whatever those projects release. A cyclic-GC-during-
construction segfault that the newest PySide6 (6.11.1) + Python 3.13/14
exposed is handled at the source in `tests/conftest.py` (`gc.disable()`
for the session) plus an idempotent `closeEvent`, not by pinning the
GUI stack down — verified green with PySide6 6.11.1 + pyqtgraph 0.14.0
on both Python 3.12 and 3.14.

## Step 1 — Is mlgidbase installable from your index?

```bash
pip index versions mlgidbase        # lists versions visible on the active index
pip download mlgidbase --no-deps -d /tmp/mlgidbase_check   # confirms it resolves + fetches
```

If those fail against public PyPI, the package is coming from a
private index — inspect the configured indexes:

```bash
pip config list                     # look for index-url / extra-index-url
cat ~/.config/pip/pip.conf 2>/dev/null
```

Web cross-check: open `https://pypi.org/project/mlgidbase/`.

## Step 2 — Does the latest stack still drive the GUI?

In a throwaway environment, so the verified-good env stays intact:

```bash
conda create -n mlgidlab_compat python=3.12 -y
conda activate mlgidlab_compat
git clone https://github.com/mlgid-project/mlgidLAB && cd mlgidLAB
pip install mlgidbase pygid pygidfit mlgidmatch     # pull the latest from PyPI
pip install -e ".[dev]"
python -c "import importlib.metadata as m; \
  [print(p, m.version(p)) for p in \
  ('mlgidbase','pygid','pygidfit','mlgidmatch','pygidsim','mlgiddetect')]"
pytest -q          # backend tests un-skip when the stack imports
```

Watch for: the pipeline / manual-fit / energy-guard / matched-
invalidation tests must **run** (not `importorskip`-skip) and pass. A
skip there means a backend failed to import — read the collection log.

Then a behavioral end-to-end pass in the GUI (the demo loop):

```bash
python -m mlgidlab
```

1. Open `example/eiger4m_0000.h5` (raw) and convert it.
2. On the converted file: Pipeline dock → Run Detection → Run Fitting.
3. Parse CIFs (or load `example/prepr_cifs.pickle`) → Run Matching.
4. Overlays + Peaks-dock tables populate for all three kinds.

If any stage errors, the failing call is in the API-surface table
below — that tells you which upstream signature changed.

## Step 3 — API surface the GUI depends on

If a stage breaks after a bump, the cause is almost always one of
these. Verbatim call sites are in the code; this is the contract.

**`mlgidbase.mlgidBASE`** (`pipeline.py`, `figure_export_window.py`)
- `mlgidBASE(filename=str(path))` constructor.
- `analysis.run_detection(**kwargs)`, `run_fitting(**kwargs)`,
  `run_matching(**kwargs)` — resolved dynamically by name.
- `analysis.set_plot_defaults(**style)` and
  `analysis.plot_analysis_results(save_fig=True, path_to_save_fig=...,
  plot_result=False, return_result=False, **kwargs)` (figure export).

**`pygid`** (`conversion.py`)
- `pygid.ExpParams(poni_path=..., ai=..., mask_path=...)`
- `pygid.CoordMaps(params, hor_positive=..., vert_positive=..., dq=...,
  dang=..., q_xy_range=..., q_z_range=..., ...)`
- `pygid.Conversion(matrix=..., path=..., dataset=..., frame_num=...)`
  then `det2q_gid` / `det2q` / `det2pol_gid` / `det2pol`
  `(frame_num=..., return_result=False, save_result=True,
  path_to_save=..., h5_group=..., overwrite_file=..., overwrite_group=...)`
- `pygid.SampleMetadata(data=...)`, `pygid.ExpMetadata(**fields)`, and
  the writable `ExpMetadata.extend_fields` attribute (set inside a
  `try/except` — older pygid lacking it is tolerated).

**`pygidsim.experiment.ExpParameters`** (`pipeline.py`)
- `ExpParameters(q_xy_max=..., q_z_max=..., ai=..., en=...)` — `en` in
  eV; the energy guard expects `1e3 <= en <= 2e5`.

**`mlgidmatch.preprocess.cif_preprocess.CifPattern`** (`pipeline.py`)
- `CifPattern(params=..., folder_path=..., cifs=..., create_all=True)`;
  the GUI reads `cif_pattern.params.en` and `.params.ai` (via
  `getattr(..., default)`) to detect a parameter mismatch.

**`pygidfit.process_scans`** (`manual_fit.py`)
- `img_preprocessing(cartesian, ai_deg, crit_angle, wavelength_A, q_z)`
- `_get_polar_grid(img_shape, (512, 1024), [0, 0])` — the polar grid is
  hardcoded `(512, 1024)`; a change here breaks byte-identity between
  manual 2D fits and pipeline fits.
- `polar_conversion(img_pre, yy, zz, cv2.INTER_LINEAR)`
- `fit_data(polar_img, radius=..., radius_width=..., angle=...,
  angle_width=..., wavelength=..., q_xy_max=..., q_z_max=..., ...)`

**Data contracts** (file_model + pipeline post-processing)
- NeXus paths: `{entry}/data/img_gid_q`, `{entry}/data/q_xy`,
  `{entry}/data/q_z`, `{entry}/data/analysis/frameNNNNN/<kind>_peaks`.
- `fitted_peaks` dtype fields: `q_xy, q_z, radius, angle,
  radius_width, angle_width, amplitude, id, score, is_ring`.
- `matched_peaks` per-solution fields: `CIF, h, k, l, peak_list,
  probability` (`peak_list` recast to int32; rows deduped by max
  `probability`).

## Step 4 — Record the result

If the latest stack passes, note the confirmed versions here (update
the baseline table) and, if a version moved, bump the `[pipeline]` pins
in `pyproject.toml`. If it fails, file the failing call + the version
that introduced the break as a known issue and keep the last-good
pin in place.
