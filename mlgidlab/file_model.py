"""Pure-Python readers for the mlgidBASE NeXus file format.

Schema reference: project_repos/mlgidBASE/docs/tutorials/output_file_format.md
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

import logging
logger = logging.getLogger(__name__)

ENTRY_PREFIX = "entry_"
FRAME_KEY_FMT = "frame{:05d}"

# Target RAM ceiling for the per-frame caches owned by FrameSource. Sized
# at ~1 GB combined across Cartesian + polar so a typical 8 GB-RAM laptop
# stays headroom-positive while still keeping scrub instant for a small
# window around the current frame. MIN keeps single-frame scrub from
# falling off the cache; MAX prevents pathological cases where tiny
# frames (<1 MB) would otherwise spawn thousands of LRU entries.
FRAME_LRU_BYTES_TARGET = 1_024 * 1_024 * 1_024
FRAME_LRU_MIN = 4
FRAME_LRU_MAX = 64


def is_entry_group_name(name: str) -> bool:
    """Whether ``name`` looks like a NeXus entry group.

    pygid's default writer uses ``entry_0000``, ``entry_0001``, …, but
    other writers expose a single ``entry`` group (no numeric suffix)
    or use mnemonic suffixes like ``entry_horiz``. Both forms are valid
    NeXus and we shouldn't filter out files just because their entries
    don't follow the numeric-suffix convention.
    """
    return name == "entry" or name.startswith(ENTRY_PREFIX)
ANALYSIS_REL = "data/analysis"
IMG_REL = "data/img_gid_q"
QXY_REL = "data/q_xy"
QZ_REL = "data/q_z"

PeakKind = str  # "detected" | "fitted"


@dataclass
class PeakTable:
    """Centers, widths, and ids for a set of peaks at one frame.

    Both Cartesian (q_xy, q_z) and polar (angle, radius) coordinates are kept
    because the peak operations API exposes both.
    """

    q_xy: np.ndarray
    q_z: np.ndarray
    angle: np.ndarray
    radius: np.ndarray
    angle_width: np.ndarray
    radius_width: np.ndarray
    is_ring: np.ndarray
    ids: np.ndarray
    # mlgidDETECT writes the model confidence into ``score``; mlgidFIT
    # copies it onto every fitted row. Manual peaks carry zeros since
    # they have no model provenance. Kept as a parallel array so that
    # rendering helpers (which iterate by index) can stay agnostic.
    score: np.ndarray
    # Peak amplitude (2D-Gaussian peak height). Present on both
    # detected and fitted rows; manual peaks carry zeros.
    amplitude: np.ndarray

    @classmethod
    def from_dataset(cls, ds: h5py.Dataset) -> PeakTable:
        arr = ds[()]
        n = int(arr.shape[0])
        # Tolerate older files written before optional fields
        # existed — return zeros so downstream code can still index.
        score = (
            np.asarray(arr["score"], dtype=float)
            if "score" in arr.dtype.names
            else np.zeros(n, dtype=float)
        )
        amplitude = (
            np.asarray(arr["amplitude"], dtype=float)
            if "amplitude" in arr.dtype.names
            else np.zeros(n, dtype=float)
        )
        return cls(
            q_xy=np.asarray(arr["q_xy"], dtype=float),
            q_z=np.asarray(arr["q_z"], dtype=float),
            angle=np.asarray(arr["angle"], dtype=float),
            radius=np.asarray(arr["radius"], dtype=float),
            angle_width=np.asarray(arr["angle_width"], dtype=float),
            radius_width=np.asarray(arr["radius_width"], dtype=float),
            is_ring=np.asarray(arr["is_ring"], dtype=bool),
            ids=np.asarray(arr["id"], dtype=int),
            score=score,
            amplitude=amplitude,
        )

    def __len__(self) -> int:
        return int(self.ids.size)


@dataclass
class EntryStack:
    """Image stack + reciprocal-space axes for one entry."""

    image_stack: np.ndarray  # shape (n_frames, n_qz, n_qxy)
    q_xy: np.ndarray         # shape (n_qxy,)
    q_z: np.ndarray          # shape (n_qz,)

    @property
    def n_frames(self) -> int:
        return int(self.image_stack.shape[0])


def list_entries(file_path: Path) -> list[str]:
    """Return q-image entry names in sorted order.

    Only ``img_gid_q`` entries are exposed: those are the ones the viewer
    knows how to render and the only ones mlgidDETECT / mlgidFIT actually
    process. Polar entries (``img_gid_pol``) and any non-q sibling groups
    are filtered out so the entry dropdown can't lead the user to an
    entry that would silently no-op a pipeline run or crash the loader.

    The filter mirrors pygid's reading convention: each entry's ``data``
    group has a ``signal`` attribute naming the active image dataset.
    """
    return [name for name, signal in list_entry_signals(file_path).items()
            if signal == "img_gid_q"]


def read_geometry_for_entry(
    file_path: Path, entry: str, frame: int = 0,
) -> dict | None:
    """Read the geometry fields a single-peak 2D fit needs.

    Returns ``{'wavelength_angstrom', 'q_xy_max', 'q_z_max', 'ai_deg',
    'q_z_axis'}`` on success, or ``None`` when any required field is
    missing. The GUI's "Add to fitted" handler uses this to feed
    ``mlgidlab.manual_fit.fit_one_peak`` without dragging in
    ``pygidsim`` (so the headless CI suite can still exercise the
    Add-to-fitted path's fallback branch by patching pygidfit out).

    Wavelength is stored in metres in NeXus; the fitter expects
    Ångströms, so we convert here at the boundary. The angle of
    incidence falls back to ``0.0`` (transmission geometry) when the
    dataset is missing, since pygidfit's default for the same
    parameter is ``0``. ``frame`` selects which entry of a
    per-frame ``angle_of_incidence`` array to read (0-D / length-1
    arrays use index 0 regardless). The q_xy / q_z extents are
    required; ``q_z_axis`` is the 1-D q_z array (returned because
    ``pygidfit.process_scans.img_preprocessing`` masks rows below
    the sample horizon via a q_z comparison).
    """
    try:
        with h5py.File(file_path, "r") as f:
            if entry not in f:
                return None
            grp = f[entry]
            q_xy_arr = np.asarray(grp["data/q_xy"][()])
            q_z_arr = np.asarray(grp["data/q_z"][()])
            q_xy_max = float(np.max(np.abs(q_xy_arr)))
            q_z_max = float(np.max(np.abs(q_z_arr)))
            wl_m = float(
                np.asarray(grp["instrument/monochromator/wavelength"]).ravel()[0]
            )
            if wl_m <= 0:
                return None
            try:
                ai_raw = np.asarray(grp["instrument/angle_of_incidence"]).ravel()
                ai_idx = int(frame) if 0 <= int(frame) < ai_raw.size else 0
                ai_deg = float(ai_raw[ai_idx])
            except Exception:
                ai_deg = 0.0
    except Exception:
        logger.debug("suppressed exception in read_geometry_for_entry", exc_info=True)
        return None
    return {
        "wavelength_angstrom": wl_m * 1e10,
        "q_xy_max": q_xy_max,
        "q_z_max": q_z_max,
        "ai_deg": ai_deg,
        "q_z_axis": q_z_arr,
    }


def count_frames(file_path: Path, entry: str) -> int:
    """Return the number of frames in ``entry`` without loading any data.

    Reads only the shape header of ``<entry>/data/img_gid_q`` (or whatever
    the entry's signal points at) via h5py — sub-millisecond on local
    storage. Returns ``0`` on any error: missing entry, missing /data,
    missing signal, dataset isn't 3-D, or any other shape mismatch. Used
    by ``PipelineWorker`` to size the multi-frame progress bar before
    invoking mlgidBASE.
    """
    try:
        with h5py.File(file_path, "r") as f:
            if entry not in f:
                return 0
            data = f[entry].get("data")
            if not isinstance(data, h5py.Group):
                return 0
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            if not isinstance(signal, str) or signal not in data:
                return 0
            ds = data[signal]
            if ds.ndim < 1:
                return 0
            return int(ds.shape[0])
    except Exception:
        logger.debug("suppressed exception in count_frames", exc_info=True)
        return 0


def list_entry_signals(file_path: Path) -> dict[str, str | None]:
    """Return ``{entry_name: signal_or_None}`` for every entry_* group.

    Used by the host to diagnose files that load with no usable entries
    (the entry combo would otherwise look mysteriously empty when the
    file contains only 1D-cut entries or polar-only data).

    Iteration uses the file's native ``f.keys()`` order — for files
    created with ``track_order=True`` (recent pygid conversions, the
    GUI's own appends) this is the original insertion order, so the
    Entry combo lists scans in the order they were collected. Files
    written without ``track_order`` fall back to HDF5's default
    alphanumeric order, which matches the previous behaviour.
    """
    out: dict[str, str | None] = {}
    with h5py.File(file_path, "r") as f:
        for name in f.keys():
            if not is_entry_group_name(name):
                continue
            data = f.get(f"{name}/data")
            if data is None:
                out[name] = None
                continue
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            out[name] = signal if isinstance(signal, str) else None
    return out


def list_pygid_incompatible_top_level(file_path: Path) -> list[str]:
    """Return top-level group names that would crash ``pygid.NexusFile``.

    pygid's ``NexusFile.read_structure`` iterates *every* top-level
    group with ``for entry in root`` and unconditionally indexes
    ``root[f"/{entry}/data"]`` (see
    ``pygid/nexus_reader.py::get_entry_type``) so HDF5 refuses
    to open with ``KeyError: "object 'data' doesn't exist"`` the
    moment any top-level group lacks a ``data`` child. This trips on:

      - raw detector files whose layout is ``/entry/data0/image``
        rather than ``/entry/data/...`` (the user "opened raw as
        NeXus" mistake), and
      - mixed files where an extra metadata / log / calibration
        group was added next to the entries.

    A group counts as incompatible when either ``/<name>/data`` is
    missing or its ``signal`` attribute names a dataset that isn't
    actually present under ``/<name>/data``. The check mirrors the
    two lookups pygid performs in ``get_entry_type``: the ``/data``
    open and the ``/data/<signal>`` shape probe.
    """
    bad: list[str] = []
    with h5py.File(file_path, "r") as f:
        for name in f.keys():
            obj = f.get(name)
            if not isinstance(obj, h5py.Group):
                bad.append(name)
                continue
            data = obj.get("data")
            if not isinstance(data, h5py.Group):
                bad.append(name)
                continue
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            if not isinstance(signal, str) or signal not in data:
                bad.append(name)
    return bad


def load_entry(file_path: Path, entry: str) -> EntryStack:
    """Open one entry as a lazily-readable stack.

    Returns an ``EntryStack`` whose ``image_stack`` is a tiny
    ``_LazyImageStack`` wrapper rather than a fully materialised numpy
    array. The wrapper holds a reference to a ``FrameSource`` that owns
    a long-lived ``h5py.File`` handle; per-frame reads happen on
    demand via ``FrameSource.get_cartesian(i)``. Axes ``q_xy`` / ``q_z``
    are still loaded eagerly (small, constant per entry).

    This switches the GUI from "load 100 GB to RAM" to "stream one
    frame at a time" without changing the public ``EntryStack`` shape.
    Callers that previously did ``stack.image_stack[i]`` keep working;
    callers that try to treat ``image_stack`` as a numpy ndarray
    (``.view()``, ``.copy()``, ``np.asarray(stack.image_stack)``) will
    fail loudly — by design, so accidental whole-stack reads surface
    in review rather than silently materialising 100 GB.
    """
    source = FrameSource(file_path=Path(file_path), entry=entry)
    source.acquire()
    lazy = _LazyImageStack(source)
    return EntryStack(
        image_stack=lazy,  # type: ignore[arg-type]  # duck-types as ndarray
        q_xy=source.q_xy,
        q_z=source.q_z,
    )


class FrameSource:
    """Owns one long-lived ``h5py.File`` handle and per-frame LRU caches.

    Backs an entry's image stack without ever pulling the full 3D array
    into memory. Cartesian frames come straight from the h5py dataset
    (chunked reads are cheap because pygid writes with ``chunks=True``).
    Polar frames are computed on demand via
    ``polar.cartesian_to_polar`` and cached separately. Axes
    (``q_xy``, ``q_z``, polar radius / angle) are read once on
    ``acquire()`` and cached as small numpy arrays.

    Lifecycle is coordinated with the silx-detach/reattach dance that
    wraps every file write: ``release()`` closes the handle and clears
    the caches; ``acquire()`` reopens. The viewer calls these via
    ``GIWAXSImageViewer.release_frame_source`` /
    ``acquire_frame_source`` so the same FrameSource instance can
    survive across pipeline runs without rebuilding from scratch.
    """

    def __init__(self, file_path: Path, entry: str) -> None:
        self._file_path = Path(file_path)
        self._entry = entry

        # State populated on acquire(); cleared on release().
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._q_xy: np.ndarray | None = None
        self._q_z: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None
        self._dtype: np.dtype | None = None

        # LRU caches. Sized on first acquire once the frame shape is known.
        # OrderedDict gives us move-to-end on hit + popitem(last=False) for
        # eviction, no external dependency.
        self._cart_lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._polar_lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cart_lru_max: int = FRAME_LRU_MIN
        self._polar_lru_max: int = max(FRAME_LRU_MIN // 2, 2)

        # Polar grid axes (computed once on first polar request and reused;
        # they're functions of q_xy / q_z only, constant across frames).
        self._polar_radius: np.ndarray | None = None
        self._polar_angle: np.ndarray | None = None

    # -- Lifecycle ----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def acquire(self) -> None:
        """Open the h5py file and load axes + shape.

        Idempotent: a second acquire on an already-open FrameSource is
        a no-op. Polar axes survive across release/acquire cycles
        because they're derived purely from q_xy / q_z, not the file.
        """
        if self._file is not None:
            return
        self._file = h5py.File(self._file_path, "r")
        try:
            self._dataset = self._file[f"{self._entry}/{IMG_REL}"]
            self._q_xy = np.asarray(
                self._file[f"{self._entry}/{QXY_REL}"][()], dtype=float
            )
            self._q_z = np.asarray(
                self._file[f"{self._entry}/{QZ_REL}"][()], dtype=float
            )
            self._shape = tuple(self._dataset.shape)  # type: ignore[assignment]
            self._dtype = self._dataset.dtype
            # Now that we know the frame size, size the Cartesian LRU so
            # `cart_lru_max * frame_bytes` lands near half the target.
            frame_bytes = int(self._shape[1]) * int(self._shape[2]) * int(
                np.dtype(self._dtype).itemsize
            )
            if frame_bytes > 0:
                half = FRAME_LRU_BYTES_TARGET // 2
                self._cart_lru_max = max(
                    FRAME_LRU_MIN, min(FRAME_LRU_MAX, half // frame_bytes)
                )
                # Polar grid is denser than Cartesian for the typical
                # (n_radius=1024, n_angle=512) defaults (= ``524288``
                # samples per frame, matching pygidfit's pipeline
                # polar grid) — size the polar LRU at half the
                # Cartesian count so total stays under the target.
                self._polar_lru_max = max(
                    FRAME_LRU_MIN // 2, self._cart_lru_max // 2
                )
        except Exception:
            # If anything failed mid-acquire, leave the FrameSource in a
            # clean closed state so a retry doesn't trip on partial state.
            self._file.close()
            self._file = None
            self._dataset = None
            raise

    def release(self) -> None:
        """Close the h5py file and drop all per-frame caches.

        Called by ``GIWAXSImageViewer.release_frame_source`` before any
        write path that needs r+ on the same file (pipeline runs, ROI
        commit, Add-to-fitted, clear-peaks). After release, calls to
        ``get_cartesian`` / ``get_polar`` will lazily re-acquire (via
        the host's reattach pairing); a manual ``acquire()`` is the
        normal way to bring the source back online.
        """
        self._cart_lru.clear()
        self._polar_lru.clear()
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.debug("suppressed exception in FrameSource.release", exc_info=True)
                pass
        self._file = None
        self._dataset = None

    def relocate(self, new_path: Path) -> None:
        """Point the FrameSource at a new filesystem path.

        Used by Save As: the temp file is renamed to match the new
        basename, so the FrameSource's stored ``_file_path`` would
        otherwise become stale and the next ``acquire()`` would fail.
        Caller is responsible for ordering the rename and the
        release/acquire calls correctly — typically
        ``release()`` → rename → ``relocate(new)`` → ``acquire()``.
        """
        self._file_path = Path(new_path)

    # -- Shape + axes -------------------------------------------------------

    @property
    def n_frames(self) -> int:
        if self._shape is None:
            return 0
        return int(self._shape[0])

    @property
    def q_xy(self) -> np.ndarray:
        if self._q_xy is None:
            raise RuntimeError("FrameSource not acquired")
        return self._q_xy

    @property
    def q_z(self) -> np.ndarray:
        if self._q_z is None:
            raise RuntimeError("FrameSource not acquired")
        return self._q_z

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(n_qz, n_qxy) — the shape of one Cartesian frame on disk."""
        if self._shape is None:
            raise RuntimeError("FrameSource not acquired")
        return int(self._shape[1]), int(self._shape[2])

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            raise RuntimeError("FrameSource not acquired")
        return self._dtype

    @property
    def cart_lru_size(self) -> int:
        return self._cart_lru_max

    @property
    def polar_lru_size(self) -> int:
        return self._polar_lru_max

    # -- Hot path -----------------------------------------------------------

    def get_cartesian(self, i: int) -> np.ndarray:
        """Read frame ``i`` from disk, returning a numpy 2D array.

        LRU-cached: a recently-viewed frame is returned without a disk
        round-trip. Caller receives the cached array reference — do not
        mutate the returned array, the LRU stores it as-is.
        """
        if self._dataset is None:
            raise RuntimeError("FrameSource not acquired")
        if i in self._cart_lru:
            self._cart_lru.move_to_end(i)
            return self._cart_lru[i]
        frame = np.asarray(self._dataset[i])
        self._cart_lru[i] = frame
        while len(self._cart_lru) > self._cart_lru_max:
            self._cart_lru.popitem(last=False)
        return frame

    def get_polar(self, i: int) -> np.ndarray:
        """Polar-resample frame ``i`` on demand.

        The polar grid (radius / angle) is computed once on first call
        and reused for every subsequent frame, since it's derived from
        ``q_xy`` / ``q_z`` only. Per-frame polar arrays are LRU-cached
        independently from the Cartesian cache.
        """
        if self._dataset is None:
            raise RuntimeError("FrameSource not acquired")
        if i in self._polar_lru:
            self._polar_lru.move_to_end(i)
            return self._polar_lru[i]
        # Local import to avoid a circular dep — polar.py imports
        # numpy/scipy only, but importing it at module top would make
        # any test of file_model pull the polar transform too.
        from mlgidlab.polar import cartesian_to_polar

        cart = self.get_cartesian(i)
        polar = cartesian_to_polar(cart, self.q_xy, self.q_z)
        if self._polar_radius is None or self._polar_angle is None:
            self._polar_radius = polar.radius
            self._polar_angle = polar.angle
        self._polar_lru[i] = polar.image
        while len(self._polar_lru) > self._polar_lru_max:
            self._polar_lru.popitem(last=False)
        return polar.image

    def polar_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(radius, angle)`` for the polar grid.

        Triggers a single polar resampling of frame 0 if the axes
        haven't been computed yet — needed by the viewer's polar
        axis labels + cursor lookup.
        """
        if self._polar_radius is None or self._polar_angle is None:
            self.get_polar(0)
        assert self._polar_radius is not None and self._polar_angle is not None
        return self._polar_radius, self._polar_angle

    # -- LRU deposit (used by the background prefetch worker) -------------

    def warm_cartesian(self, i: int, frame: np.ndarray) -> None:
        """Deposit a pre-read Cartesian frame into the LRU.

        Called only from the main thread, via MainWindow's
        prefetched-signal slot. The background prefetch worker has
        its own independent ``h5py.File`` handle and does the disk
        read on a worker thread; this method just deposits the
        result so subsequent ``get_cartesian(i)`` calls hit cache.

        No-op when the FrameSource is currently released (the
        worker's signal might still be in flight after MainWindow
        detached silx for a pipeline run) or when the index is
        out of range (entry shape changed between emit and slot).
        """
        if self._dataset is None:
            return
        if not (0 <= i < self.n_frames):
            return
        self._cart_lru[i] = frame
        self._cart_lru.move_to_end(i)
        while len(self._cart_lru) > self._cart_lru_max:
            self._cart_lru.popitem(last=False)

    def warm_polar(
        self,
        i: int,
        polar_frame: np.ndarray,
        radius: np.ndarray | None = None,
        angle: np.ndarray | None = None,
    ) -> None:
        """Deposit a pre-computed polar frame into the LRU.

        ``radius`` / ``angle`` are stashed on the FrameSource only
        when our own axes haven't been computed yet — that way the
        worker's first emit primes ``polar_axes()`` and saves the
        viewer from a duplicate frame-0 resample when entering
        polar mode for the first time. Subsequent emits ignore the
        axis args (they're identical anyway).
        """
        if self._dataset is None:
            return
        if not (0 <= i < self.n_frames):
            return
        self._polar_lru[i] = polar_frame
        self._polar_lru.move_to_end(i)
        while len(self._polar_lru) > self._polar_lru_max:
            self._polar_lru.popitem(last=False)
        if self._polar_radius is None and radius is not None:
            self._polar_radius = np.asarray(radius)
        if self._polar_angle is None and angle is not None:
            self._polar_angle = np.asarray(angle)


class _LazyImageStack:
    """Thin shim presenting a ``(n_frames, n_qz, n_qxy)`` indexable view.

    Exposes only the surface mlgidLAB actually uses: ``__getitem__``,
    ``shape``, ``ndim``, ``dtype``. Deliberately NOT a numpy ndarray
    subclass and deliberately NOT supporting ``view`` / ``copy`` /
    ``__array__`` — any caller that tries to treat it as a real array
    fails loudly, which catches accidental whole-stack reads in review.
    """

    __slots__ = ("_source",)

    def __init__(self, source: FrameSource) -> None:
        self._source = source

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def shape(self) -> tuple[int, int, int]:
        n_qz, n_qxy = self._source.frame_shape
        return (self._source.n_frames, n_qz, n_qxy)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    def __len__(self) -> int:
        return self._source.n_frames

    def __getitem__(self, key):
        """Frame index or (frame, row, col) tuple index.

        - ``stack[i]`` returns the 2D Cartesian frame (n_qz, n_qxy).
        - ``stack[frame, r, c]`` returns one pixel — used by the
          cursor-readout's intensity lookup.

        Slices and ellipsis intentionally raise — we want to know
        about callers that try to grab subsets of the stack so they
        can be moved onto a FrameSource access pattern.
        """
        if isinstance(key, (int, np.integer)):
            return self._source.get_cartesian(int(key))
        if isinstance(key, tuple) and len(key) == 3:
            frame, r, c = key
            if (
                isinstance(frame, (int, np.integer))
                and isinstance(r, (int, np.integer))
                and isinstance(c, (int, np.integer))
            ):
                return self._source.get_cartesian(int(frame))[int(r), int(c)]
        raise TypeError(
            f"_LazyImageStack supports stack[i] or stack[frame, r, c] "
            f"only; got {key!r}."
        )


class _LazyPolarStack:
    """``(n_frames, n_radius, n_angle)`` indexable view on polar frames.

    Mirrors ``_LazyImageStack`` but routes to ``FrameSource.get_polar``.
    Supports two indexing forms used by the GUI:

    - ``stack[i]`` — return a polar frame (``profile_viewer.py:341``)
    - ``stack[frame, r, a]`` — single-pixel lookup used by the cursor
      readout (``image_viewer._compute_cursor_info``)
    """

    __slots__ = ("_source",)

    def __init__(self, source: FrameSource) -> None:
        self._source = source

    @property
    def shape(self) -> tuple[int, int, int]:
        radius, angle = self._source.polar_axes()
        return (self._source.n_frames, int(radius.size), int(angle.size))

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return self._source.n_frames

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._source.get_polar(int(key))
        if isinstance(key, tuple) and len(key) == 3:
            frame, r, a = key
            if (
                isinstance(frame, (int, np.integer))
                and isinstance(r, (int, np.integer))
                and isinstance(a, (int, np.integer))
            ):
                return self._source.get_polar(int(frame))[int(r), int(a)]
        raise TypeError(
            f"_LazyPolarStack supports stack[i] or stack[frame, r, a] "
            f"only; got {key!r}."
        )


# Raw-detector dataset enumeration & loading. These are used only by the
# Conversion (raw-mode) workflow — no overlap with the converted-NeXus
# readers above.

# Minimum H/W in pixels for a dataset to count as a detector image.
# Filters out coordinate axes (1D), small lookup tables, and per-frame
# scalar arrays that h5py would otherwise misclassify as 3D.
RAW_MIN_DETECTOR_HW = 32


@dataclass
class RawEntry:
    """One candidate raw-detector dataset within an HDF5 file.

    ``dataset_path`` is the absolute path inside the HDF5 file (without
    the leading slash) — pygid's ``DataLoader`` accepts this form
    directly as its ``dataset`` kwarg.
    """

    file_path: Path
    dataset_path: str
    shape: tuple[int, int, int]
    dtype: str

    @property
    def label(self) -> str:
        """Human-readable identifier for the entry combo / log lines."""
        return f"{self.file_path.name}::{self.dataset_path}"


def list_raw_entries(file_path: Path) -> list[RawEntry]:
    """Walk a raw HDF5 file and return every 3D detector-image candidate.

    A dataset qualifies when it is 3D ``(N, H, W)`` with both spatial
    dimensions ≥ ``RAW_MIN_DETECTOR_HW`` and a numeric dtype. The walker
    recurses through every group so unusual beamline layouts work too.
    Datasets are sorted by path for stable UI ordering.

    Reads only metadata (shape + dtype) — pixel data is not loaded
    until ``load_raw_dataset`` is called for a specific entry.
    """
    out: list[RawEntry] = []
    with h5py.File(file_path, "r") as f:
        def visit(_name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim != 3:
                return
            n, h, w = obj.shape
            if h < RAW_MIN_DETECTOR_HW or w < RAW_MIN_DETECTOR_HW:
                return
            if obj.dtype.kind not in ("i", "u", "f"):
                return
            out.append(
                RawEntry(
                    file_path=file_path,
                    dataset_path=obj.name.lstrip("/"),
                    shape=(int(n), int(h), int(w)),
                    dtype=str(obj.dtype),
                )
            )
        f.visititems(visit)
    out.sort(key=lambda e: e.dataset_path)
    return out


def load_raw_dataset(entry: RawEntry) -> np.ndarray:
    """Read the full 3D pixel data for one ``RawEntry``.

    Returns a contiguous float32 array; integer detector data is
    upcast so pyqtgraph's intensity scaling and the same percentile-
    based level helpers used for converted data work uniformly.
    """
    with h5py.File(entry.file_path, "r") as f:
        ds = f[entry.dataset_path]
        arr = np.asarray(ds[()])
    if arr.dtype.kind in ("i", "u"):
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def load_peaks(
    file_path: Path, entry: str, frame: int
) -> dict[PeakKind, PeakTable | None]:
    """Read detected/fitted peak tables for one frame; missing tables → None."""
    frame_key = FRAME_KEY_FMT.format(frame)
    out: dict[PeakKind, PeakTable | None] = {"detected": None, "fitted": None}
    with h5py.File(file_path, "r") as f:
        group_path = f"{entry}/{ANALYSIS_REL}/{frame_key}"
        if group_path not in f:
            return out
        group = f[group_path]
        if "detected_peaks" in group:
            out["detected"] = PeakTable.from_dataset(group["detected_peaks"])
        if "fitted_peaks" in group:
            out["fitted"] = PeakTable.from_dataset(group["fitted_peaks"])
    return out


@dataclass
class MatchedStructure:
    """One matched crystal structure within a frame.

    Each ``matched_*`` NeXus dataset can contain several rows — multi-phase
    solutions — and each row corresponds to one ``MatchedStructure``. The
    ``peaks`` table is a subset of the frame's ``fitted_peaks`` taken at the
    indices listed in ``peak_list``, so the existing peak-rendering helpers
    can draw matched peaks without any special-casing.
    """

    solution_field: str  # e.g. "matched_segments_0000"
    local_idx: int       # row within ``solution_field``
    cif: str             # CIF basename (no extension)
    h: int
    k: int
    l: int
    probability: float
    peaks: PeakTable
    # Indices into the frame's ``fitted_peaks`` table that produced ``peaks``.
    # Kept so matched overlays can be re-derived after an in-memory edit of a
    # fitted peak without a second file read.
    peak_list: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def unique_id(self) -> str:
        """Stable handle within one frame."""
        return f"{self.solution_field}/{self.local_idx}"

    @property
    def label(self) -> str:
        ori = "rand" if (self.h, self.k, self.l) == (0, 0, 0) else (
            f"({self.h}{self.k}{self.l})"
        )
        return f"{self.cif} {ori}  p={self.probability:.2f}"


def load_matched_peaks(
    file_path: Path,
    entry: str,
    frame: int,
    fitted_peaks: PeakTable | None,
) -> list[MatchedStructure]:
    """Read all ``matched_*`` solutions for a frame.

    Each row of each ``matched_*`` dataset becomes one ``MatchedStructure``.
    Returns an empty list when the file has no matches for this frame, or when
    ``fitted_peaks`` is None — peak_list indices reference the frame's
    fitted_peaks, so without them the matched peaks have no geometry.
    """
    if fitted_peaks is None or len(fitted_peaks) == 0:
        return []
    frame_key = FRAME_KEY_FMT.format(frame)
    n_fit = len(fitted_peaks)
    out: list[MatchedStructure] = []
    with h5py.File(file_path, "r") as f:
        group_path = f"{entry}/{ANALYSIS_REL}/{frame_key}"
        if group_path not in f:
            return out
        group = f[group_path]
        for name in sorted(group.keys()):
            if not name.startswith("matched_"):
                continue
            ds = group[name]
            arr = ds[()]
            for i in range(len(arr)):
                cif_raw = arr["CIF"][i]
                cif_str = (
                    cif_raw.decode("utf-8", errors="replace")
                    if isinstance(cif_raw, bytes)
                    else str(cif_raw)
                )
                # Strip the .cif extension for display.
                if cif_str.lower().endswith(".cif"):
                    cif_str = cif_str[:-4]
                idx = np.asarray(arr["peak_list"][i], dtype=int)
                # Tolerate stale matches: drop indices that no longer exist
                # in fitted_peaks (e.g. fitting was re-run after matching).
                # Physics-audit F-04 is closed at the *write* side — the
                # ``pipeline.execute`` pre-flight clears matched_* on every
                # entry/frame before run_fitting rewrites fitted_peaks, so
                # this clamp should never fire on files produced by the
                # current GUI. Kept as defence-in-depth for files written
                # by older GUI builds (or external tooling) where matches
                # may still reference stale fitted positions.
                idx = idx[(idx >= 0) & (idx < n_fit)]
                if idx.size == 0:
                    continue
                subset = PeakTable(
                    q_xy=fitted_peaks.q_xy[idx],
                    q_z=fitted_peaks.q_z[idx],
                    angle=fitted_peaks.angle[idx],
                    radius=fitted_peaks.radius[idx],
                    angle_width=fitted_peaks.angle_width[idx],
                    radius_width=fitted_peaks.radius_width[idx],
                    is_ring=fitted_peaks.is_ring[idx],
                    ids=fitted_peaks.ids[idx],
                    score=fitted_peaks.score[idx],
                    amplitude=fitted_peaks.amplitude[idx],
                )
                out.append(
                    MatchedStructure(
                        solution_field=name,
                        local_idx=i,
                        cif=cif_str,
                        h=int(arr["h"][i]),
                        k=int(arr["k"][i]),
                        l=int(arr["l"][i]),
                        probability=float(arr["probability"][i]),
                        peaks=subset,
                        peak_list=idx,
                    )
                )
    return out


def add_fitted_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
    amplitude: float,
    is_ring: bool = False,
    theta: float = 0.0,
    score: float = 0.0,
    A: float = 0.0,
    B: float = 0.0,
    C: float = 0.0,
    is_cut_qz: bool = False,
    is_cut_qxy: bool = False,
    visibility: int = 0,
) -> int:
    """Append a new row to the frame's ``fitted_peaks`` dataset.

    Mirrors mlgidbase's row layout (see pygid.datasaver pygid_results_dtype):
    polar geometry + amplitude + 2D-Gaussian shape params (A, B, C, theta)
    + flags + an auto-assigned id. q_xy/q_z are recomputed from polar.

    Caller-supplied fields drive only the meaningful values; the 2D-shape
    params (A/B/C/theta) and score default to zero because a 1D-fit-derived
    row carries no 2D context. Returns the new peak's ``id``.
    """
    frame_key = FRAME_KEY_FMT.format(frame)
    with h5py.File(file_path, "r+") as f:
        ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/fitted_peaks"
        if ds_path not in f:
            raise KeyError(
                f"fitted_peaks dataset missing at {ds_path} — run fitting "
                "at least once on this frame before adding manual fitted rows."
            )
        ds = f[ds_path]
        arr = ds[()]
        new_id = int(arr["id"].max() + 1) if len(arr) > 0 else 0
        # Build the new row by name to stay schema-resilient if the dtype
        # ever gains/loses a field.
        new_row = np.zeros(1, dtype=arr.dtype)
        from mlgidlab.polar import polar_to_qxyz
        q_xy_val, q_z_val = polar_to_qxyz(radius, angle)
        fields = {
            "amplitude": amplitude,
            "angle": angle,
            "angle_width": angle_width,
            "radius": radius,
            "radius_width": radius_width,
            "q_z": q_z_val,
            "q_xy": q_xy_val,
            "theta": theta,
            "score": score,
            "A": A,
            "B": B,
            "C": C,
            "is_ring": is_ring,
            "is_cut_qz": is_cut_qz,
            "is_cut_qxy": is_cut_qxy,
            "visibility": visibility,
            "id": new_id,
        }
        for name, value in fields.items():
            if name in arr.dtype.names:
                new_row[name] = value
        new_arr = np.concatenate([arr, new_row])
        # pygid creates fixed-shape datasets (maxshape == shape), so resize
        # is unavailable — delete and recreate, preserving any attrs.
        attrs = dict(ds.attrs)
        del f[ds_path]
        new_ds = f.create_dataset(ds_path, data=new_arr)
        for k, v in attrs.items():
            new_ds.attrs[k] = v
        return new_id


def normalize_for_pygid(file_path: Path) -> dict[str, list[str]]:
    """Patch every entry so pygid's per-frame writers / readers behave.

    Two normalizations are applied to the temp copy (the original file
    is never touched):

    1. **0-D ``angle_of_incidence``** → 1-D length-``n_frames`` array.
       ``pygid.nexus_reader.get_ai`` indexes by ``frame_num``; a 0-D
       scalar dataset (some writers do this for single-frame data)
       raises ``IndexError: invalid index to scalar variable`` and
       blocks every pipeline op before it starts.

    2. **Missing per-frame analysis groups** → empty groups created.
       ``pygid.datasaver._save_img_container_detect`` (called from
       ``mlgidbase.save_detect`` / ``save_fit`` / matched-write paths)
       assumes ``data/analysis/frame{N:05d}`` already exists. Files
       written for a single frame only ship with ``frame00000``; the
       per-frame detection / fitting code then errors with
       ``KeyError: 'frameXXXXX' doesn't exist`` when run on any
       higher-numbered frame. We pre-create the missing groups with
       the same NX attrs pygid uses in the bulk save path.

    Returns a dict ``{"angle": [...], "frames": [...]}`` describing
    which entries were patched in each pass (for logging). Idempotent
    on already-normalized files.
    """
    patched_angle: list[str] = []
    patched_frames: list[str] = []
    with h5py.File(file_path, "r+") as f:
        for entry_name in list(f.keys()):
            if not is_entry_group_name(entry_name):
                continue
            entry = f[entry_name]
            # Look up n_frames from the entry's primary signal dataset.
            data = entry.get("data")
            if data is None:
                continue
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            if not signal or signal not in data:
                continue
            n_frames = int(data[signal].shape[0])

            # --- 1. angle_of_incidence: scalar → 1-D
            ai_path = "instrument/angle_of_incidence"
            if ai_path in entry:
                ai_ds = entry[ai_path]
                # A length-mismatched 1-D field is intentionally not
                # "fixed" because the per-frame values may legitimately
                # differ (true per-frame metadata) and broadcasting
                # would silently lie.
                if ai_ds.ndim == 0:
                    scalar_value = float(ai_ds[()])
                    new_arr = np.full((n_frames,), scalar_value, dtype=np.float64)
                    attrs = dict(ai_ds.attrs)
                    del entry[ai_path]
                    new_ds = entry.create_dataset(ai_path, data=new_arr)
                    for k, v in attrs.items():
                        new_ds.attrs[k] = v
                    patched_angle.append(entry_name)

            # --- 2. analysis/frameXXXXX groups: pre-create missing ones
            analysis = entry.require_group("data/analysis")
            created_any = False
            for i in range(n_frames):
                group_name = FRAME_KEY_FMT.format(i)
                if group_name not in analysis:
                    g = analysis.create_group(group_name)
                    # Match the NX attrs pygid sets in its bulk save path
                    # so downstream readers don't choke on stricter
                    # NeXus validators.
                    g.attrs["NX_class"] = "NXparameters"
                    g.attrs["EX_required"] = "true"
                    created_any = True
            if created_any:
                patched_frames.append(entry_name)
    return {"angle": patched_angle, "frames": patched_frames}


def delete_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    kind: str,
    peak_id: int,
) -> int:
    """Delete a single row from one kind's ``<kind>_peaks`` dataset.

    ``kind`` is ``"detected"`` or ``"fitted"``. Removes the row whose
    ``id`` field matches ``peak_id`` and writes the surviving rows
    back; the dataset is recreated at the new length, preserving
    dtype and attrs. Returns the number of rows removed (0 if no
    row had the requested id, in which case the file is untouched).

    **Does not cascade across kinds.** mlgidbase's
    ``_delete_peak_single_frame`` removes the row from detected,
    fitted, and matched_* in one shot (and reindexes ids on each).
    This helper is the opposite: deletes only the kind you ask for,
    leaves the other kinds alone, and **does not reindex**
    ``id`` — the remaining ids stay stable so a delete operation
    can't silently shift external references. Callers that need a
    cascade (e.g. fitted-delete invalidates matched ``peak_list``
    integer indices into ``fitted_peaks``) must invoke
    ``clear_peaks(..., "matched", frame=frame)`` separately.

    ``raise ValueError`` for unsupported ``kind`` (matched can't be
    row-deleted through here because ``matched_*`` lives across
    multiple per-solution datasets; use ``clear_peaks(...,
    "matched", ...)`` to wipe them or write a dedicated helper).
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(
            f"delete_peak_row: kind must be detected or fitted, got {kind!r}"
        )
    ds_name = f"{kind}_peaks"
    frame_key = FRAME_KEY_FMT.format(int(frame))
    with h5py.File(file_path, "r+") as f:
        ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/{ds_name}"
        if ds_path not in f:
            return 0
        ds = f[ds_path]
        arr = ds[()]
        keep = arr["id"] != int(peak_id)
        if keep.all():
            return 0
        new_arr = arr[keep]
        # Preserve dtype + attrs by recreating the dataset, mirroring
        # the in-place delete + recreate pattern used by clear_peaks
        # and add_fitted_peak_row. pygid creates these as fixed-shape
        # datasets so .resize is unavailable.
        attrs = dict(ds.attrs)
        parent = ds.parent
        del parent[ds_name]
        new_ds = parent.create_dataset(
            ds_name, data=new_arr,
        )
        for k, v in attrs.items():
            new_ds.attrs[k] = v
        # ``fitted_peaks`` row ordering changed → matched_* peak_list
        # integer indices into this frame's fitted_peaks may now be
        # stale. The caller is expected to invoke
        # ``clear_peaks(..., "matched", frame=frame)`` after this
        # helper for the fitted case; we don't do it here so the
        # contract stays "this helper touches exactly one kind".
        return int(arr.shape[0] - new_arr.shape[0])


def clear_peaks(
    file_path: Path,
    entry: str,
    kind: str,
    frame: int | None = None,
) -> int:
    """Empty every ``<kind>_peaks`` dataset under ``entry/data/analysis/``.

    ``kind`` is one of:

    - ``"detected"``   — empties ``detected_peaks`` per frame.
    - ``"fitted"``     — empties ``fitted_peaks`` and ``fitted_peaks_errors``
                         per frame (the two are paired by id).
    - ``"matched"``    — empties every ``matched_*`` dataset per frame.

    ``frame`` restricts the wipe to a single ``frameNNNNN`` group when
    given; the default ``None`` clears every frame in the entry. The
    frame-scoped form is used by the Tools → Reset → Active-frame
    action; pipeline-level cascades pass ``None``.

    Datasets are recreated empty (shape ``(0,)``) preserving dtype + attrs,
    because pygid creates them as fixed-shape datasets so ``.resize`` is
    unavailable. Returns the number of rows removed across all frames.

    Manual peaks live in memory only — clear them via the viewer.
    """
    if kind not in ("detected", "fitted", "matched"):
        raise ValueError(
            f"clear_peaks: kind must be detected/fitted/matched, got {kind!r}"
        )
    removed = 0
    with h5py.File(file_path, "r+") as f:
        ana_path = f"{entry}/{ANALYSIS_REL}"
        if ana_path not in f:
            return 0
        ana_group = f[ana_path]
        if frame is None:
            frame_names = list(ana_group.keys())
        else:
            target = FRAME_KEY_FMT.format(frame)
            frame_names = [target] if target in ana_group else []
        for frame_name in frame_names:
            frame_group = ana_group[frame_name]
            if not isinstance(frame_group, h5py.Group):
                continue
            if kind == "matched":
                for ds_name in list(frame_group.keys()):
                    if ds_name.startswith("matched_"):
                        removed += _empty_dataset_in_place(frame_group, ds_name)
            elif kind == "fitted":
                for ds_name in ("fitted_peaks", "fitted_peaks_errors"):
                    if ds_name in frame_group:
                        removed += _empty_dataset_in_place(frame_group, ds_name)
            else:
                ds_name = "detected_peaks"
                if ds_name in frame_group:
                    removed += _empty_dataset_in_place(frame_group, ds_name)
    return removed


def _iter_frame_keys(
    ana_group: h5py.Group, frame: int | None
) -> Iterator[tuple[str, int]]:
    """Yield ``(frame_group_name, frame_index)`` pairs in the entry.

    When ``frame`` is None every ``frameNNNNN`` group under the entry's
    analysis group is yielded; otherwise just the matching one (if it
    exists). Frame index is parsed from the trailing digits of the
    group name so it matches the value the viewer reports for
    ``current_frame`` and the ``frame_num`` column the rest of the
    pipeline writes into peak rows.
    """
    if frame is None:
        for name in ana_group.keys():
            if not name.startswith("frame"):
                continue
            try:
                idx = int(name[len("frame"):])
            except ValueError:
                continue
            yield name, idx
    else:
        target = FRAME_KEY_FMT.format(frame)
        if target in ana_group:
            yield target, int(frame)


def _csv_value(v):
    """Coerce a structured-array element into something the csv module
    can serialise without surprises (bytes → utf-8 str, numpy scalars
    pass through as their repr)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def export_peaks_csv(
    file_path: Path,
    targets: list[tuple[str, int | None]],
    kind: str,
    out_path: Path,
) -> int:
    """Write detected or fitted peaks to ``out_path`` as CSV.

    ``targets`` is a list of ``(entry, frame_or_None)`` tuples; pass
    ``frame=None`` to dump every frame in the entry. ``kind`` is
    ``"detected"`` or ``"fitted"``; for fitted, the ``fitted_peaks_errors``
    sibling dataset is joined per row and its fields appear as
    ``<name>_err`` columns. Returns the row count written.

    The header is the union of every dtype field seen in the scope
    (in first-encounter order), prefixed with ``entry`` and
    ``frame_num``. Missing fields on a row become empty strings —
    the writer uses ``DictWriter`` so heterogeneity across entries
    is tolerated.
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(f"export_peaks_csv: kind must be detected/fitted, got {kind!r}")
    ds_name = "detected_peaks" if kind == "detected" else "fitted_peaks"
    err_ds_name = "fitted_peaks_errors" if kind == "fitted" else None

    rows: list[dict] = []
    fieldnames: list[str] = ["entry", "frame_num"]
    seen: set[str] = set(fieldnames)

    with h5py.File(file_path, "r") as f:
        for entry, frame in targets:
            ana_path = f"{entry}/{ANALYSIS_REL}"
            if ana_path not in f:
                continue
            ana_group = f[ana_path]
            for frame_name, frame_idx in _iter_frame_keys(ana_group, frame):
                fg = ana_group[frame_name]
                if ds_name not in fg:
                    continue
                arr = fg[ds_name][()]
                if len(arr) == 0:
                    continue
                err_arr = None
                if err_ds_name is not None and err_ds_name in fg:
                    err_arr = fg[err_ds_name][()]
                for i in range(len(arr)):
                    row = {"entry": entry, "frame_num": frame_idx}
                    for name in arr.dtype.names:
                        row[name] = _csv_value(arr[name][i])
                        if name not in seen:
                            fieldnames.append(name)
                            seen.add(name)
                    if err_arr is not None and i < len(err_arr):
                        for name in err_arr.dtype.names:
                            col = f"{name}_err"
                            row[col] = _csv_value(err_arr[name][i])
                            if col not in seen:
                                fieldnames.append(col)
                                seen.add(col)
                    rows.append(row)

    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def export_matched_csv(
    file_path: Path,
    targets: list[tuple[str, int | None]],
    out_path: Path,
) -> int:
    """Write matched solutions to ``out_path`` as CSV, one row per solution.

    Mirrors what silx shows in its Data tab for the ``matched_*``
    datasets: every solution gets a single row, with its referenced
    fitted peaks compacted into a ``peak_list`` cell formatted as
    ``[id_a, id_b, …]``. CIF is decoded and stripped of its ``.cif``
    suffix to match the Display dock; any extra dtype fields on the
    matched dataset are passed through verbatim so schema additions
    don't silently disappear from the export.

    Returns the row count written.
    """
    rows: list[dict] = []
    fieldnames: list[str] = [
        "entry", "frame_num", "solution_field", "local_idx",
        "cif", "h", "k", "l", "probability", "peak_list",
    ]
    seen: set[str] = set(fieldnames)
    extra_solution_fields: list[str] = []  # any dtype field beyond the well-known ones

    with h5py.File(file_path, "r") as f:
        for entry, frame in targets:
            ana_path = f"{entry}/{ANALYSIS_REL}"
            if ana_path not in f:
                continue
            ana_group = f[ana_path]
            for frame_name, frame_idx in _iter_frame_keys(ana_group, frame):
                fg = ana_group[frame_name]
                for sol_name in sorted(fg.keys()):
                    if not sol_name.startswith("matched_"):
                        continue
                    sol_arr = fg[sol_name][()]
                    if len(sol_arr) == 0:
                        continue
                    sol_dtype_names = sol_arr.dtype.names or ()
                    well_known = {"CIF", "h", "k", "l", "probability", "peak_list"}
                    for name in sol_dtype_names:
                        if name in well_known or name in seen:
                            continue
                        fieldnames.append(name)
                        seen.add(name)
                        extra_solution_fields.append(name)
                    for i in range(len(sol_arr)):
                        cif_raw = sol_arr["CIF"][i]
                        cif_str = (
                            cif_raw.decode("utf-8", errors="replace")
                            if isinstance(cif_raw, bytes) else str(cif_raw)
                        )
                        if cif_str.lower().endswith(".cif"):
                            cif_str = cif_str[:-4]
                        peak_list = np.atleast_1d(
                            np.asarray(sol_arr["peak_list"][i], dtype=int)
                        )
                        # Verbatim list of indices as silx shows them in
                        # the Data tab. Negative padding sentinels are
                        # filtered so the cell shows just the real
                        # peaks; users who need raw values can read the
                        # HDF5 dataset directly.
                        peak_ids = [int(x) for x in peak_list if int(x) >= 0]
                        row = {
                            "entry": entry,
                            "frame_num": frame_idx,
                            "solution_field": sol_name,
                            "local_idx": int(i),
                            "cif": cif_str,
                            "h": int(sol_arr["h"][i]),
                            "k": int(sol_arr["k"][i]),
                            "l": int(sol_arr["l"][i]),
                            "probability": float(sol_arr["probability"][i]),
                            "peak_list": "[" + ", ".join(str(x) for x in peak_ids) + "]",
                        }
                        for name in extra_solution_fields:
                            if name in sol_dtype_names:
                                row[name] = _csv_value(sol_arr[name][i])
                        rows.append(row)

    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _empty_dataset_in_place(parent: h5py.Group, name: str) -> int:
    """Replace ``parent[name]`` with an empty array of the same dtype, keep attrs.

    Returns the row count that was removed (for logging). Used only by
    ``clear_peaks`` — internal helper, not part of the public surface.
    """
    ds = parent[name]
    n = int(ds.shape[0]) if len(ds.shape) > 0 else 0
    attrs = dict(ds.attrs)
    empty = np.zeros(0, dtype=ds.dtype)
    del parent[name]
    new_ds = parent.create_dataset(name, data=empty)
    for k, v in attrs.items():
        new_ds.attrs[k] = v
    return n


def update_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    kind: str,
    peak_id: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
) -> None:
    """Mutate one row of detected_peaks/fitted_peaks in place, keyed by `id`.

    The id field is unique within a frame (mlgidbase reindexes on delete), so
    finding a row by id is unambiguous. Polar fields are written verbatim;
    q_xy/q_z are recomputed from the new (radius, angle).

    Raises KeyError when no row with ``peak_id`` exists — the caller should
    treat this as "the file moved on under us" and clear the undo stack.
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(f"update_peak_row only handles detected/fitted, not {kind!r}")
    ds_name = f"{kind}_peaks"
    frame_key = FRAME_KEY_FMT.format(frame)
    with h5py.File(file_path, "r+") as f:
        ds = f[f"{entry}/{ANALYSIS_REL}/{frame_key}/{ds_name}"]
        arr = ds[()]
        matches = np.where(arr["id"] == peak_id)[0]
        if matches.size == 0:
            raise KeyError(
                f"peak_id={peak_id} not found in {entry}/{frame_key}/{ds_name}"
            )
        idx = int(matches[0])
        arr["radius"][idx] = radius
        arr["radius_width"][idx] = radius_width
        arr["angle"][idx] = angle
        arr["angle_width"][idx] = angle_width
        from mlgidlab.polar import polar_to_qxyz
        arr["q_xy"][idx], arr["q_z"][idx] = polar_to_qxyz(radius, angle)
        ds[...] = arr
