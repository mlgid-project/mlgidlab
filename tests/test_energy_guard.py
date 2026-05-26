"""F-02 guard: ``_exp_params_from_nexus`` must raise a clear error
when the photon energy derived from
``instrument/monochromator/wavelength`` lies outside the plausible
1-200 keV X-ray range.

Two failure modes the audit calls out:
  * the wavelength datum is malformed (wrong units, e.g. Å instead
    of meters)
  * pygidsim silently changed its ``en`` contract from eV to keV,
    shrinking the computed value 1000x

Either way, the matching path must fail loud before mlgidmatch
builds a CIF pattern against a wrong wavelength and returns
plausible-but-bogus solutions.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

# Lazy import — pipeline imports pygidsim which we want to discover
# only if the env has it. The other tests in this dir already touch
# pipeline.execute so the import is known to work here.
from mlgidlab.pipeline import _EnergyOutOfRangeError, _exp_params_from_nexus


def _make_nexus_with_wavelength(path: Path, wavelength_m: float) -> Path:
    """Write a minimal valid NeXus file whose
    ``instrument/monochromator/wavelength`` is ``wavelength_m`` (in
    meters, as pygid expects)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        entry = f.create_group("entry_0000", track_order=True)
        d = entry.create_group("data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((2, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
        instr = entry.create_group("instrument", track_order=True)
        mono = instr.create_group("monochromator", track_order=True)
        mono.create_dataset("wavelength", data=np.array([wavelength_m]))
        instr.create_dataset("angle_of_incidence", data=np.array([0.3]))
    return path


def test_realistic_xray_wavelength_passes(tmp_path):
    """1 Å (≈ 12.4 keV) is the canonical synchrotron GIWAXS
    energy and must pass cleanly — confirms the guard does not
    over-fire on plausible inputs. ``ExpParameters`` stores the
    wavelength downstream of its ``en`` kwarg (en in eV converted
    to wavelength in Å for the rest of pygidsim's math), so we
    verify by round-tripping back to the seeded value."""
    path = _make_nexus_with_wavelength(tmp_path / "ok.h5", wavelength_m=1.0e-10)
    params = _exp_params_from_nexus(path, entry="entry_0000")
    # Round-trip should land within float-precision of 1 Å.
    assert params.wavelength == pytest.approx(1.0, rel=1e-3)


def test_pygidsim_kev_flip_trips_lower_bound(tmp_path):
    """Synthesise the divergence scenario: a wavelength of 12.4 nm
    (= 100 eV photon, way below the 1 keV floor) trips the guard.
    Models what would happen if pygidsim silently flipped the
    ``en`` unit to keV — the computed value would shrink by 1000x
    and land below the lower bound."""
    path = _make_nexus_with_wavelength(
        tmp_path / "soft.h5", wavelength_m=1.24e-8
    )
    with pytest.raises(_EnergyOutOfRangeError, match="F-02"):
        _exp_params_from_nexus(path, entry="entry_0000")


def test_angstrom_units_mistake_trips_lower_bound(tmp_path):
    """If a writer stores wavelength in Å rather than meters (a
    1Å wavelength written as the literal 1.0), the derived energy
    is ~12.4 keV * 1e-10 ≈ 1.24e-6 eV. Guard must catch this."""
    path = _make_nexus_with_wavelength(tmp_path / "angstrom.h5", wavelength_m=1.0)
    with pytest.raises(_EnergyOutOfRangeError, match="F-02"):
        _exp_params_from_nexus(path, entry="entry_0000")


def test_extreme_hard_xray_trips_upper_bound(tmp_path):
    """Wavelength of 1e-12 m (1 pm) corresponds to ~1.24 MeV,
    well past the 200 keV upper bound. Guard catches this too."""
    path = _make_nexus_with_wavelength(
        tmp_path / "gamma.h5", wavelength_m=1.0e-12
    )
    with pytest.raises(_EnergyOutOfRangeError, match="F-02"):
        _exp_params_from_nexus(path, entry="entry_0000")


def test_silent_fallback_still_silent_when_metadata_missing(tmp_path):
    """The guard is only triggered when the wavelength datum is
    actually present and produces a malformed energy. The existing
    "metadata not present → return defaults" fallback path stays
    silent (that's the F-05 audit topic, addressed separately)."""
    path = tmp_path / "no_instr.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w", track_order=True) as f:
        entry = f.create_group("entry_0000", track_order=True)
        d = entry.create_group("data", track_order=True)
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset(
            "img_gid_q", data=rng.random((2, 8, 8), dtype=np.float32)
        )
        d.create_dataset("q_xy", data=np.linspace(-1, 1, 8, dtype=np.float32))
        d.create_dataset("q_z", data=np.linspace(0, 1, 8, dtype=np.float32))
    # No instrument subtree at all — the broad-except in
    # _exp_params_from_nexus should catch the KeyError and return
    # ExpParameters(**defaults), with en = 18000 eV which is in
    # range. No exception escapes.
    params = _exp_params_from_nexus(path, entry="entry_0000")
    # 18000 eV ≈ 0.689 Å wavelength; we just need to confirm the
    # call returned an ExpParameters at all (no raise).
    assert params.wavelength == pytest.approx(0.6888, rel=1e-2)
