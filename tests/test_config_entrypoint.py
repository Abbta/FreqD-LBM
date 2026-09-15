"""Single-run contract for the JSON entrypoint."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Main_FreqDLBM_config_file as entry  # noqa: E402


def test_defaults_and_overrides_are_one_case(tmp_path):
    config_path = tmp_path / "case.json"
    config_path.write_text(json.dumps({"n": 5, "RSph_nm": 8.0}), encoding="utf-8")
    config = entry.load_config(config_path)
    sps = entry.initialize_sps(config)
    assert sps["n"] == 5
    assert sps["RSph_nm"] == 8.0
    assert sps["novt"] == sps["navg"] == 1
    assert sps["nPar1"] == sps["nPar2"] == sps["nPar3"] == 1
    np.testing.assert_array_equal(sps["ns"], [5])
    np.testing.assert_array_equal(sps["etaabscenSphmPass"], [1e4])
    assert "Par1values" not in sps

    config_path.write_text(json.dumps({"ns": [3, 5]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        entry.load_config(config_path)


def test_run_calls_single_simulation_once(monkeypatch):
    calls = []

    def geometry(sps):
        calls.append("geometry")

    def boundaries(sps):
        calls.append("boundaries")

    def filename(sps):
        calls.append("filename")

    def simulation(sps):
        calls.append("simulation")
        assert sps["n"] == 7
        assert sps["iovt"] == sps["iavg"] == 0
        assert sps["om"] == pytest.approx((1.0 / 3.0) / sps["delta"] ** 2)

    monkeypatch.setattr(entry, "_prepare_geometry", geometry)
    monkeypatch.setattr(entry, "_prepare_boundaries", boundaries)
    monkeypatch.setattr(entry.IO, "Set_fname", filename)
    monkeypatch.setattr(entry.Single_Sim, "SingleSimulation", simulation)
    entry.run(_defaults())
    assert calls == ["geometry", "boundaries", "filename", "simulation"]


def test_cylinder_qcm_3d_reuses_periodic_particle_boundary_path(tmp_path, monkeypatch):
    config_path = tmp_path / "cylinder-qcm.json"
    config_path.write_text(
        json.dumps(
            {
                "ProblemType": "CylinderQCM3D",
                "Dx_nm": 10.0,
                "RCyl_nm": 5.0,
                "HCyl_nm": 10.0,
                "CylBoxWidth_nm": 30.0,
                "CylBoxHeight_nm": 40.0,
            }
        ),
        encoding="utf-8",
    )
    sps = entry.initialize_sps(entry.load_config(config_path))
    entry._prepare_geometry(sps)
    monkeypatch.setattr(
        entry.OscBnd.Plots_from_Main,
        "Plot_LinkProps_3D",
        lambda *args, **kwargs: None,
    )
    entry._prepare_boundaries(sps)

    assert (sps["nx"], sps["ny"], sps["nz"]) == (3, 4, 3)
    assert sps["OscBndLocked"] is True
    assert sps["OscBndLockedTo"] == "Substrate"
    assert sps["CoverageTrue"] == pytest.approx(np.pi / 36)
    domains = sps["OscBndPars"]["OutsideLBMDomains"]
    np.testing.assert_array_equal(domains[:, 0, :], domains[:, 1, :])
    assert domains[1, 0, 1]
    assert domains[1, 1, 1]
    assert not domains[1, 2, 1]
    assert not domains[:, :, 0].any()
    boundary_types = set(np.unique(sps["OscBndPars"]["i_BCs"]))
    assert {1, 2, 3}.issubset(boundary_types)
    assert boundary_types.intersection({4, 5})
    assert np.all(sps["OscBndPars"]["uxLs"] == 1)
    assert np.all((sps["OscBndPars"]["qs"] > 0) & (sps["OscBndPars"]["qs"] <= 1))
    stencil = entry.OscBnd.General.ReadStencil(3)
    fields = entry.OscBnd.Calc_dr_ux_uy_uz_OscBnd(
        np.zeros((3, 4, 3, stencil[0]), dtype=complex),
        stencil[1], stencil[2], stencil[3], domains,
        sps["OscBndPars"]["InParticles"],
        sps["OscBndPars"]["OscBndAmps"], sps,
    )
    assert np.all(fields[1][domains.astype(bool)] == 1)
    stencil = entry.OscBnd.General.ReadStencil(3)
    fields = entry.OscBnd.Calc_dr_ux_uy_uz_OscBnd(
        np.zeros((3, 4, 3, stencil[0]), dtype=complex),
        stencil[1], stencil[2], stencil[3], domains,
        sps["OscBndPars"]["InParticles"],
        sps["OscBndPars"]["OscBndAmps"], sps,
    )
    assert np.all(fields[1][domains.astype(bool)] == 1)


def test_locked_qcm_cylinder_transmits_hydrodynamic_force_to_qcm():
    sps = {
        "ProblemType": "CylinderQCM3D",
        "Dx_nm": 10.0,
        "f0_SI": 5e6,
        "n": 1,
        "om": 0.01,
    }
    force_on_fluid = 2.0 + 3.0j
    result = entry.OscBnd.Update_Motion_3D(
        np.array([1]),
        1,
        np.array([0]),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.ones(1, dtype=complex),
        np.zeros(1, dtype=complex),
        np.zeros(1, dtype=complex),
        np.array([force_on_fluid]),
        np.zeros(1, dtype=complex),
        np.zeros(1, dtype=complex),
        1,
        0.01,
        np.zeros((6, 1), dtype=complex),
        np.zeros((3, 1)),
        np.zeros(16, dtype=complex),
        0.02,
        True,
        "Substrate",
        np.array([[0]]),
        1,
        sps,
    )
    assert result[3] == pytest.approx(force_on_fluid)
    assert sps["CylinderForceXOnCylinderByLiquid_LBM_Last"] == pytest.approx(
        -force_on_fluid
    )


def _defaults():
    config = dict(entry.DEFAULTS)
    config.update(entry.PROBLEM_DEFAULTS["StiffParticles"])
    config["delta0_nm"] = 252.0
    return config
