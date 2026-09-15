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


def _defaults():
    config = dict(entry.DEFAULTS)
    config.update(entry.PROBLEM_DEFAULTS["StiffParticles"])
    config["delta0_nm"] = 252.0
    return config
