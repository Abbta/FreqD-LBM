"""Numerical parity checks for the optional JAX D3Q19 backend."""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Main_FreqDLBM_config_file as entry  # noqa: E402
from Libs import Lib_General as general  # noqa: E402
from Libs import Lib_Handle_Top as top  # noqa: E402
from Libs import Lib_JAX as jax_backend  # noqa: E402
from Libs import Lib_StreamCollide as stream_collide  # noqa: E402


pytest.importorskip("jax")


def test_jax_config_is_optional_and_cylinder_3d_only():
    assert entry.DEFAULTS["ComputeBackend"] == "numba"
    config = dict(entry.DEFAULTS)
    config.update(entry.PROBLEM_DEFAULTS["CylinderQCM3D"])
    config["ProblemType"] = "CylinderQCM3D"
    config["delta0_nm"] = 252.0
    config["ComputeBackend"] = "jax"
    entry._validate_config(config)

    config.update(entry.PROBLEM_DEFAULTS["Cylinder2D"])
    config["ProblemType"] = "Cylinder2D"
    with pytest.raises(ValueError, match="3D oscillating boundaries"):
        entry._validate_config(config)


@pytest.mark.parametrize("nx", [3, 4])
def test_jax_step_matches_numba_d3q19(nx):
    ny = 5
    nz = nx
    nd, cxs, cys, czs, ibars, wi, *_ = general.ReadStencil(3)
    rng = np.random.default_rng(42 + nx)
    h = rng.normal(size=(nx, ny, nz, nd)) + 1j * rng.normal(
        size=(nx, ny, nz, nd)
    )
    outside = np.zeros((nx, ny, nz), dtype=np.int64)
    i_bcs = np.ones((nx, ny, nz, nd), dtype=np.int64)
    for direction in range(nd):
        if cys[direction] == 1:
            i_bcs[:, 0, :, direction] = 2
        elif cys[direction] == -1:
            i_bcs[:, -1, :, direction] = 3

    links = [(1, 2, 1, 1, 4), (nx - 1, 2, nz - 1, 2, 5)]
    poi_ls = np.zeros_like(i_bcs)
    for link_id, (x, y, z, direction, boundary) in enumerate(links):
        i_bcs[x, y, z, direction] = boundary
        poi_ls[x, y, z, direction] = link_id
    qs = np.array([0.37, 0.8])
    ux_ls = np.array([1.0 + 0.2j, 0.8 - 0.1j])
    uy_ls = np.array([0.1 - 0.3j, -0.2 + 0.4j])
    uz_ls = np.array([-0.3 + 0.1j, 0.5 + 0.2j])
    tau_invs = 0.7 + 0.01j + rng.random((nx, ny, nz)) * 0.1
    tau_invs_asym = 0.9 - 0.02j + rng.random((nx, ny, nz)) * 0.1
    omega = 0.03
    one_m_tau = 1.0 - tau_invs - 1j * omega
    one_m_tau_asym = 1.0 - tau_invs_asym - 1j * omega
    matrices_top = top.Calc_MatricesTop_3D(nx, nz, 1.0 / 6.0, omega)

    expected = stream_collide.FreqDLBMStep_OscBnd_3D(
        h.copy(), nx, ny, nz, nd, cxs, cys, czs, wi, ibars, matrices_top,
        len(qs), outside, i_bcs, poi_ls, qs, ux_ls, uy_ls, uz_ls,
        tau_invs, tau_invs_asym, one_m_tau, one_m_tau_asym,
    )
    stepper = jax_backend.OscBndStepper3D(
        cxs, cys, czs, wi, ibars, matrices_top, outside, i_bcs, poi_ls, qs,
        tau_invs, tau_invs_asym, one_m_tau, one_m_tau_asym,
        device_name="cpu",
    )
    actual = stepper.step(h.copy(), ux_ls, uy_ls, uz_ls)

    for jax_value, numba_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(jax_value), np.asarray(numba_value), rtol=2e-13, atol=2e-13
        )


def test_bounded_cylinder_result_matches_numba(tmp_path, monkeypatch):
    monkeypatch.setattr(
        entry.OscBnd.Plots_from_Main, "Plot_LinkProps_3D", lambda *args: None
    )
    base = json.loads(
        (ROOT.parent / "configs" / "fdlbm-cylinder-qcm-3d-fast.json").read_text(
            encoding="utf-8"
        )
    )
    results = {}
    for backend in ("numba", "jax"):
        config_path = tmp_path / f"{backend}.json"
        config_path.write_text(
            json.dumps(
                {
                    **base,
                    "folder": str(tmp_path / backend),
                    "ComputeBackend": backend,
                    "JAXDevice": "cpu",
                }
            ),
            encoding="utf-8",
        )
        results[backend] = entry.run(entry.load_config(config_path))

    for key in ("ResponseRawLast", "CylinderForceXOnCylinderByLiquid_LBM_Last"):
        assert results["jax"][key] == pytest.approx(
            results["numba"][key], rel=2e-12, abs=2e-12
        )
    # This deliberately under-converged smoke case makes the nonlinear
    # extrapolation much more sensitive than its raw response.
    assert results["jax"]["Dfcbyn_Extrapol"] == pytest.approx(
        results["numba"]["Dfcbyn_Extrapol"], rel=5e-5
    )
    assert results["jax"]["steps"] == results["numba"]["steps"]
    assert results["jax"]["ProblemFlag"] == results["numba"]["ProblemFlag"]
    assert results["jax"]["JAXPlatformResolved"] == "cpu"
