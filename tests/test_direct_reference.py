import json
from unittest.mock import patch

import numpy as np
import pytest

from Libs import Lib_General as General
from Libs import Lib_Reference as Reference
from Libs import Lib_StreamCollide as Stream
from Main_FreqDLBM_config_file import load_config
from scipy.sparse.linalg import spsolve


def test_reference_trt_coefficients_match_published_equations():
    omega = 0.1
    tau_inv = 0.8
    sps = dict(om=omega, tauInvBulk=tau_inv, Lambda_TRT=0.25)
    eq_sym, eq_asym, pop_sym, pop_asym = Reference.Relaxation_Parameters(sps)
    tau_inv_asym = (
        (4.0 - 2.0 * tau_inv)
        / (2.0 - tau_inv + 4.0 * 0.25 * tau_inv)
    )
    second_order = 2.5 * omega**2

    assert eq_sym == pytest.approx(tau_inv + second_order)
    assert eq_asym == pytest.approx(tau_inv_asym + second_order)
    assert pop_sym == pytest.approx(1.0 - tau_inv - 1j * omega)
    assert pop_asym == pytest.approx(1.0 - tau_inv_asym - 1j * omega)

    eq_sym, eq_asym, pop_sym, pop_asym = Reference.Relaxation_Parameters(
        sps, include_second_order=False
    )
    assert eq_sym == pytest.approx(tau_inv)
    assert eq_asym == pytest.approx(tau_inv_asym)
    assert pop_sym == pytest.approx(1.0 - tau_inv - 1j * omega)
    assert pop_asym == pytest.approx(1.0 - tau_inv_asym - 1j * omega)


def test_reference_stability_comparison_isolates_four_cases():
    sps = dict(
        dimensions=3, ny=6, om=0.05, tauInvBulk=1.0,
        Lambda_TRT=0.25, ZBulk=0.1 + 0.1j,
    )
    cases = Reference.Compare_Stability_Cases(sps)
    assert [case['name'] for case in cases] == [
        'periodic_no_correction',
        'periodic_with_correction',
        'qcm_impedance_no_correction',
        'qcm_impedance_with_correction',
    ]
    assert all(np.isfinite(case['spectral_radius']) for case in cases)


def test_reference_kernel_applies_published_wall_conditions():
    ny = 4
    z_bulk = 0.2 + 0.1j
    nd, cx, cy, cz, ib, wi, up, notup, down, notdown = General.ReadStencil(3)
    initial = (
        np.arange(ny * nd, dtype=float).reshape(ny, nd)
        + 1j * np.arange(ny * nd, dtype=float).reshape(ny, nd) / 10.0
    )
    expected_top_velocity = 2.0 * (initial[-1, 7] - initial[-1, 14]) / (z_bulk + 1.0 / 3.0)

    # Identity collision isolates the streaming and bounce-back equations.
    updated, wall_force = Stream.FD_LBM_Step_Ref(
        initial.copy(), ny, nd, cx, cy, cz, wi, ib, up, notup,
        down, notdown, z_bulk, 0.0, 0.0, 0.0, 1.0, 1.0
    )

    for i in up:
        opposite = ib[i]
        expected = initial[0, opposite] - 6.0 * wi[opposite] * cx[opposite]
        assert updated[0, i] == pytest.approx(expected)
    for i in down:
        opposite = ib[i]
        expected = (
            initial[-1, opposite]
            - 6.0 * wi[opposite] * cx[opposite] * expected_top_velocity
        )
        assert updated[-1, i] == pytest.approx(expected)
    expected_force = sum(
        (updated[0, i] + initial[0, ib[i]]) * cx[i] for i in up
    )
    assert wall_force == pytest.approx(expected_force)


@pytest.mark.parametrize("ny,eta,tandel", [(24, 1.0, 1e33), (352, 1.0, 1e33), (32, 2.0, 10.0)])
def test_direct_reference_satisfies_existing_kernel(ny, eta, tandel):
    # Construct only the parameters needed by the direct reference and kernel.
    sps = dict(dimensions=3, ny=ny, Dx_nm=12.5, n=9, f0_SI=5e6,
               Zq_SI=8.8e6, etaabsBulk=eta, tandelBulk=tandel,
               Lambda_TRT=0.25, om=0.0073631077818510785)
    General.Calc_tauInvBulk_ZBulk(sps)
    solved = []
    def record_solve(*args):
        solution = spsolve(*args)
        solved.append(solution)
        return solution
    with patch.object(Reference, "spsolve", side_effect=record_solve):
        response = Reference.Solve_Direct(sps)
    moments_solved = solved[0].reshape(ny, 3)
    assert np.isfinite(response)
    assert sps["ReferenceResidual_LBM"] < 1e-12
    if eta == 1.0:
        assert response == pytest.approx(complex(-238.979381288382, 238.980538030148), abs=0.01)

    # An independent full D3Q19 kernel step must leave the solved reference fixed.
    nd, cx, cy, cz, ib, wi, up, notup, down, notdown = General.ReadStencil(3)
    pack, unpack = Reference._shear_basis(cx, cy, wi)
    coeffs = Reference.Relaxation_Parameters(sps)
    # Check that the production reference collision preserves the stated
    # three-dimensional invariant subspace for non-BGK parameters too.
    rng = np.random.default_rng(11)
    moments = rng.normal(size=(ny, 3)) + 1j * rng.normal(size=(ny, 3))
    populations = moments @ pack.T
    updated, force = Stream.FD_LBM_Step_Ref(
        populations.copy(), ny, nd, cx, cy, cz, wi, ib, up, notup,
        down, notdown, sps["ZBulk"], sps["om"], *coeffs
    )
    assert np.max(np.abs(updated - (updated @ unpack.T) @ pack.T)) < 1e-11

    fixed_populations = moments_solved @ pack.T
    next_populations, wall_force = Stream.FD_LBM_Step_Ref(
        fixed_populations.copy(), ny, nd, cx, cy, cz, wi, ib, up, notup,
        down, notdown, sps["ZBulk"], sps["om"], *coeffs
    )
    assert np.max(np.abs(next_populations - fixed_populations)) < 1e-11
    assert General.Calc_Dfcbyn(sps, wall_force, 0, True) == pytest.approx(response, abs=1e-9)


def test_reference_method_config_validation(tmp_path):
    base = {"ProblemType": "CylinderQCM3D", "ReferenceMethod": "direct"}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(base))
    assert load_config(path)["ReferenceMethod"] == "direct"
    path.write_text(json.dumps({**base, "ReferenceMethod": "unknown"}))
    with pytest.raises(ValueError, match="ReferenceMethod"):
        load_config(path)
    path.write_text(json.dumps({"ProblemType": "Cylinder2D", "ReferenceMethod": "direct"}))
    with pytest.raises(ValueError, match="D3Q19"):
        load_config(path)
