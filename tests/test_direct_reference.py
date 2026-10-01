import json
from unittest.mock import patch

import numpy as np
import pytest

from Libs import Lib_General as General
from Libs import Lib_Reference as Reference
from Libs import Lib_StreamCollide as Stream
from Main_FreqDLBM_config_file import load_config
from scipy.sparse.linalg import spsolve


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
