"""Cylinder-free QCM reference solved from the discrete FreqD-LBM equations.

The laterally uniform, x-polarized reference is an invariant three-moment
subspace of the existing D3Q19 streaming/collision kernel. This module solves
its stationary affine update directly, without changing the loaded solver.
"""

import numpy as np
from scipy.sparse import coo_matrix, eye, kron
from scipy.sparse.linalg import eigs, spsolve

from Libs import Lib_General as General
from Libs import Lib_Relax as Relax


def Relaxation_Parameters(SPs, include_second_order=True):
    """Return TRT coefficients for Johannsmann Eqs. 75 and Sec. 6.2.

    The first two values multiply the equilibrium populations and therefore
    include the empirical ``+5/2 omega**2 h_eq`` correction.  The last two
    multiply the current populations.  ``tauInvBulk`` already stores
    ``tau**-1``.
    """
    om = SPs['om']
    tau_inv = SPs['tauInvBulk']
    lam = General.Validate_Lambda_TRT(SPs)
    tau_inv_asym = (4. - 2. * tau_inv) / (2. - tau_inv + 4 * lam * tau_inv)
    second_order = 5. / 2. * om**2 if include_second_order else 0.
    equilibrium_coeff = tau_inv + second_order
    equilibrium_coeff_asym = tau_inv_asym + second_order
    population_coeff = 1. - tau_inv - 1j * om
    population_coeff_asym = 1. - tau_inv_asym - 1j * om
    return (equilibrium_coeff, equilibrium_coeff_asym,
            population_coeff, population_coeff_asym)


def _shear_basis(cxs, cys, wi):
    """Map the three y-velocity groups to/from D3Q19 populations."""
    directions = (-1, 0, 1)
    pack = np.zeros((len(wi), 3), dtype=np.complex128)
    unpack = np.zeros((3, len(wi)), dtype=np.complex128)
    for group, y_velocity in enumerate(directions):
        mask = cys == y_velocity
        pack[mask, group] = wi[mask] * cxs[mask] / np.sum(wi[mask] * cxs[mask]**2)
        unpack[group, mask] = cxs[mask]
    return pack, unpack


def Build_Reduced_Update(SPs, boundary_mode='qcm_impedance',
                         include_second_order=True):
    """Build a reduced liquid update for a selected boundary experiment."""
    if SPs['dimensions'] != 3:
        raise ValueError('Direct reference currently supports the D3Q19 reference only')
    ny = int(SPs['ny'])
    if ny < 2:
        raise ValueError('Direct reference needs at least two y nodes')
    nd, cxs, cys, czs, ibars, wi, *_ = General.ReadStencil(3)
    pack, unpack = _shear_basis(cxs, cys, wi)
    coefficients = Relaxation_Parameters(SPs, include_second_order)
    collision_full = np.column_stack([
        Relax.Relax_3D(pack[:, j].copy(), cxs, cys, czs, wi, ibars, *coefficients)
        for j in range(3)
    ])
    collision = unpack @ collision_full
    if not np.allclose(collision_full, pack @ collision, atol=1e-12, rtol=0):
        raise RuntimeError('Reference collision left the uniform shear subspace')

    # Stream the three x-momentum groups (c_y = -1, 0, +1).
    # At the QCM: m_+(0) = 1/3 - m_-(0).
    # At the Fourier top: m_-(ny-1) = (1-3Z)/(1+3Z) m_+(ny-1).
    if boundary_mode not in ('qcm_impedance', 'periodic'):
        raise ValueError(f'Unknown reference boundary mode {boundary_mode!r}')
    z_bulk = SPs['ZBulk']
    if boundary_mode == 'qcm_impedance' and abs(1 + 3*z_bulk) < 1e-14:
        raise ValueError('Singular reference top impedance')
    reflection = (1 - 3*z_bulk) / (1 + 3*z_bulk)
    rows, columns, values = [], [], []
    stream_drive = np.zeros(3 * ny, dtype=np.complex128)
    for y in range(ny):
        if boundary_mode == 'periodic':
            terms = (
                (3 * ((y+1) % ny), 1),
                (3*y+1, 1),
                (3 * ((y-1) % ny)+2, 1),
            )
        else:
            terms = (
                (3 * (y+1), 1) if y < ny-1 else (3*y+2, reflection),
                (3*y+1, 1),
                (3 * (y-1)+2, 1) if y > 0 else (3*y, -1),
            )
        for group, (column, value) in enumerate(terms):
            rows.append(3*y + group)
            columns.append(column)
            values.append(value)
        if boundary_mode == 'qcm_impedance' and y == 0:
            stream_drive[2] = 1 / 3
    streaming = coo_matrix(
        (values, (rows, columns)), shape=(3*ny, 3*ny), dtype=np.complex128
    ).tocsr()
    block_collision = kron(eye(ny, format='csr'), collision, format='csr')
    update = block_collision @ streaming
    drive = block_collision @ stream_drive
    return update, drive, pack, unpack


def Compare_Stability_Cases(SPs):
    """Return spectral stability with boundaries and correction isolated."""
    cases = (
        ('periodic_no_correction', 'periodic', False),
        ('periodic_with_correction', 'periodic', True),
        ('qcm_impedance_no_correction', 'qcm_impedance', False),
        ('qcm_impedance_with_correction', 'qcm_impedance', True),
    )
    results = []
    for name, boundary_mode, include_second_order in cases:
        update, _, _, _ = Build_Reduced_Update(
            SPs, boundary_mode=boundary_mode,
            include_second_order=include_second_order,
        )
        eigenvalues = eigs(
            update, k=min(6, update.shape[0] - 2), which='LM',
            return_eigenvectors=False, tol=1e-11,
            maxiter=max(2000, 20 * update.shape[0]),
        )
        dominant = eigenvalues[np.argmax(np.abs(eigenvalues))]
        results.append({
            'name': name,
            'boundary_mode': boundary_mode,
            'include_second_order': include_second_order,
            'dominant_eigenvalue': complex(dominant),
            'spectral_radius': float(abs(dominant)),
        })
    return results


def Solve_Direct(SPs):
    """Set the reference load from the stationary D3Q19 update, in Hz/n.

    The walls, frequency correction and collision coefficients match
    FD_LBM_Step_Ref. Raises if the linear solve does not satisfy its equations.
    """
    update, drive, pack, _ = Build_Reduced_Update(SPs)
    ny = int(SPs['ny'])
    solution = spsolve(eye(3*ny, format='csr') - update, drive)
    residual = np.max(np.abs(update @ solution + drive - solution))
    if not np.all(np.isfinite(solution)) or not np.isfinite(residual) or residual > 1e-9:
        raise RuntimeError(f'Direct reference solve failed: residual={residual}')

    # The reference kernel computes the QCM wall force before collision.
    wall_force = 1 / 3 - 2 * solution[0]
    response = General.Calc_Dfcbyn(SPs, wall_force, 0, True)
    SPs['Dfcbyn_Ref'] = response
    SPs['Dfratio_Ref'] = response.imag / (-response.real)
    SPs['StdErr_Ref'] = np.nan  # A direct solve has no ring-in fit error.
    SPs['ReferenceResidual_LBM'] = float(residual)
    SPs['ProblemFlag'] = 0
    print(f'Direct liquid reference: Dfcbyn_Ref={response} Hz/n residual={residual:.3g}', flush=True)
    return response


def Analyze_Iteration_Stability(SPs, warmup_steps, measurement_steps,
                                perturbation_relative, random_seed):
    """Measure growth of a small perturbation under the reference iteration.

    The reference update is affine. Two states subject to the same wall drive
    therefore differ by the homogeneous update operator. Renormalizing that
    difference after every step implements a matrix-free power iteration.
    """
    from Libs import Lib_StreamCollide as Stream

    update, _, pack, _ = Build_Reduced_Update(SPs)
    eigenvalues = eigs(
        update, k=min(6, update.shape[0] - 2), which='LM',
        return_eigenvectors=False, tol=1e-11,
        maxiter=max(2000, 20 * update.shape[0]),
    )
    dominant_eigenvalue = eigenvalues[np.argmax(np.abs(eigenvalues))]

    ny = int(SPs['ny'])
    nd, cxs, cys, czs, ibars, wi, i_ups, i_notups, i_downs, i_notdowns = \
        General.ReadStencil(3)
    coefficients = Relaxation_Parameters(SPs)

    def step(state):
        return Stream.FD_LBM_Step_Ref(
            state, ny, nd, cxs, cys, czs, wi, ibars,
            i_ups, i_notups, i_downs, i_notdowns,
            SPs['ZBulk'], SPs['om'], *coefficients,
        )

    baseline = np.zeros((ny, nd), dtype=np.complex128)
    for _ in range(warmup_steps):
        baseline, _ = step(baseline)

    rng = np.random.default_rng(random_seed)
    moment_perturbation = (
        rng.standard_normal((ny, 3)) + 1j * rng.standard_normal((ny, 3))
    )
    perturbation = moment_perturbation @ pack.T
    baseline_norm = np.linalg.norm(baseline)
    target_norm = perturbation_relative * max(baseline_norm, 1.0)
    perturbation *= target_norm / np.linalg.norm(perturbation)
    trial = baseline + perturbation

    amplification = np.empty(measurement_steps)
    rayleigh = np.empty(measurement_steps, dtype=np.complex128)
    wall_force_difference = np.empty(measurement_steps, dtype=np.complex128)
    for index in range(measurement_steps):
        previous = trial - baseline
        baseline, baseline_force = step(baseline)
        trial, trial_force = step(trial)
        difference = trial - baseline
        previous_norm = np.linalg.norm(previous)
        difference_norm = np.linalg.norm(difference)
        amplification[index] = difference_norm / previous_norm
        rayleigh[index] = np.vdot(previous, difference) / np.vdot(previous, previous)
        wall_force_difference[index] = trial_force - baseline_force
        if not np.isfinite(difference_norm) or difference_norm == 0:
            raise RuntimeError('Reference perturbation became non-finite or vanished')
        difference *= target_norm / difference_norm
        trial = baseline + difference

    tail_start = measurement_steps // 2
    tail_amplification = amplification[tail_start:]
    geometric_mean = float(np.exp(np.mean(np.log(tail_amplification))))
    result = {
        'dominant_eigenvalue': complex(dominant_eigenvalue),
        'spectral_radius': float(abs(dominant_eigenvalue)),
        'tail_geometric_mean_amplification': geometric_mean,
        'tail_median_amplification': float(np.median(tail_amplification)),
        'warmup_steps': int(warmup_steps),
        'measurement_steps': int(measurement_steps),
        'perturbation_relative': float(perturbation_relative),
        'random_seed': int(random_seed),
        'amplification': amplification,
        'rayleigh': rayleigh,
        'wall_force_difference': wall_force_difference,
        'comparison_cases': Compare_Stability_Cases(SPs),
    }
    SPs['ReferenceStabilityResult'] = result
    return result
