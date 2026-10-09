"""Cylinder-free QCM reference solved from the discrete FreqD-LBM equations.

The laterally uniform, x-polarized reference is an invariant three-moment
subspace of the existing D3Q19 streaming/collision kernel. This module solves
its stationary affine update directly, without changing the loaded solver.
"""

import numpy as np
from scipy.sparse import coo_matrix, eye, kron
from scipy.sparse.linalg import spsolve

from Libs import Lib_General as General
from Libs import Lib_Relax as Relax


def Relaxation_Parameters(SPs):
    """Return exactly the coefficients used by the existing reference ring-in."""
    om = SPs['om']
    tau_inv = 1. / SPs['tauInvBulk']
    lam = General.Validate_Lambda_TRT(SPs)
    tau_inv_asym = (4. - 2. * tau_inv) / (2. - tau_inv + 4 * lam * tau_inv)
    one_m_tau_inv_m_iom = 1. - tau_inv - 1j * om + 5. / 2. * om**2
    # Preserve the existing reference kernel's asymmetric coefficient.
    one_m_tau_inv_m_iom_asym = one_m_tau_inv_m_iom
    return tau_inv, tau_inv_asym, one_m_tau_inv_m_iom, one_m_tau_inv_m_iom_asym


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


def Solve_Direct(SPs):
    """Set the reference load from the stationary D3Q19 update, in Hz/n.

    The walls, frequency correction and collision coefficients match
    FD_LBM_Step_Ref. Raises if the linear solve does not satisfy its equations.
    """
    if SPs['dimensions'] != 3:
        raise ValueError('Direct reference currently supports the D3Q19 reference only')
    ny = int(SPs['ny'])
    if ny < 2:
        raise ValueError('Direct reference needs at least two y nodes')
    nd, cxs, cys, czs, ibars, wi, *_ = General.ReadStencil(3)
    pack, unpack = _shear_basis(cxs, cys, wi)
    coefficients = Relaxation_Parameters(SPs)
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
    z_bulk = SPs['ZBulk']
    if abs(1 + 3*z_bulk) < 1e-14:
        raise ValueError('Singular reference top impedance')
    reflection = (1 - 3*z_bulk) / (1 + 3*z_bulk)
    rows, columns, values = [], [], []
    stream_drive = np.zeros(3 * ny, dtype=np.complex128)
    for y in range(ny):
        terms = (
            (3 * (y+1), 1) if y < ny-1 else (3*y+2, reflection),
            (3*y+1, 1),
            (3 * (y-1)+2, 1) if y > 0 else (3*y, -1),
        )
        for group, (column, value) in enumerate(terms):
            rows.append(3*y + group)
            columns.append(column)
            values.append(value)
        if y == 0:
            stream_drive[2] = 1 / 3
    streaming = coo_matrix(
        (values, (rows, columns)), shape=(3*ny, 3*ny), dtype=np.complex128
    ).tocsr()
    block_collision = kron(eye(ny, format='csr'), collision, format='csr')
    update = block_collision @ streaming
    drive = block_collision @ stream_drive
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
