"""Optional JAX backends for the existing 2D and 3D oscillating-boundary steps."""

from __future__ import annotations

import os

import numpy as np


def _jax_runtime(device_name, preallocate):
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE", "true" if preallocate else "false"
    )
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise RuntimeError(
            "ComputeBackend='jax' requires the optional JAX dependency"
        ) from error
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices() if device_name == "auto" else jax.devices(device_name)
    if not devices:
        raise RuntimeError(f"No JAX {device_name!r} device is available")
    return jax, jnp, devices[0]


class OscBndStepper3D:
    """JIT-compiled equivalent of ``FreqDLBMStep_OscBnd_3D``."""

    def __init__(
        self,
        cxs,
        cys,
        czs,
        wi,
        ibars,
        matrices_top,
        outside_lbm_domains,
        i_bcs,
        poi_ls,
        qs,
        tau_invs,
        tau_invs_asym,
        one_m_tau_invs_m_iom,
        one_m_tau_invs_m_iom_asym,
        device_name="auto",
        preallocate=False,
    ):
        jax, jnp, device = _jax_runtime(device_name, preallocate)
        self.jax = jax
        self.jnp = jnp
        self.device = device
        self.cxs = tuple(int(value) for value in cxs)
        self.cys = tuple(int(value) for value in cys)
        self.czs = tuple(int(value) for value in czs)
        self.ibars = tuple(int(value) for value in ibars)
        self.nd = len(self.cxs)
        self.nx, self.ny, self.nz = np.asarray(outside_lbm_domains).shape
        if self.nx != self.nz:
            raise ValueError("The existing Fourier-top model requires nx == nz")

        put = lambda value: jax.device_put(np.asarray(value), device)
        self.wi = put(np.asarray(wi, dtype=np.float64))
        self.outside = put(np.asarray(outside_lbm_domains, dtype=np.int64))
        self.i_bcs = put(np.asarray(i_bcs, dtype=np.int64))
        self.poi_ls = put(np.asarray(poi_ls, dtype=np.int64))
        self.qs = put(np.asarray(qs, dtype=np.float64))
        self.tau_invs = put(np.asarray(tau_invs, dtype=np.complex128))
        self.tau_invs_asym = put(np.asarray(tau_invs_asym, dtype=np.complex128))
        self.one_m_tau_invs_m_iom = put(
            np.asarray(one_m_tau_invs_m_iom, dtype=np.complex128)
        )
        self.one_m_tau_invs_m_iom_asym = put(
            np.asarray(one_m_tau_invs_m_iom_asym, dtype=np.complex128)
        )
        self.n_links = int(np.asarray(qs).size)
        if self.n_links == 0:
            raise ValueError("The JAX oscillating-boundary step requires link data")

        qmax = self.nx // 2
        modes = np.arange(-qmax, qmax + 1)
        qx = 2.0 * np.pi * modes / self.nx
        qz = 2.0 * np.pi * modes / self.nz
        self.forward_x = put(
            np.exp(-1j * np.outer(qx, np.arange(self.nx))) / self.nx
        )
        self.forward_z = put(
            np.exp(-1j * np.outer(qz, np.arange(self.nz))) / self.nz
        )
        self.inverse_x = put(np.exp(1j * np.outer(qx, np.arange(self.nx))))
        self.inverse_z = put(np.exp(1j * np.outer(qz, np.arange(self.nz))))
        matrix_indices = modes % np.asarray(matrices_top).shape[0]
        ordered_matrices = np.asarray(matrices_top)[
            matrix_indices[:, None], matrix_indices[None, :]
        ]
        self.matrices_top = put(ordered_matrices)

        flat_link_positions = []
        flat_link_ids = []
        i_bcs_np = np.asarray(i_bcs)
        poi_ls_np = np.asarray(poi_ls)
        for direction in range(self.nd):
            surface = (i_bcs_np[..., direction] == 4) | (
                i_bcs_np[..., direction] == 5
            )
            positions = np.flatnonzero(surface.reshape(-1))
            flat_link_positions.append(put(positions.astype(np.int64)))
            flat_link_ids.append(
                put(poi_ls_np[..., direction].reshape(-1)[positions].astype(np.int64))
            )
        self.flat_link_positions = tuple(flat_link_positions)
        self.flat_link_ids = tuple(flat_link_ids)
        self.has_links = tuple(value.size > 0 for value in flat_link_positions)
        self._compiled_step = jax.jit(self._build_step(), donate_argnums=(0,))

    @property
    def platform(self):
        return self.device.platform

    @property
    def device_description(self):
        return getattr(self.device, "device_kind", str(self.device))

    def _top_velocity(self, h):
        jnp = self.jnp
        top = h[:, -1, :, :]
        stresses = jnp.stack(
            (
                2.0 * (top[..., 7] - top[..., 14]),
                2.0
                * (
                    top[..., 7]
                    + top[..., 14]
                    + top[..., 11]
                    + top[..., 17]
                    + top[..., 3]
                ),
                2.0 * (top[..., 11] - top[..., 17]),
            ),
            axis=-1,
        )
        stress_modes = jnp.einsum(
            "ax,xzc,bz->abc", self.forward_x, stresses, self.forward_z
        )
        velocity_modes = jnp.einsum(
            "abij,abj->abi", self.matrices_top, stress_modes
        )
        return jnp.einsum(
            "ax,abi,bz->xzi", self.inverse_x, velocity_modes, self.inverse_z
        )

    def _build_step(self):
        jnp = self.jnp
        wi = self.wi
        outside = self.outside
        i_bcs = self.i_bcs
        poi_ls = self.poi_ls
        qs = self.qs
        ibars_array = jnp.asarray(self.ibars, dtype=jnp.int64)

        def step(h, ux_ls, uy_ls, uz_ls):
            top_velocity = self._top_velocity(h)
            streamed = []
            wall_forces = []
            fx_ls = jnp.zeros(self.n_links, dtype=jnp.complex128)
            fy_ls = jnp.zeros(self.n_links, dtype=jnp.complex128)
            fz_ls = jnp.zeros(self.n_links, dtype=jnp.complex128)
            for direction in range(self.nd):
                cx, cy, cz = (
                    self.cxs[direction],
                    self.cys[direction],
                    self.czs[direction],
                )
                ibar = self.ibars[direction]
                boundary = i_bcs[..., direction]
                h_bar = h[..., ibar]
                bulk = jnp.roll(h[..., direction], (cx, cy, cz), axis=(0, 1, 2))
                bottom = h_bar - 6.0 * wi[ibar] * self.cxs[ibar]
                top = (
                    h_bar
                    - 6.0 * wi[ibar] * self.cxs[ibar] * top_velocity[:, None, :, 0]
                    - 6.0 * wi[ibar] * self.cys[ibar] * top_velocity[:, None, :, 1]
                    - 6.0 * wi[ibar] * self.czs[ibar] * top_velocity[:, None, :, 2]
                )
                link_ids = poi_ls[..., direction]
                q = qs[link_ids]
                link_velocity = (
                    self.cxs[ibar] * ux_ls[link_ids]
                    + self.cys[ibar] * uy_ls[link_ids]
                    + self.czs[ibar] * uz_ls[link_ids]
                )
                h_plus = jnp.roll(h_bar, (-cx, -cy, -cz), axis=(0, 1, 2))
                surface_far = jnp.where(
                    q <= 0.5,
                    2.0 * q * h_bar
                    + (1.0 - 2.0 * q) * h_plus
                    - 6.0 * wi[ibar] * link_velocity,
                    h_bar / (2.0 * q)
                    + (1.0 - 1.0 / (2.0 * q)) * h[..., direction]
                    - 3.0 / q * wi[ibar] * link_velocity,
                )
                surface_near = h_bar - 6.0 * wi[ibar] * link_velocity
                value = jnp.zeros_like(h_bar)
                value = jnp.where(boundary == 1, bulk, value)
                value = jnp.where(boundary == 2, bottom, value)
                value = jnp.where(boundary == 3, top, value)
                value = jnp.where(boundary == 4, surface_far, value)
                value = jnp.where(boundary == 5, surface_near, value)
                streamed.append(value)
                wall_forces.append(
                    jnp.where(
                        boundary == 2,
                        (value + h_bar) * self.cxs[direction],
                        0.0j,
                    )
                )
                if self.has_links[direction]:
                    positions = self.flat_link_positions[direction]
                    ids = self.flat_link_ids[direction]
                    momentum = (value + h_bar).reshape(-1)[positions]
                    fx_ls = fx_ls.at[ids].set(momentum * cx)
                    fy_ls = fy_ls.at[ids].set(momentum * cy)
                    fz_ls = fz_ls.at[ids].set(momentum * cz)

            h_streamed = jnp.stack(streamed, axis=-1)
            density = jnp.sum(h_streamed, axis=-1)
            ux = jnp.einsum("xyzi,i->xyz", h_streamed, jnp.asarray(self.cxs))
            uy = jnp.einsum("xyzi,i->xyz", h_streamed, jnp.asarray(self.cys))
            uz = jnp.einsum("xyzi,i->xyz", h_streamed, jnp.asarray(self.czs))
            equilibrium = wi * (
                density[..., None]
                + 3.0
                * (
                    ux[..., None] * jnp.asarray(self.cxs)
                    + uy[..., None] * jnp.asarray(self.cys)
                    + uz[..., None] * jnp.asarray(self.czs)
                )
            )
            partner_h = h_streamed[..., ibars_array]
            partner_equilibrium = equilibrium[..., ibars_array]
            symmetric_h = (h_streamed + partner_h) / 2.0
            asymmetric_h = (h_streamed - partner_h) / 2.0
            symmetric_equilibrium = (equilibrium + partner_equilibrium) / 2.0
            asymmetric_equilibrium = (equilibrium - partner_equilibrium) / 2.0
            relaxed = (
                self.one_m_tau_invs_m_iom[..., None] * symmetric_h
                + self.tau_invs[..., None] * symmetric_equilibrium
                + self.one_m_tau_invs_m_iom_asym[..., None] * asymmetric_h
                + self.tau_invs_asym[..., None] * asymmetric_equilibrium
            )
            h_next = jnp.where(outside[..., None] == 0, relaxed, h)
            wall_force = jnp.sum(jnp.stack(wall_forces, axis=-1), axis=(1, 3))
            return h_next, wall_force, fx_ls, fy_ls, fz_ls

        return step

    def step(self, h, ux_ls, uy_ls, uz_ls):
        put = lambda value: self.jax.device_put(value, self.device)
        result = self._compiled_step(
            put(h), put(ux_ls), put(uy_ls), put(uz_ls)
        )
        h_next, *forces = result
        return h_next, *(np.asarray(force) for force in forces)


class OscBndStepperCylinder2D:
    """JIT-compiled equivalent of the periodic D2Q9 cylinder step."""

    def __init__(
        self,
        cxs,
        cys,
        wi,
        ibars,
        outside_lbm_domains,
        i_bcs,
        poi_ls,
        qs,
        tau_invs,
        tau_invs_asym,
        one_m_tau_invs_m_iom,
        one_m_tau_invs_m_iom_asym,
        device_name="auto",
        preallocate=False,
    ):
        jax, jnp, device = _jax_runtime(device_name, preallocate)
        self.jax = jax
        self.jnp = jnp
        self.device = device
        self.cxs = tuple(int(value) for value in cxs)
        self.cys = tuple(int(value) for value in cys)
        self.ibars = tuple(int(value) for value in ibars)
        self.nd = len(self.cxs)
        self.nx, self.ny = np.asarray(outside_lbm_domains).shape

        put = lambda value: jax.device_put(np.asarray(value), device)
        self.wi = put(np.asarray(wi, dtype=np.float64))
        # Compact exact integer geometry; numeric state remains complex128.
        self.outside = put(np.asarray(outside_lbm_domains, dtype=np.uint8))
        self.i_bcs = put(np.asarray(i_bcs, dtype=np.int8))
        self.poi_ls = put(np.asarray(poi_ls, dtype=np.int32))
        self.qs = put(np.asarray(qs, dtype=np.float64))
        self.tau_invs = put(np.asarray(tau_invs, dtype=np.complex128))
        self.tau_invs_asym = put(np.asarray(tau_invs_asym, dtype=np.complex128))
        self.one_m_tau_invs_m_iom = put(
            np.asarray(one_m_tau_invs_m_iom, dtype=np.complex128)
        )
        self.one_m_tau_invs_m_iom_asym = put(
            np.asarray(one_m_tau_invs_m_iom_asym, dtype=np.complex128)
        )
        self.n_links = int(np.asarray(qs).size)
        if self.n_links == 0:
            raise ValueError("The JAX 2D cylinder step requires boundary-link data")

        flat_link_positions = []
        flat_link_ids = []
        i_bcs_np = np.asarray(i_bcs)
        poi_ls_np = np.asarray(poi_ls)
        for direction in range(self.nd):
            surface = (i_bcs_np[..., direction] == 4) | (
                i_bcs_np[..., direction] == 5
            )
            positions = np.flatnonzero(surface.reshape(-1))
            flat_link_positions.append(put(positions.astype(np.int32)))
            flat_link_ids.append(
                put(poi_ls_np[..., direction].reshape(-1)[positions].astype(np.int32))
            )
        self.flat_link_positions = tuple(flat_link_positions)
        self.flat_link_ids = tuple(flat_link_ids)
        self.has_links = tuple(value.size > 0 for value in flat_link_positions)
        self._compiled_step = jax.jit(self._build_step(), donate_argnums=(0,))
        self._compiled_batches = {}

    @property
    def platform(self):
        return self.device.platform

    @property
    def device_description(self):
        return getattr(self.device, "device_kind", str(self.device))

    def _build_step(self):
        jnp = self.jnp
        wi = self.wi
        outside = self.outside
        i_bcs = self.i_bcs
        poi_ls = self.poi_ls
        qs = self.qs
        ibars_array = jnp.asarray(self.ibars, dtype=jnp.int32)
        cxs_array = jnp.asarray(self.cxs)
        cys_array = jnp.asarray(self.cys)

        def step(h, ux_ls, uy_ls):
            streamed = []
            fx_ls = jnp.zeros(self.n_links, dtype=jnp.complex128)
            fy_ls = jnp.zeros(self.n_links, dtype=jnp.complex128)
            for direction in range(self.nd):
                cx = self.cxs[direction]
                cy = self.cys[direction]
                ibar = self.ibars[direction]
                boundary = i_bcs[..., direction]
                h_bar = h[..., ibar]
                bulk = jnp.roll(h[..., direction], (cx, cy), axis=(0, 1))
                link_ids = poi_ls[..., direction]
                q = qs[link_ids]
                link_velocity = (
                    self.cxs[ibar] * ux_ls[link_ids]
                    + self.cys[ibar] * uy_ls[link_ids]
                )
                h_plus = jnp.roll(h_bar, (-cx, -cy), axis=(0, 1))
                surface_far = jnp.where(
                    q <= 0.5,
                    2.0 * q * h_bar
                    + (1.0 - 2.0 * q) * h_plus
                    - 6.0 * wi[ibar] * link_velocity,
                    h_bar / (2.0 * q)
                    + (1.0 - 1.0 / (2.0 * q)) * h[..., direction]
                    - 3.0 / q * wi[ibar] * link_velocity,
                )
                surface_near = h_bar - 6.0 * wi[ibar] * link_velocity
                value = jnp.where(boundary == 1, bulk, 0.0j)
                value = jnp.where(boundary == 4, surface_far, value)
                value = jnp.where(boundary == 5, surface_near, value)
                streamed.append(value)
                if self.has_links[direction]:
                    positions = self.flat_link_positions[direction]
                    ids = self.flat_link_ids[direction]
                    momentum = (value + h_bar).reshape(-1)[positions]
                    fx_ls = fx_ls.at[ids].set(momentum * cx)
                    fy_ls = fy_ls.at[ids].set(momentum * cy)

            h_streamed = jnp.stack(streamed, axis=-1)
            density = jnp.sum(h_streamed, axis=-1)
            ux = jnp.einsum("xyi,i->xy", h_streamed, cxs_array)
            uy = jnp.einsum("xyi,i->xy", h_streamed, cys_array)
            equilibrium = wi * (
                density[..., None]
                + 3.0
                * (
                    ux[..., None] * cxs_array
                    + uy[..., None] * cys_array
                )
            )
            partner_h = h_streamed[..., ibars_array]
            partner_equilibrium = equilibrium[..., ibars_array]
            symmetric_h = (h_streamed + partner_h) / 2.0
            asymmetric_h = (h_streamed - partner_h) / 2.0
            symmetric_equilibrium = (equilibrium + partner_equilibrium) / 2.0
            asymmetric_equilibrium = (equilibrium - partner_equilibrium) / 2.0
            relaxed = (
                self.one_m_tau_invs_m_iom[..., None] * symmetric_h
                + self.tau_invs[..., None] * symmetric_equilibrium
                + self.one_m_tau_invs_m_iom_asym[..., None] * asymmetric_h
                + self.tau_invs_asym[..., None] * asymmetric_equilibrium
            )
            h_next = jnp.where(outside[..., None] == 0, relaxed, h)
            return h_next, fx_ls, fy_ls

        return step

    def step(self, h, ux_ls, uy_ls):
        put = lambda value: self.jax.device_put(value, self.device)
        h_next, fx_ls, fy_ls = self._compiled_step(
            put(h), put(ux_ls), put(uy_ls)
        )
        return h_next, np.asarray(fx_ls), np.asarray(fy_ls)

    def _compile_batch(self, batch_size):
        jax = self.jax
        jnp = self.jnp
        step = self._build_step()

        def run_batch(h, ux_ls, uy_ls):
            zero_fx = jnp.zeros(self.n_links, dtype=jnp.complex128)
            zero_fy = jnp.zeros(self.n_links, dtype=jnp.complex128)

            def advance(carry, _):
                current_h, _, _ = carry
                next_h, fx, fy = step(current_h, ux_ls, uy_ls)
                return (next_h, fx, fy), (fx, fy)

            (final_h, final_fx, final_fy), force_history = jax.lax.scan(
                advance, (h, zero_fx, zero_fy), None, length=batch_size
            )
            return final_h, force_history[0], force_history[1], final_fx, final_fy

        return jax.jit(run_batch, donate_argnums=(0,))

    def step_batch(self, h, ux_ls, uy_ls, batch_size):
        """Advance several ring-in steps on device and transfer link forces once."""
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size not in self._compiled_batches:
            self._compiled_batches[batch_size] = self._compile_batch(batch_size)
        put = lambda value: self.jax.device_put(value, self.device)
        h_next, fx_history, fy_history, _, _ = self._compiled_batches[batch_size](
            put(h), put(ux_ls), put(uy_ls)
        )
        return h_next, np.asarray(fx_history), np.asarray(fy_history)
