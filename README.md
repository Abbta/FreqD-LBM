# Frequency-Domain Lattice Boltzmann (FreqD-LBM) Simulation Software

This github repository contains software that implements frequency-domain lattice-Boltzmann simulation method described in the article
*The Frequency-Domain Lattice Boltzmann Method (FreqD-LBM): A Versatile Tool to Predict the QCM
Response Induced by Structured Samples* by Diethelm Johannsmann, Paul Häusner, Arne Langhoff,
Christian Leppin, Ilya Reviakine, and Viktor Vanoppen. 

## Reference guide

The reference guide is available in the file [FLBMHelp.pdf](FLBMHelp.pdf)

## JSON configuration entrypoint

Run one case with `python Main_FreqDLBM_config_file.py path/to/config.json`.
The JSON is a flat object of scalar `SPs` parameters; omitted fields use defaults.
For example, `{"ProblemType":"StiffParticles","n":7,"RSph_nm":5}` runs one
sphere case. Sweep arrays and repeat counts are not accepted. The legacy result
writer still receives one-value metadata internally. `Cylinder2D` has centered-circle
geometry with periodic boundaries in the physical x-z plane. It uses the existing
ring-in and D2Q9 collision path. Its output is lattice-unit force per length on the
cylinder by the liquid; friction per length is minus that force divided by velocity.
`Do_SavePlots=true` saves a complex-velocity field plot. Cylinder plots are off by
default. Numerical verification remains to be done.

`CylinderQCM3D` adds one finite cylinder standing on the QCM with its axis along
`y`, normal to the plate, and its circular base attached at `y=0`. Motion is along
`x`; the cylinder and QCM move together. Streaming is
periodic in `x` and `z`; the established QCM bottom, Fourier top, D3Q19,
interpolated-boundary, reference-subtraction, and ring-in paths are reused.
`RCyl_nm`, `HCyl_nm`, `CylBoxWidth_nm`, and `CylBoxHeight_nm` define the geometry.
The lateral periods are equal because the existing Fourier-top implementation
assumes `nx == nz`.
CPU kernels use Numba's on-disk cache. Cylinder masks and final field diagnostics
are precomputed/vectorized; the single-case solver remains serial.
All production paths require TRT with `Lambda_TRT=0.25`; conflicting configs fail.
Set `ComputeBackend="jax"` to run `Cylinder2D` (D2Q9) or the 3D
oscillating-boundary step (D3Q19) with JAX on `JAXDevice="auto"`, `"cpu"`, or
`"gpu"`. Double precision is mandatory. The default remains `"numba"`; JAX
  does not change geometry, TRT collision, interpolated cylinder boundaries,
  convergence, or output handling. The 2D cylinder JAX step preserves its periodic
  in-plane boundaries and does not use the 3D Fourier-top operator.
  Set `SaveCylinder3DFields=true` for `CylinderQCM3D` to save the converged complex
  velocity, density perturbation, and solid/cylinder masks in `cylinder3d-fields.npz`.
Install the CPU validation dependency with `requirements-jax-cpu.txt`. On Pelle,
install the JAX GPU wheel matching the loaded CUDA version before selecting `"gpu"`;
the solver fails instead of silently falling back when no GPU is available.
`RingInFitIntervalSteps=0` retains the normal size-derived fitting cadence;
positive values explicitly set the cadence for bounded runtime experiments.
`MaxRingInSteps=0` disables the step cap; a positive value stops after recording
that many steps, primarily for matched backend diagnostics.
If ring-in stops, stdout reports whether the raw response was unstable, the time
limit was exceeded, or both, together with the raw response and fit drift.
`RingInResponseLimitFactor` bounds the raw response relative to the liquid
reference (or the first finite response when no reference exists); its default is
`1e4`. `SaveRingInHistory=true` writes compact raw, reference-subtracted, and fit
histories as `.npz` files.
`ReferenceMethod="direct"` solves the cylinder-free reference's stationary D3Q19
equations as a sparse linear system; the default `"ring_in"` keeps the original
iteration. The direct option supports 3D cases and records its residual and
reference load. The loaded cylinder still uses ring-in and may independently fail.
Set `ReferenceStabilityDiagnostic=true` to run only the cylinder-free D3Q19
reference analysis. It compares a matrix-free perturbation power iteration with
the dominant eigenvalue of the reduced affine update and saves the histories as
`<fname0>_reference_stability.npz`.
