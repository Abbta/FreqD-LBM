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
Cylinder plots are off by default. Numerical verification remains to be done.
