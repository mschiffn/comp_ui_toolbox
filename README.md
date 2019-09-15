## Computational Ultrasound Imaging Toolbox for MATLAB

This MATLAB (The MathWorks, Inc., Natick, MA, USA) toolbox facilitates
the development and evaluation of
computational ultrasound imaging methods for
freely programmable UI systems.

These methods recover
the acoustic material parameters in
a specified field of view from
a relatively short sequence of
pulse-echo scattering experiments.
Each experiment comprises
(i) the synthesis of
an arbitrary incident wave,
(ii) the subsequent recording of
the resulting echoes via
a fully-sampled transducer array, and
(iii) their optional mixing into
compound signals.

Considering soft tissue structures as
lossy heterogeneous fluids,
the toolbox provides numerical solutions to
these inverse problems based on
discretized scattering operators and
their adjoints.
These operators map
the relative spatial fluctuations in
compressibility and/or mass density to
the mixed RF voltage signals.

Usability and simplicity were
crucial design paradigms.

Typical applications include
- regularized structured insonification,
- coded excitation, and
- compressed sensing / sparse recovery.

The toolbox excels at
the efficient repetitive evaluation of
identical scattering operators in
iterative algorithms.
This goal contrasts
popular simulation tools, e.g.
Field II (Jensen et al.) and
FoCUS (Michigan), which permit
the rapid evaluation of
various parameters.
The setup and
precomputation steps are relatively costly, whereas
the evaluation is much faster.
Although
the computational costs and
the memory consumption for

**Main Features**:

- d-dimensional Euclidean space (d = 2, 3)
- two types of heterogeneous acoustic material parameters: compressibility and mass density
- arbitrary dispersion relations describing
  the combination of
  frequency-dependent absorption and
  dispersion, e.g.
  the time-causal model
- arbitrary types of incident waves, including
  - steered quasi-plane waves,
  - quasi-(d-1)-spherical waves with virtual sources,
  - focused beams,
  - random waves, and
  - coded waves
- regularization based on
  lq-minimization (convex and nonconvex)
- efficient implementations using
  hierarchical matrix factorizations
- multi GPU support via mex / CUDA API

**Current Limitations**:

- Born approximation (future releases might support Rytov, WKB, Padé, and full wave solutions)
- linear systems (wave propagation, transfer behavior)
- pulse-echo mode, i.e. no transmission measurements
- half-space with rigid boundary
- symmetric grids (future releases might support the fast multipole method (FMM) and adaptive cross approximation (ACA))
- developed and tested in MATLAB R2018b / CUDA Toolkit v10.1.168 on Ubuntu 12.04/16.04/18.04

**Relationship to popular software **


**References**:

The physical models underlying this toolbox and exemplary images were published in:

[1] M. F. Schiffner, "Random Incident Waves for Fast Compressed Pulse-Echo Ultrasound Imaging", arXiv:1801.00205 [physics.med-ph]
[2] M. F. Schiffner and G. Schmitz, "Compensating the Combined Effects of Absorption and Dispersion in Plane Wave Pulse-Echo Ultrasound Imaging Using Sparse Recovery", 2013 IEEE Int. Ultrasonics Symp. (IUS), pp. 573--576, DOI: 10.1109/ULTSYM.2013.0148
[3] M. F. Schiffner and G. Schmitz, "The Separate Recovery of Spatial Fluctuations in Compressibility and Mass Density in Plane Wave Pulse-Echo Ultrasound Imaging", 2013 IEEE Int. Ultrasonics Symp. (IUS), pp. 577--580, DOI: 10.1109/ULTSYM.2013.0149

