# Computational Ultrasound Imaging Toolbox for [MATLAB](mathworks-url)

[mathworks-url]: https://mathworks.com/products/matlab.html

[![GitHub](license-image)](license-url)
[![GitHub](downloads-image)](downloads-url)
![GitHub repo size](https://img.shields.io/github/repo-size/mschiffn/comp_ui_toolbox)
![GitHub All Releases](https://img.shields.io/github/downloads/mschiffn/comp_ui_toolbox/total)

[license-image]: https://img.shields.io/github/license/mschiffn/comp_ui_toolbox
[license-url]: https://github.com/mschiffn/comp_ui_toolbox/COPYING
[downloads-image]: https://img.shields.io/github/downloads/mschiffn/comp_ui_toolbox/total
[downloads-url]: https://npmjs.org/package/ieee754

Develop and
evaluate computational image recovery methods for
freely programmable ultrasound imaging systems with
only a few lines of
code.

## Motivation

Advances in
electronic miniaturization and
processing power have recently led to
freely programmable ultrasound imaging (UI) systems and
software-based "ultrafast" imaging modes, e.g.

- coherent plane-wave compounding,
- synthetic aperture imaging, or
- limited-diffraction beam imaging, that

capture large fields of view (FOVs) at
rates in the kilohertz range.

Established image recovery methods, e.g.
delay-and-sum (DAS) or
Fourier methods, however, gradually trade
the image quality for
the frame rate.
They rely on
relatively simple physical models that
do not support
complex imaging sequences and, thus, significantly reduce
the flexibility of
these systems.

## What Does This Toolbox Accomplish?

Computational UI methods leverage
the available processing power for
realistic physical models that reflect
the abilities of
freely programmable UI systems.
They recover
fluctuations in
the acoustic material parameters in
a specified FOV from
a relatively short sequence of
arbitrarily complex pulse-echo scattering experiments.

Each experiment comprises

1. the synthesis of
an arbitrary incident wave,
2. the subsequent recording of
the resulting echoes via
a fully-sampled transducer array, and
3. their optional mixing into
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
the mixed radio frequency voltage signals.

The toolbox excels in
the *repetitive* application of
identical scattering operators in
iterative image recovery methods and, thus, complements
popular simulation tools, e.g.
[Field II](https://field-ii.dk/) and
[FOCUS](https://www.egr.msu.edu/~fultras-web/).
It compensates
the relatively costly initialization of
a scattering operator by
an extremely fast evaluation.

Typical applications include

- regularized structured insonification,
- coded excitation,
- compressed sensing / sparse recovery,
- statistical (Bayesian) methods, and
- machine learning.

Usability and simplicity were
crucial design paradigms.
The toolbox enables
the solution of
complex inverse scattering problems with
only a few lines of code.

## Main Features

- d-dimensional Euclidean space (d = 2, 3)
- two types of heterogeneous acoustic material parameters: compressibility and mass density
- modular object-oriented design
- arbitrary dispersion relations describing
  the combination of
  frequency-dependent absorption and
  dispersion, e.g.
  the time-causal model
- arbitrary types of incident waves, including
  - steered quasi-plane waves,
  - quasi-(d-1)-spherical waves with virtual sources,
  - steered and focused beams,
  - random waves, and
  - coded waves
- regularization based on
  lq-minimization (convex and nonconvex)
- efficient implementations using
  hierarchical matrix factorizations
- multi GPU support via mex / CUDA API

## Current Limitations

- Born approximation (future releases might support Rytov, WKB, Padé, and full wave solutions)
- linear systems (wave propagation, scattering, transducer behavior)
- pulse-echo mode, i.e. no transmission measurements
- half-space with rigid (Neumann) boundary
- symmetric grids (future releases might support the fast multipole method and adaptive cross approximation)
- developed and tested in MATLAB R2018b, R2019a, R2020a / CUDA Toolkit v10.1.168 on Ubuntu 12.04/16.04/18.04

## References :notebook:

The physical models underlying this toolbox and exemplary images were published in:

1. M. F. Schiffner, "Random Incident Waves for Fast Compressed Pulse-Echo Ultrasound Imaging", [![physics.med-ph:arXiv:1801.00205](https://img.shields.io/badge/physics.med--ph-arXiv%3A1801.00205-B31B1B)](https://arxiv.org/abs/1801.00205 "Preprint on arXiv.org")
2. M. F. Schiffner and G. Schmitz, "Compensating the Combined Effects of Absorption and Dispersion in Plane Wave Pulse-Echo Ultrasound Imaging Using Sparse Recovery", 2013 IEEE Int. Ultrasonics Symp. (IUS), pp. 573--576, [![DOI:10.1109/ULTSYM.2013.0148](https://img.shields.io/badge/DOI-10.1109%2FULTSYM.2013.0148-blue)](http://dx.doi.org/10.1109/ULTSYM.2013.0148)
3. M. F. Schiffner and G. Schmitz, "The Separate Recovery of Spatial Fluctuations in Compressibility and Mass Density in Plane Wave Pulse-Echo Ultrasound Imaging", 2013 IEEE Int. Ultrasonics Symp. (IUS), pp. 577--580, [![DOI:10.1109/ULTSYM.2013.0149](https://img.shields.io/badge/DOI-10.1109%2FULTSYM.2013.0149-blue)](http://dx.doi.org/10.1109/ULTSYM.2013.0149)
