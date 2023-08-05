# Computational Ultrasound Imaging Toolbox for [MATLAB][mathworks-url]

[mathworks-url]: https://mathworks.com/products/matlab.html

<!-- shields -->
[![GitHub][license-shield]][license-url]
![GitHub repository size][size-shield]
![GitHub language count][languages-shield]
![GitHub stargazers][stars-shield]
![GitHub forks][forks-shield]
[![View on File Exchange][fex-shield]][fex-url]
[![ko-fi][ko-fi-shield]][ko-fi-url]

[license-shield]: https://img.shields.io/badge/license-citationware-blue
[license-url]: https://github.com/mschiffn/comp_ui_toolbox/blob/main/LICENSE
[size-shield]: https://img.shields.io/github/repo-size/mschiffn/comp_ui_toolbox
[languages-shield]: https://img.shields.io/github/languages/count/mschiffn/comp_ui_toolbox
[stars-shield]: https://img.shields.io/github/stars/mschiffn/comp_ui_toolbox.svg
[forks-shield]: https://img.shields.io/github/forks/mschiffn/comp_ui_toolbox.svg
[fex-shield]: https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg
[fex-url]: https://www.mathworks.com/matlabcentral/fileexchange/125285-computational-ultrasound-imaging-toolbox
[ko-fi-shield]: https://img.shields.io/badge/ko--fi-Donate%20a%20coffee-yellowgreen
[ko-fi-url]: https://ko-fi.com/L4L7CCWYS

Develop and
evaluate computational image formation methods for
freely programmable ultrasound imaging systems with
only a few lines of
code.

## Motivation

Advances in
electronic miniaturization and
processing power have recently led to
freely programmable ultrasound imaging (UI) systems and
software-based "ultrafast" imaging modes, such as

- coherent plane-wave compounding,
- synthetic aperture imaging, or
- limited-diffraction beam imaging.

These imaging modes capture
large fields of view (FOVs) at
rates in the kilohertz range.

Standard image formation algorithms, such as
delay-and-sum (DAS) or
Fourier methods, however, increase
the frame rate at
the expense of
the image quality.
They rely on
relatively simple physical models that
do not support
complex imaging sequences and, thus, neglect
the special abilities of
these systems.

## :mag: What Does This Toolbox Accomplish?

Computational UI methods leverage
the available processing power for
realistic physical models that reflect
the abilities of
freely programmable UI systems.
They recover
acoustic material parameter fluctuations in
a specified FOV from
a relatively short sequence of
arbitrarily complex pulse-echo scattering experiments.

Each experiment comprises

1. the synthesis of
an arbitrary incident wave,
2. the subsequent recording of
the resulting echoes via
a fully-sampled transducer array, and
3. the optional mixing of the recorded echoes into
compound signals.

The toolbox, considering
soft tissue structures as
lossy heterogeneous fluids, provides
numerical solutions to
these inverse problems based on
discretized scattering operators and
their adjoints.
These operators map
the material parameter fluctuations to
the mixed radio frequency voltage signals.

The toolbox excels in
the *repetitive* application of
identical scattering operators in
iterative image formation methods and, thus, complements
popular simulation tools, e.g.
[Field II](https://field-ii.dk/) and
[FOCUS](https://www.egr.msu.edu/~fultras-web/).
It compensates
the relatively costly initialization of
a scattering operator by
a fast evaluation.

Typical applications include

- regularized structured insonification,
- coded excitation,
- compressed sensing / sparse recovery,
- statistical (Bayesian) methods,
- machine learning, and
- physical model in plug-and-play methods.

Usability and
simplicity were
design paradigms.
The toolbox enables
the solution of
complex inverse scattering problems with
only a few lines of
code.

## :star: Main Features

- d-dimensional Euclidean space (d = 2, 3)
- one type of heterogeneous acoustic material parameter: compressibility
- modular object-oriented design
- arbitrary dispersion relations describing
  the combination of
  frequency-dependent absorption and
  dispersion, such as
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
- GPU support via mex / CUDA API

## Current Limitations

- Born approximation
- linear systems (wave propagation, scattering, transducer behavior)
- pulse-echo mode (i.e., no transmission measurements)
- half-space with rigid (Neumann) boundary
- symmetric grids
- developed and tested in MATLAB R2018b, R2019a, R2020a / CUDA Toolkit v10.1.168 on Ubuntu 12.04/16.04/18.04

## :gear: Installation

## :notebook: References

The physical models underlying this toolbox and exemplary images were published in:

1. M. F. Schiffner, "Random Incident Waves for Fast Compressed Pulse-Echo Ultrasound Imaging", [![physics.med-ph:arXiv:1801.00205](https://img.shields.io/badge/physics.med--ph-arXiv%3A1801.00205-B31B1B)](https://arxiv.org/abs/1801.00205 "Preprint on arXiv.org")
2. M. F. Schiffner and G. Schmitz, "Compensating the Combined Effects of Absorption and Dispersion in Plane Wave Pulse-Echo Ultrasound Imaging Using Sparse Recovery", 2013 IEEE Int. Ultrasonics Symp. (IUS), pp. 573--576, [![DOI:10.1109/ULTSYM.2013.0148](https://img.shields.io/badge/DOI-10.1109%2FULTSYM.2013.0148-blue)](http://dx.doi.org/10.1109/ULTSYM.2013.0148)
3. M. F. Schiffner and G. Schmitz, "The Separate Recovery of Spatial Fluctuations in Compressibility and Mass Density in Plane Wave Pulse-Echo Ultrasound Imaging", 2013 IEEE Int. Ultrasonics Symp. (IUS), pp. 577--580, [![DOI:10.1109/ULTSYM.2013.0149](https://img.shields.io/badge/DOI-10.1109%2FULTSYM.2013.0149-blue)](http://dx.doi.org/10.1109/ULTSYM.2013.0149)
