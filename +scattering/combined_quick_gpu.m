% COMBINED_QUICK_GPU Apply filtered pulse-echo measurement matrix or its adjoint
%
%    CUDA/C++ implementation of
%    the matrix-matrix products between
%    (i) the filtered pulse-echo measurement matrix (N_observations x N_points) or
%    (ii) its adjoint (N_points x N_observations_sel) with
%    a suitable matrix.
%
%    The calling syntax is:
%
%      output_matrix = combined_quick_gpu( operator_born, mode, input_matrix )
%
%   INPUTS:
%
%     operator_born:	object of class scattering.operator_born (scalar)
%     mode:             mode of operation (1 = direct, 2 = adjoint)
%     input_matrix:     mode = 1: relative spatial fluctuations in compressibility (N_points x N_objects)
%                       mode = 2: mixed RF voltage signals (N_observations x N_objects)
%
%   OUTPUT:
%
%     output_matrix     mode = 1: mixed RF voltage signals (N_observations x N_objects)
%                       mode = 2: adjoint relative spatial fluctuations in compressibility (N_points x N_objects)
%
%   REFERENCES:
%
%     [1] M. F. Schiffner, "Random Incident Waves for Fast Compressed Pulse-Echo Ultrasound Imaging,"
%         <a href="matlab:web('https://arxiv.org/abs/1801.00205')">physics.med-ph:arXiv:1801.00205</a>
%
%   REMARKS:
%
%     - Requires the interleaved complex API introduced in MATLAB Version 9.4 (R2018a),
%       see <a href="matlab:web('https://mathworks.com/help/matlab/matlab_external/matlab-support-for-interleaved-complex.html')">MATLAB Support for Interleaved Complex API in MEX Functions</a>
%     - Compile using: nvmex_linux( 'combined_quick_gpu.cu' )
%     - Created and tested with MATLAB R2019b / R2020a; CUDA Toolkit 10.1; Ubuntu 16.04 / 18.04
%
%   author: Martin F. Schiffner
%   date: 2019-06-29
%   modified: 2020-06-18
%
%   MEX File function.
