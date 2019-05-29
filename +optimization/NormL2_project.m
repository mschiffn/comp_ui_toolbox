function [x itn] = NormL2_project(b, weights, tau)
% NormL2_project  Projects x onto the weighted two-norm ball of radius tau
%
%    [X,ITN] = ONEPROJECTOR(B,TAU) returns the orthogonal projection
%    of the vector b onto the one-norm ball of radius tau. The return
%    vector X which solves the problem
%
%            minimize  ||b-x||_2  st  ||x||_1 <= tau.
%               x
%
%    [X,ITN] = ONEPROJECTOR(B,D,TAU) returns the orthogonal
%    projection of the vector b onto the weighted one-norm ball of
%    radius tau, which solves the problem
%
%            minimize  ||b-x||_2  st  || Dx ||_1 <= tau.
%               x
%
%    If D is empty, all weights are set to one, i.e., D = I.
%
%    In both cases, the return value ITN given the number of elements
%    of B that were thresholded.
%
% See also spgl1.

% Initialization
n     = length(b);
x     = zeros(n,1);
bNorm = norm(b,2);

% Check for quick exit.
if (tau >= bNorm), x = b; itn = 0; return; end
if (tau <  eps  ),        itn = 0; return; end

b = b / bNorm;
x = b * tau;
