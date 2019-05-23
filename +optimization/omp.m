function [ theta_recon, y_m_res, info ] = omp( op_A, y_m, options )
%
% orthogonal matching pursuit (OMP) to recover a sparse coefficient vector
% OMP never selects the same atom twice
%
% author: Martin F. Schiffner
% date: 2011-06-15
% modified: 2019-05-09

% start time measurement
time_start = tic;

%--------------------------------------------------------------------------
% 1.) check arguments
%--------------------------------------------------------------------------
% size of the linear operator
if isnumeric( op_A )
    size_A = size( op_A );
else
    size_A = op_A( [], 0 );
end
N_observations = size_A( 1 );
N_coefficients = size_A( 2 );

% input vector and residual
y_m = y_m( : );
y_m_norm = norm( y_m, 2 );

% maximum number of iterations
if isempty( options.max_iterations ) || options.max_iterations > N_observations
	N_iterations_max = N_observations;
else
    N_iterations_max = options.max_iterations;
end

%--------------------------------------------------------------------------
% 2.) initialization
%--------------------------------------------------------------------------
% initialize reduced sensing matrix with selected columns
A_sel = zeros( N_observations, N_iterations_max );

% initialize atoms
atoms = zeros( 1, N_iterations_max );

% initialize residual and its norms
y_m_res = y_m;
y_m_res_norm_rel = zeros( 1, N_iterations_max );

% initialize initial estimate
theta_recon = zeros( N_coefficients, 1 );

%--------------------------------------------------------------------------
% 3.) recover transform coefficients
%--------------------------------------------------------------------------
for k_iter = 1:N_iterations_max
    
%     fprintf('iteration %d of %d...\n', k_iter, N_iterations_max);
    
    %----------------------------------------------------------------------
    % 1.) select new atom
    %----------------------------------------------------------------------
    % a.) apply adjoint operator
    if isnumeric( op_A )
        temp = op_A' * y_m_res;
    else
        temp = op_A( y_m_res, 2 );
    end

    % b.) select largest element and save its index
    [ val_abs_max, atoms( k_iter ) ] = max( abs( temp( : ) ) );

    % c.) extract column vector from sensing matrix
    if isnumeric( op_A )
        A_sel( :, k_iter ) = op_A( :, atoms( k_iter ) );
    else
        temp = zeros( N_coefficients, 1 );
        temp( atoms( k_iter ) ) = 1;

        A_sel( :, k_iter ) = op_A( temp, 1 );
    end

%     figure(k_iter);
%     subplot(1,2,1);
%     imagesc(reshape(abs(temp), [N_lattice_z, N_lattice_x]));

    %----------------------------------------------------------------------
    % 2.) find optimal values of coefficients (least-squares approximation of original y_m)
    %----------------------------------------------------------------------
    coeff = ( A_sel( :, 1:k_iter )' * A_sel( :, 1:k_iter ) ) \ ( A_sel( :, 1:k_iter )' * y_m );
    theta_recon( atoms( 1:k_iter ) ) = coeff;

    %----------------------------------------------------------------------
    % 3.) compute residual
    %----------------------------------------------------------------------
    y_approx = A_sel( :, 1:k_iter ) * coeff;
    y_m_res = y_m - y_approx;
    y_m_res_norm_rel( k_iter ) = norm( y_m_res, 2 ) / y_m_norm;
    
%     subplot(1,2,2);
%     imagesc(reshape(abs(theta_recon), [N_lattice_z, N_lattice_x]));
    
%     fprintf( 'rel. error: %.6f\n', y_m_res_norm_rel( k_iter ) );
    
    %----------------------------------------------------------------------
    % 4.) stopping criterion
    %----------------------------------------------------------------------
    if y_m_res_norm_rel( k_iter ) <= options.rel_rmse    
       break;
    end

end

% truncate data structures to actual number of iterations
y_m_res_norm_rel = y_m_res_norm_rel( 1:k_iter );
atoms = atoms( 1:k_iter );

% stop time measurement
time_elapsed = toc( time_start );

%--------------------------------------------------------------------------
% 4.) return info structure
%--------------------------------------------------------------------------
info.size_A = size_A;
info.atoms = atoms;
info.y_m_res_norm_rel = y_m_res_norm_rel;
info.time_total = time_elapsed;