function [ theta_recon, y_m_res, info ] = omp( op_A, y_m, options )
%
% orthogonal matching pursuit (OMP) recovers sparse coefficient vectors
% it never selects the same atom twice
% (cf. Sect. II.C.2) in [1])
%
% [1] J. A. Tropp, "Greed is Good: Algorithmic Results for Sparse Approximation",
%     IEEE Trans. Inf. Theory, Oct. 2004, Vol. 50, No. 10, pp. 2231-2242
%
% author: Martin F. Schiffner
% date: 2011-06-15
% modified: 2020-01-03
%

	% print status
	time_start = tic;
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );
    fprintf( ' %s (%s)\n', 'OMP v. 1.0', str_date_time );
    fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );

    %----------------------------------------------------------------------
    % 1.) check arguments
    %----------------------------------------------------------------------
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

    % ensure class regularization.options.algorithm_omp
    if ~isa( options, 'regularization.options.algorithm_omp' )
        errorStruct.message = 'options must be regularization.options.algorithm_omp!';
        errorStruct.identifier = 'omp:NoOptionsAlgorithmOMP';
        error( errorStruct );
    end

    % maximum number of iterations
    N_iterations_max = min( options.N_iterations_max, N_observations );

    strings = { 'number of rows', 'number of columns'; 'y_m_norm', 'number of iterations' };
    N_chars_max = max( max( cellfun( @numel, strings ) ) );
	fprintf( ' %-20s: %7d %10s %-30s: %7d\n', 'number of rows', N_observations, '', 'number of columns', N_coefficients );
	fprintf( ' %-20s: %7.2f %10s %-30s: %7d\n', 'y_m_norm', y_m_norm, '', 'number of iterations', N_iterations_max );
    fprintf( ' %-20s: %7.2f\n', 'objective', options.rel_RMSE );
    fprintf( ' %s\n', repmat( '-', [ 1, 80 ] ) );

    %----------------------------------------------------------------------
    % 2.) initialization
    %----------------------------------------------------------------------
	% initialize reduced sensing matrix with selected columns
    A_sel = zeros( N_observations, N_iterations_max );

    % initialize atoms
    atoms = zeros( 1, N_iterations_max );

    % initialize residual and its norms
    y_m_res = y_m;
    y_m_res_norm_rel = zeros( 1, N_iterations_max );

    % initialize initial estimate
    theta_recon = zeros( N_coefficients, 1 );

    %----------------------------------------------------------------------
    % 3.) recover transform coefficients
    %----------------------------------------------------------------------
	fprintf( ' %-10s %-10s\n', 'Iter', 'rel. RMSE' );
	for k_iter = 1:N_iterations_max

        %------------------------------------------------------------------
        % 1.) select new atom
        %------------------------------------------------------------------
        % a.) apply adjoint operator
        if isnumeric( op_A )
            temp = op_A' * y_m_res;
        else
            temp = op_A( y_m_res, 2 );
        end

        % b.) select largest element and save its index
        [ ~, atoms( k_iter ) ] = max( abs( temp( : ) ) );

        % c.) extract column vector from sensing matrix
        if isnumeric( op_A )
            A_sel( :, k_iter ) = op_A( :, atoms( k_iter ) );
        else
            temp = zeros( N_coefficients, 1 );
            temp( atoms( k_iter ) ) = 1;

            A_sel( :, k_iter ) = op_A( temp, 1 );
        end

%         figure( k_iter );
%         subplot( 1, 2, 1 );
%         imagesc( reshape( abs( temp ), [ N_lattice_z, N_lattice_x] ) );

        %------------------------------------------------------------------
        % 2.) find optimal values of coefficients (least-squares approximation of original y_m)
        %------------------------------------------------------------------
        coeff = ( A_sel( :, 1:k_iter )' * A_sel( :, 1:k_iter ) ) \ ( A_sel( :, 1:k_iter )' * y_m );
        theta_recon( atoms( 1:k_iter ) ) = coeff;

        %------------------------------------------------------------------
        % 3.) compute residual
        %------------------------------------------------------------------
        y_approx = A_sel( :, 1:k_iter ) * coeff;
        y_m_res = y_m - y_approx;
        y_m_res_norm_rel( k_iter ) = norm( y_m_res, 2 ) / y_m_norm;

        axis = math.sequence_increasing_regular( 193, 577, physical_values.hertz( 11685.655857435000143595971167087554931640625 ) );
        y_m_tilde = signal( discretizations.signal_matrix( axis, physical_values.volt( reshape( y_m, [ 385, 128 ] ) ) ), 1400, physical_values.second( 1/40e6 ) );
        y_approx_tilde = signal( discretizations.signal_matrix( axis, physical_values.volt( reshape( y_approx, [ 385, 128 ] ) ) ), 1400, physical_values.second( 1/40e6 ) );
        y_m_res_tilde = signal( discretizations.signal_matrix( axis, physical_values.volt( reshape( y_m_res, [ 385, 128 ] ) ) ), 1400, physical_values.second( 1/40e6 ) );
        y_m_tilde_max = max( abs( y_m_tilde.samples(:) ) );

        figure( k_iter );
        subplot( 2, 3, 1 );
        imagesc( 20*log10( abs( y_m_tilde.samples ) / y_m_tilde_max ), [ -50, 0 ] );
        subplot( 2, 3, 2 );
        imagesc( 20*log10( abs( y_approx_tilde.samples ) / y_m_tilde_max ), [ -50, 0 ] );
        subplot( 2, 3, 3 );
        imagesc( 20*log10( abs( y_m_res_tilde.samples ) / y_m_tilde_max ), [ -50, 0 ] );
        subplot( 2, 3, 4 );
        plot( (y_m_tilde.axis.q_lb:y_m_tilde.axis.q_ub), y_m_tilde.samples( :, 32 ), 'b', ...
              (y_approx_tilde.axis.q_lb:y_approx_tilde.axis.q_ub), y_approx_tilde.samples( :, 32 ), 'g', ...
              (y_m_res_tilde.axis.q_lb:y_m_res_tilde.axis.q_ub), y_m_res_tilde.samples( :, 32 ), 'r:' );
        subplot( 2, 3, 5 );
        plot( (y_m_tilde.axis.q_lb:y_m_tilde.axis.q_ub), y_m_tilde.samples( :, 64 ), 'b', ...
              (y_approx_tilde.axis.q_lb:y_approx_tilde.axis.q_ub), y_approx_tilde.samples( :, 64 ), 'g', ...
              (y_m_res_tilde.axis.q_lb:y_m_res_tilde.axis.q_ub), y_m_res_tilde.samples( :, 64 ), 'r:' );
        subplot( 2, 3, 6 );
        plot( (y_m_tilde.axis.q_lb:y_m_tilde.axis.q_ub), y_m_tilde.samples( :, 96 ), 'b', ...
              (y_approx_tilde.axis.q_lb:y_approx_tilde.axis.q_ub), y_approx_tilde.samples( :, 96 ), 'g', ...
              (y_m_res_tilde.axis.q_lb:y_m_res_tilde.axis.q_ub), y_m_res_tilde.samples( :, 96 ), 'r:' );
        colormap parula;

%         subplot(1,2,2);
%         imagesc(reshape(abs(theta_recon), [N_lattice_z, N_lattice_x]));

        % print status
        fprintf( ' %10d %10.4f\n', k_iter, y_m_res_norm_rel( k_iter ) );

        %------------------------------------------------------------------
        % 4.) stopping criterion
        %------------------------------------------------------------------
        if y_m_res_norm_rel( k_iter ) <= options.rel_RMSE
            break;
        end

    end % for k_iter = 1:N_iterations_max

    % truncate data structures to actual number of iterations
    atoms = atoms( 1:k_iter );
    y_m_res_norm_rel = y_m_res_norm_rel( 1:k_iter );

    % stop time measurement
    time_elapsed = toc( time_start );

    %----------------------------------------------------------------------
    % 4.) return info structure
    %----------------------------------------------------------------------
    info.size_A = size_A;
    info.atoms = atoms;
    info.y_m_res_norm_rel = y_m_res_norm_rel;
    info.time_total = time_elapsed;

end % function [ theta_recon, y_m_res, info ] = omp( op_A, y_m, options )
