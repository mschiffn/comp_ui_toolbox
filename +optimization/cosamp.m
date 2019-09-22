function [ theta_recon, y_m_res, info ] = cosamp( op_A, y_m, options )
%
% compressive sampling matching pursuit (CoSaMP) to recover a sparse coefficient vector
% (cf. Algorithm 1 in
%   D. Needell, J. A. Tropp, "CoSaMP: Iterative signal recovery from incomplete and inaccurate samples,"
%   Appl. Comput. Harmon. A., 2009, vol. 26, no. 3, pp. 301-321, DOI: 10.1016/j.acha.2008.07.002
% )
%
% author: Martin F. Schiffner
% date: 2013-05-13
% modified: 2019-05-09
%

    % print status
	time_start = tic;
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );
    fprintf( ' %s (%s)\n', 'CoSaMP v. 0.1', str_date_time );
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

    % ensure class optimization.options.algorithm_cosamp
    if ~isa( options, 'optimization.options.algorithm_cosamp' )
        errorStruct.message = 'options must be optimization.options.algorithm_cosamp!';
        errorStruct.identifier = 'omp:NoOptionsAlgorithmCoSaMP';
        error( errorStruct );
    end

    % initialize parameters
    N_iterations_max = options.max_iterations;
    sparsity = options.cosamp_sparsity;

    atoms = [];
    N_atoms = 0;
    A = [];

    y_m_res = y_m;
    y_m_energy = sqrt(sum(abs(y_m).^2));
    theta_recon_cosamp = zeros(N_lattice, 1);

    % recover basis coefficients
    for k_iter = 1:N_iterations_max

        fprintf('iteration %d of %d...\n', k_iter, N_iterations_max);

        %------------------------------------------------------------------
        % 1.) select atoms
        %------------------------------------------------------------------
        % a.) apply adjoint operator (form signal proxy)
        if isnumeric( op_A )
            temp = op_A' * y_m_res;
        else
            temp = op_A( y_m_res, 2 );
        end

        % b.) identify the 2s largest components
        [ ~, indices ] = sort( abs( temp( : ) ), 'descend' );
        atoms_identified = indices( 1:2*sparsity );

        % c.) merge supports
        atoms_old = atoms;
        [ atoms, indices_old, indices_new ] = union( atoms_old, atoms_identified );
        N_atoms = numel( atoms );
        atoms_old = atoms_old( indices_old );
        atoms_new = atoms_identified( indices_new );

        % d.) extract novel column vectors from sensing matrix
% TODO: batch processing
        A_new = zeros( N_observations, numel(indices_new) );
        for index_atom_new = 1:numel( indices_new )
            temp = zeros( N_coefficients, 1 );
            temp( atoms_new( index_atom_new ) ) = 1;

            A_new( :, index_atom_new ) = op_A( temp, 1 );
        end

        % select column vectors of old atoms
        A_old = A( :, indices_old );
    
        % construct novel dictionary matrix
        A = zeros(numel(y_m), N_atoms);
        if k_iter ~= 1
            [indicator, index_in_atoms] = ismember(atoms_old, atoms);
            A(:, index_in_atoms) = A_old;
        end
        [indicator, index_in_atoms] = ismember(atoms_new, atoms);
        A(:, index_in_atoms) = A_new;

        % find optimal values of coefficients (least-squares approximation)
        %coeff = (A' * A) \ (A' * y_m);
        coeff = A \ y_m;
        theta_recon_cosamp( atoms ) = coeff;
    
        % prune atoms (retain s largest entries)
        [ ~, indices ] = sort( abs( theta_recon_cosamp( atoms ) ), 'descend' );
        theta_recon_cosamp = zeros( N_coefficients, 1 );
        theta_recon_cosamp( atoms( indices( 1:sparsity ) ) ) = coeff( indices( 1:sparsity ) );
    
        % compute residual
        % A * theta_recon_cosamp
        y_approx = opphi(theta_recon_cosamp, 1);
        y_m_res = y_m - y_approx;
        y_m_res_energy_cosamp = norm( y_m_res, 2 ) / y_m_energy;
    
        fprintf('rel. error: %.2f\n', y_m_res_energy_cosamp);

        if y_m_res_energy_cosamp <= options.rel_RMSE
            break;
        end

    end % for k_iter = 1:N_iterations_max

end % function [ theta_recon, y_m_res, info ] = cosamp( op_A, y_m, options )
