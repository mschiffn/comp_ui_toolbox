function [ theta_recon, theta_recon_normed, info ] = lq_minimization( operators_born, u_M, options, linear_transform ) % epsilon_n, theta_0
%
% minimize the lq-norm using SPGL1 to recover a sparse coefficient vector
%
% author: Martin F. Schiffner
% date: 2015-06-01
% modified: 2019-05-28
%

% TODO: vectorize

	% start time measurement
	time_start = tic;

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class scattering.operator_born (scalar)
	if ~( isa( operators_born, 'scattering.operator_born' ) && isscalar( operators_born ) )
        errorStruct.message = 'operators_born must be scattering.operator_born!';
        errorStruct.identifier = 'lq_minimization:NoOperatorBorn';
        error( errorStruct );
    end

	% ensure class linear_transforms.linear_transform (scalar)
    if ~( isa( linear_transform, 'linear_transforms.linear_transform' ) && isscalar( linear_transform ) )
        errorStruct.message = 'Psi must be linear_transforms.linear_transform!';
        errorStruct.identifier = 'lq_minimization:NoLinearTransform';
        error( errorStruct );
    end

	% ensure class discretizations.signal_matrix
    if ~isa( u_M, 'discretizations.signal_matrix' )
        errorStruct.message = 'u_M must be discretizations.signal_matrix!';
        errorStruct.identifier = 'lq_minimization:NoSignalMatrix';
        error( errorStruct );
    end

	% ensure class optimization.options_lq_minimization
	if ~isa( options, 'optimization.options_lq_minimization' )
        errorStruct.message = 'options must be optimization.options_lq_minimization!';
        errorStruct.identifier = 'lq_minimization:NoOptions';
        error( errorStruct );
    end

    %----------------------------------------------------------------------
    % 2.) perform lq-minimization
    %----------------------------------------------------------------------
	% normalize mixed voltage signals
	u_M = return_vector( u_M );
	u_M_norm = norm( u_M );
	u_M_normed = u_M / u_M_norm;

	% define anonymous function for sensing matrix
	op_A = @( x, mode ) combined( operators_born, x, mode, linear_transform );

	% create SPGL1 options structure
	spgl_opts = spgSetParms( 'verbosity', 1, 'optTol', 1e-4, 'iterations', options.N_iterations_max );

	% replace projection methods for l2-minimization
	if options.q == 2
        spgl_opts.project = @( x, weight, tau ) optimization.NormL2_project( x, weight, tau );
        spgl_opts.primal_norm = @( x, weight ) optimization.NormL2_primal( x, weight );
        spgl_opts.dual_norm = @( x, weight ) optimization.NormL2_dual( x, weight );
    end

    %----------------------------------------------------------------------
	% l2- or l1-minimization (convex)
	%----------------------------------------------------------------------
	% call SPGL1 with sensing matrix
% 	[ theta_recon_normed, u_M_res, gradient, info ] = spgl1( op_A, u_M_normed, options.tau, options.rel_RMSE, options.x_0 / y_m_l2_norm, spgl_opts );
	[ theta_recon_normed, u_M_res, gradient, info ] = spgl1( op_A, u_M_normed, [], options.rel_RMSE, [], spgl_opts );

	%----------------------------------------------------------------------
	% lq-minimization ( 0 <= q < 1, nonconvex, Foucart's algorithm )
	%----------------------------------------------------------------------
	if options.q < 1

        % determine problem size
        info.size_A = op_A_norm([], [], 0);
        N_samples = info.size_A(2);

        % numer of iterations
        N_iterations = size( epsilon_n, 2 );

        % allocate memory for results
        theta_n = zeros( N_samples, N_iterations + 1 );
        theta_n(:, 1) = theta_0;  % set start vector (minimizer of P_{(1, eta)})

        info.info_inner = cell(1, N_iterations);
        info.N_prod_A = 0;
        info.N_prod_A_adj = 0;
        info.N_iter = 0;

        for index_iter = 1:N_iterations

            % compute diagonal matrix with weights
            theta_normalization = ( abs( theta_n( :, index_iter ) ) + epsilon_n( index_iter ) ).^( 1 - options.q );

            % solve weighted l1-minimization problem using reformulation

                % define normalized sensing operator
                op_A_norm_n = @( x, mode ) op_A_norm( x, theta_normalization, mode );

                % solve normalized l1-minimization problem
                [temp, r, g, spgl1_info] = spgl1( op_A_norm_n, y_m, [], options.rel_RMSE, [], spgl_opts );

            % remove normalization
            theta_n(:, index_iter + 1) = temp .* theta_normalization;

            % statistics for computational costs
            info.info_inner{ index_iter } = spgl1_info;
            info.N_prod_A = info.N_prod_A + spgl1_info.nProdA;
            info.N_prod_A_adj = info.N_prod_A_adj + spgl1_info.nProdAt;
            info.N_iter = info.N_iter + spgl1_info.iter;

        end

        theta_recon_l_q = theta_n(:, end);

    end % if options.q < 1

	% correct normalization
	theta_recon = operator_transform( linear_transform, theta_recon_normed, 1 ) * u_M_norm;

    % stop time measurement (seconds)
	info.time_total = toc( time_start );

end

%                         
%                         % dimensions of sensing matrix
%                         info.size_A = op_phi_mlfma_normed_gpu([], 0);
% 
%                         for index_n = 1:options.l_q_N_iterations
%                 
%                             % modify sensing matrix
%                             theta_normalization = ( abs( theta_recon ) + options.epsilon_n(index_n) ).^(1 - options.q); 
%                             op_phi_current_transform_n = @(x, mode) op_phi_mlfma_normed_gpu_modified_n(x, theta_normalization, mode);
%                 
%                             % solve weighted l1-minimization problem using reformulation
%                             [theta_recon, y_m_res, gradient, info] = spgl1( op_phi_current_transform_n, y_m / y_m_energy, [], options.rel_mse, [], spgl_opts );
%                         
%                             % remove normalization
%                             theta_recon = theta_recon .* theta_normalization;
%                         end
%                     
%                         % correct scaling
%                         theta_recon = theta_recon * y_m_energy;

% %----------------------------------------------------------
%                 % normalize initial conditions if provided and
%                 % options.normalize_columns == true
%                 %----------------------------------------------------------
%                 if (~ isempty(options.x_0)) && options.normalize_columns
%                     
%                     fprintf('normalizing initial conditions...\n');
%                     
%                     if  redundant_dictionary == false  
%                         %no redundant transform
%                         % a value for x_0 is given: denormalize x_0 to match normalized sensing matrix
%                         if options.material_parameter == 0
%                             % both material parameters
%                             options.x_0(1:FOV_N_points) = options.x_0(1:FOV_N_points) .* norms_cols_kappa(:);
%                             options.x_0((FOV_N_points + 1):end) = options.x_0((FOV_N_points + 1):end) .* norms_cols_rho(:);
%                         elseif options.material_parameter == 1
%                             % gamma_kappa only
%                             options.x_0 = options.x_0 .* norms_cols_kappa(:);
%                         else
%                             % gamma_rho only
%                             options.x_0 = options.x_0 .* norms_cols_rho(:);
%                         end
%                     else
%                         % TODO: insert code for redundant dictionary
%                         
%                     end %if redundant_dictionary == false
%                     
%                     fprintf('done!\n');
%                     
%                 end
%                 
%                 
%                     %------------------------------------------------------
%                     % minimize lq-norm (according to Foucart's algorithm)
%                     %------------------------------------------------------
%                     if options.normalize_columns
% 
%                         % compute start vector as solution to l1-minimization problem
%                         [theta_recon_normed, y_m_res, gradient, info] = spgl1( op_A_mlfma_normed_gpu, y_m_normed, [], options.rel_RMSE, [], spgl_opts );
% 
%                         % minimize the lq-norm
%                         [theta_recon_normed, info] = lq_minimization_v2( @op_A_mlfma_normed_gpu_modified_n, y_m_normed, options.q, options.epsilon_n, theta_recon_normed, options.rel_RMSE, spgl_opts );
% 
%                         % correct scaling
%                         theta_recon_normed = theta_recon_normed * y_m_l2_norm;
% 
%                         y_m_res = [];
%                         gradient = [];
%                     else
% 
%                         % compute start vector as solution to l1-minimization problem
%                         [theta_recon, y_m_res, gradient, info] = spgl1( op_A_mlfma_scaled_gpu, y_m_normed, [], options.rel_RMSE, [], spgl_opts );
% 
%                         % minimize the lq-norm
%                         [theta_recon, info] = lq_minimization_v2( @op_A_mlfma_scaled_gpu_modified_n, y_m_normed, options.q, options.epsilon_n, theta_recon, options.rel_RMSE, spgl_opts );
% 
%                         % correct scaling
%                         theta_recon = theta_recon * FOV_N_points;
% 
%                         y_m_res = [];
%                         gradient = [];
%                     end
% 
%                 % compute residue and energy of residue
%                 y_m_res = y_m_res * y_m_l2_norm;                
%                 y_m_res_l2_norm_rel = norm( y_m_res(:), 2 ) / y_m_l2_norm;