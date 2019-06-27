function [ gamma_recon, theta_recon_normed, info ] = lq_minimization( operators_born, u_M, options, varargin )
%
% minimize the lq-norm using SPGL1 to recover a sparse coefficient vector
%
% author: Martin F. Schiffner
% date: 2015-06-01
% modified: 2019-06-25
%

	% start time measurement
% 	time_start = tic;

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class scattering.operator_born
	if ~isa( operators_born, 'scattering.operator_born' )
        errorStruct.message = 'operators_born must be scattering.operator_born!';
        errorStruct.identifier = 'lq_minimization:NoOperatorsBorn';
        error( errorStruct );
    end

	% ensure cell array for u_M
	if ~iscell( u_M )
        u_M = { u_M };
    end

	% ensure cell array for options
	if ~iscell( options )
        options = { options };
    end

	% ensure nonempty linear_transforms
	if nargin >= 4 && ~isempty( varargin{ 1 } )
        linear_transforms = varargin{ 1 };
    else
        % empty linear_transform is identity
        linear_transforms = cell( size( operators_born ) );
    end

	% ensure cell array for linear_transforms
	if ~iscell( linear_transforms )
        linear_transforms = { linear_transforms };
    end

	% multiple operators_born / single u_M
	if ~isscalar( operators_born ) && isscalar( u_M )
        u_M = repmat( u_M, size( operators_born ) );
    end

	% multiple operators_born / single options
	if ~isscalar( operators_born ) && isscalar( options )
        options = repmat( options, size( operators_born ) );
    end

	% multiple operators_born / single linear_transforms
	if ~isscalar( operators_born ) && isscalar( linear_transforms )
        linear_transforms = repmat( linear_transforms, size( operators_born ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( operators_born, u_M, options, linear_transforms );

	%----------------------------------------------------------------------
	% 2.) perform lq-minimization
	%----------------------------------------------------------------------
	% specify cell arrays
	gamma_recon = cell( size( operators_born ) );
	theta_recon_normed = cell( size( operators_born ) );
	info = cell( size( operators_born ) );

	% iterate scattering operators
	for index_object = 1:numel( operators_born )

        %------------------------------------------------------------------
        % a) check mixed voltage signals, options, and linear transform
        %------------------------------------------------------------------
% TODO: u_M{ index_object } may be cell array containing signals or signal_matrices
        % ensure class discretizations.signal_matrix
        if ~isa( u_M{ index_object }, 'discretizations.signal_matrix' )
            errorStruct.message = sprintf( 'u_M{ %d } must be discretizations.signal_matrix!', index_object );
            errorStruct.identifier = 'lq_minimization:NoSignalMatrices';
            error( errorStruct );
        end

        % ensure class optimization.options_lq_minimization
        if ~isa( options{ index_object }, 'optimization.options_lq_minimization' )
            errorStruct.message = sprintf( 'options{ %d } must be optimization.options_lq_minimization!', index_object );
            errorStruct.identifier = 'lq_minimization:NoOptions';
            error( errorStruct );
        end

        % ensure class linear_transforms.linear_transform (scalar)
        if ~( isa( linear_transforms{ index_object }, 'linear_transforms.linear_transform' ) && isscalar( linear_transforms{ index_object } ) )
            errorStruct.message = sprintf( 'linear_transforms{ %d } must be a single linear_transforms.linear_transform!', index_object );
            errorStruct.identifier = 'lq_minimization:NoSingleLinearTransform';
            error( errorStruct );
        end

        %------------------------------------------------------------------
        % b) normalize mixed voltage signals and settings
        %------------------------------------------------------------------
        % normalize mixed voltage signals
        u_M{ index_object } = return_vector( u_M{ index_object } );
        u_M_norm = norm( u_M{ index_object } );
        u_M_normed = u_M{ index_object } / u_M_norm;

        % define anonymous function for sensing matrix
        op_A_bar = @( x, mode ) combined( operators_born( index_object ), x, mode, linear_transforms{ index_object } );

        % specify cell arrays
        gamma_recon{ index_object } = cell( size( options{ index_object } ) );
        theta_recon_normed{ index_object } = cell( size( options{ index_object } ) );
        info{ index_object } = cell( size( options{ index_object } ) );

        % iterate options
        for index_options = 1:numel( options{ index_object } )

            %--------------------------------------------------------------
            % i.) options and start vector
            %--------------------------------------------------------------
            % create SPGL1 options structure
            spgl_opts = spgSetParms( 'verbosity', 1, 'optTol', 1e-4, 'iterations', options{ index_object }( index_options ).N_iterations_max );

            % replace projection methods for l2-minimization
            if options{ index_object }( index_options ).q == 2
                spgl_opts.project = @( x, weight, tau ) optimization.NormL2_project( x, weight, tau );
                spgl_opts.primal_norm = @( x, weight ) optimization.NormL2_primal( x, weight );
                spgl_opts.dual_norm = @( x, weight ) optimization.NormL2_dual( x, weight );
            end

            % create start vector
            if index_options > 1 && isequal( options{ index_object }( index_options ).q, options{ index_object }( index_options - 1 ).q ) && options{ index_object }( index_options ).rel_RMSE < options{ index_object }( index_options - 1 ).rel_RMSE
                x_0 = theta_recon_normed{ index_object }{ index_options - 1 };
                tau = info{ index_object }{ index_options - 1 }.tau;
            else
                x_0 = [];
                tau = [];
            end

            %--------------------------------------------------------------
            % ii.) l2- or l1-minimization (convex)
            %--------------------------------------------------------------
            % call SPGL1 with sensing matrix
            [ theta_recon_normed{ index_object }{ index_options }, u_M_res, gradient, info{ index_object }{ index_options } ] = spgl1( op_A_bar, u_M_normed, tau, options{ index_object }( index_options ).rel_RMSE, x_0, spgl_opts );

            %--------------------------------------------------------------
            % iii.) optional lq-minimization ( 0 <= q < 1, nonconvex, Foucart's algorithm )
            %--------------------------------------------------------------
            if options{ index_object }( index_options ).q < 1

                % determine problem size
                size_A = op_A_bar( [], 0 );
                N_samples = size_A( 1 );

                % numer of iterations
                N_iterations = numel( options{ index_object }( index_options ).epsilon_n );

                % allocate memory for results
                theta_n = zeros( N_samples, N_iterations + 1 );
                theta_n( :, 1 ) = theta_recon_normed{ index_object }{ index_options };	% specify start vector ( minimizer of P_{(1, eta)} )

                info.info_inner = cell(1, N_iterations);
                info.N_prod_A = 0;
                info.N_prod_A_adj = 0;
                info.N_iter = 0;

                % iterate reweighted l1-minimization problems
                for index_iter = 1:N_iterations

                    % specify diagonal weighting matrix
                    weights_act = ( abs( theta_n( :, index_iter ) ) + options{ index_object }( index_options ).epsilon_n( index_iter ) ).^( 1 - options{ index_object }( index_options ).q );
                    LT_weighting_act = linear_transforms.weighting( weights_act );

                    % solve weighted l1-minimization problem using reformulation

                        % define normalized sensing operator
                        op_A_norm_n = @( x, mode ) op_A_norm( x, weights_act, mode );

                        % solve reweighted l1-minimization problem
                        [ temp, r, g, spgl1_info ] = spgl1( op_A_norm_n, u_M_normed, [], options{ index_object }( index_options ).rel_RMSE, [], spgl_opts );

                    % remove normalization
                    theta_n( :, index_iter + 1 ) = temp .* weights_act;

                    % statistics for computational costs
                    info.info_inner{ index_iter } = spgl1_info;
                    info.N_prod_A = info.N_prod_A + spgl1_info.nProdA;
                    info.N_prod_A_adj = info.N_prod_A_adj + spgl1_info.nProdAt;
                    info.N_iter = info.N_iter + spgl1_info.iter;

                end % for index_iter = 1:N_iterations

                theta_recon_normed{ index_object }{ index_options } = theta_n( :, end );

            end % if options{ index_object }( index_options ).q < 1

            %--------------------------------------------------------------
            % iv.) invert normalization and apply adjoint linear transform
            %--------------------------------------------------------------
            gamma_recon{ index_object }{ index_options } = operator_transform( linear_transforms{ index_object }, theta_recon_normed{ index_object }{ index_options }, 2 ) * u_M_norm;

            % display result
            figure( index_options );
            imagesc( squeeze( reshape( illustration.dB( gamma_recon{ index_object }{ index_options }, 20 ), operators_born( index_object ).discretization.spatial.grid_FOV.N_points_axis ) ), [ -70, 0 ] );
            colormap gray;

        end % for index_options = 1:numel( options{ index_object } )

        % avoid cell arrays for single options{ index_object }
        if isscalar( options{ index_object } )
            gamma_recon{ index_object } = gamma_recon{ index_object }{ 1 };
            theta_recon_normed{ index_object } = theta_recon_normed{ index_object }{ 1 };
            info{ index_object } = info{ index_object }{ 1 };
        end

    end % for index_object = 1:numel( operators_born )

	% avoid cell arrays for single operators_born
	if isscalar( operators_born )
        gamma_recon = gamma_recon{ 1 };
        theta_recon_normed = theta_recon_normed{ 1 };
        info = info{ 1 };
	end

    % stop time measurement (seconds)
% 	info{ index_object }.time_total = toc( time_start );

end % function [ gamma_recon, theta_recon_normed, info ] = lq_minimization( operators_born, u_M, options, varargin )

%                         
%                         % dimensions of sensing matrix
%                         info.size_A = op_phi_mlfma_normed_gpu([], 0);
% 
%                         for index_n = 1:options.l_q_N_iterations
%                 
%                             % modify sensing matrix
%                             weights_act = ( abs( theta_recon ) + options.epsilon_n(index_n) ).^(1 - options.q); 
%                             op_phi_current_transform_n = @(x, mode) op_phi_mlfma_normed_gpu_modified_n(x, weights_act, mode);
%                 
%                             % solve weighted l1-minimization problem using reformulation
%                             [theta_recon, y_m_res, gradient, info] = spgl1( op_phi_current_transform_n, y_m / y_m_energy, [], options.rel_mse, [], spgl_opts );
%                         
%                             % remove normalization
%                             theta_recon = theta_recon .* weights_act;
%                         end
%                     
%                         % correct scaling
%                         theta_recon = theta_recon * y_m_energy;
%                 
%                 
%                     %------------------------------------------------------
%                     % minimize lq-norm (according to Foucart's algorithm)
%                     %------------------------------------------------------
%                     if options.normalize_columns
% 
%                         % compute start vector as solution to l1-minimization problem
%                         theta_recon_normed = spgl1( op_A_mlfma_normed_gpu, y_m_normed, [], options.rel_RMSE, [], spgl_opts );
% 
%                         % minimize the lq-norm
%                         [theta_recon_normed, info] = lq_minimization_v2( @op_A_mlfma_normed_gpu_modified_n, y_m_normed, options.q, options.epsilon_n, theta_recon_normed, options.rel_RMSE, spgl_opts );
% 
%                         % correct scaling
%                         theta_recon_normed = theta_recon_normed * y_m_l2_norm;
%                     end