function [ gamma_recon, theta_recon_normed, info ] = lq_minimization( operators_born, u_M, options, varargin )
%
% minimize the lq-norm using SPGL1 to recover a sparse coefficient vector
%
% author: Martin F. Schiffner
% date: 2015-06-01
% modified: 2019-08-10
%

	% print status
	time_start = tic;
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( '\t %s: minimizing lq-norms using SPGL1... ', str_date_time );

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
	if ~iscell( u_M ) || all( cellfun( @( x ) ~iscell( x ), u_M ) )
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
                % case 1.a)
                linear_transforms = num2cell( linear_transforms );
            end

            % 
            if all( cellfun( @( x ) ~iscell( x ), linear_transforms ) )
                % cases 1.b) or 2.a)
                if ~isscalar( linear_transforms{ 1 } ) || ( numel( linear_transforms ) > 1 && N_config == 1 )
                    % case 2.a)
                    for index_transform = 1:numel( linear_transforms )
                        linear_transforms{ index_transform } = num2cell( linear_transforms{ index_transform } );
                    end
                else
                    % case 1.b)
                    linear_transforms = { linear_transforms };
                end
            end

            if all( cellfun( @( x ) all( cellfun( @( y ) ~iscell( y ), x ) ), linear_transforms ) )
                % cases 2.b) or 3.a)
                if ~isscalar( linear_transforms{ 1 }{ 1 } ) || ( numel( linear_transforms{1} ) > 1 && N_config == 1 )
                    % case 3.a)
                    for index_operator = 1:numel( linear_transforms )
                        for index_transform = 1:numel( linear_transforms{ index_operator } )
                            linear_transforms{ index_operator }{ index_transform } = num2cell( linear_transforms{ index_operator }{ index_transform } );
                        end
                    end
                else
                    % case 2.b)
                    linear_transforms = { linear_transforms };
                end
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
	% 2.) process scattering operators
	%----------------------------------------------------------------------
	% specify cell arrays
	gamma_recon = cell( size( operators_born ) );
	theta_recon_normed = cell( size( operators_born ) );
	info = cell( size( operators_born ) );

	% iterate scattering operators
	for index_operator = 1:numel( operators_born )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure cell array for linear_transforms{ index_operator }
        if all( cellfun( @( x ) ~iscell( x ), linear_transforms{ index_operator } ) )
            linear_transforms{ index_operator } = linear_transforms( index_operator );
        end

        %------------------------------------------------------------------
        % b) process linear transforms
        %------------------------------------------------------------------
        % specify cell arrays
        gamma_recon{ index_operator } = cell( size( linear_transforms{ index_operator } ) );
        theta_recon_normed{ index_operator } = cell( size( linear_transforms{ index_operator } ) );
        info{ index_operator } = cell( size( linear_transforms{ index_operator } ) );

        % iterate linear transforms
        for index_transform = 1:numel( linear_transforms{ index_operator } )

            %--------------------------------------------------------------
            % i.) check arguments
            %--------------------------------------------------------------
            % set momentary scattering operator options
            operators_born_config = set_properties_momentary( operators_born( index_operator ), varargin{ 2:end } );

            % multiple operators_born_config / single linear_transforms{ index_operator }{ index_transform }
            if ~isscalar( operators_born_config ) && isscalar( linear_transforms{ index_operator }{ index_transform } )
                linear_transforms{ index_operator }{ index_transform } = repmat( linear_transforms{ index_operator }{ index_transform }, size( operators_born_config ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born_config, linear_transforms{ index_operator }{ index_transform } );

            %--------------------------------------------------------------
            % ii.) process configurations
            %--------------------------------------------------------------
            % specify cell arrays
            gamma_recon{ index_operator }{ index_transform } = cell( size( operators_born_config ) );
            theta_recon_normed{ index_operator }{ index_transform } = cell( size( operators_born_config ) );
            info{ index_operator }{ index_transform } = cell( size( operators_born_config ) );

            % iterate configurations
            for index_config = 1:numel( operators_born_config )

                %----------------------------------------------------------
                % A) check mixed voltage signals, options, and linear transform
                %--------------------------------------------------------------
% TODO: u_M{ index_operator } may be cell array containing signals or signal_matrices
                % ensure class discretizations.signal_matrix
                if ~isa( u_M{ index_operator }{ index_config }, 'discretizations.signal_matrix' )
                    errorStruct.message = sprintf( 'u_M{ %d } must be discretizations.signal_matrix!', index_operator );
                    errorStruct.identifier = 'lq_minimization:NoSignalMatrices';
                    error( errorStruct );
                end

                % ensure class optimization.options_lq_minimization
                if ~isa( options{ index_operator }{ index_transform }{ index_config }, 'optimization.options_lq_minimization' )
                    errorStruct.message = sprintf( 'options{ %d }{ %d }{ %d } must be optimization.options_lq_minimization!', index_operator, index_transform, index_config );
                    errorStruct.identifier = 'lq_minimization:NoOptions';
                    error( errorStruct );
                end

                % ensure class linear_transforms.linear_transform (scalar)
%                 if ~( isa( linear_transforms{ index_operator }, 'linear_transforms.linear_transform' ) && isscalar( linear_transforms{ index_operator } ) )
%                     errorStruct.message = sprintf( 'linear_transforms{ %d } must be a single linear_transforms.linear_transform!', index_operator );
%                     errorStruct.identifier = 'lq_minimization:NoSingleLinearTransform';
%                     error( errorStruct );
%                 end

                %----------------------------------------------------------
                % B) normalize mixed voltage signals
                %----------------------------------------------------------
                u_M_vect = return_vector( u_M{ index_operator }{ index_config } );
                u_M_vect_norm = norm( u_M_vect );
                u_M_vect_normed = u_M_vect / u_M_vect_norm;

                %----------------------------------------------------------
                % C) process optimization options
                %----------------------------------------------------------
                % specify cell arrays
                gamma_recon{ index_operator }{ index_transform }{ index_config } = cell( size( options{ index_operator }{ index_transform }{ index_config } ) );
                theta_recon_normed{ index_operator }{ index_transform }{ index_config } = cell( size( options{ index_operator }{ index_transform }{ index_config } ) );
                info{ index_operator }{ index_transform }{ index_config } = cell( size( options{ index_operator }{ index_transform }{ index_config } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator }{ index_transform }{ index_config } )

                    %------------------------------------------------------
                    % i.) options and start vector
                    %------------------------------------------------------
                    % create SPGL1 options structure
                    spgl_opts = spgSetParms( 'verbosity', 1, 'optTol', 1e-4, 'iterations', options{ index_operator }{ index_transform }{ index_config }( index_options ).N_iterations_max );

                    % replace projection methods for l2-minimization
                    if options{ index_operator }{ index_transform }{ index_config }( index_options ).q == 2
                        spgl_opts.project = @( x, weight, tau ) optimization.NormL2_project( x, weight, tau );
                        spgl_opts.primal_norm = @( x, weight ) optimization.NormL2_primal( x, weight );
                        spgl_opts.dual_norm = @( x, weight ) optimization.NormL2_dual( x, weight );
                    end

                    % check normalization status
                    if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization, 'optimization.options_normalization_off' )

                        %--------------------------------------------------
                        % a) inactive spatial anti-aliasing filter
                        %--------------------------------------------------
                        % copy linear transform
                        LT_act = linear_transforms{ index_operator }{ index_transform }{ index_config };

                    elseif isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization, 'optimization.options_normalization_threshold' )

                        %--------------------------------------------------
                        % b) apply threshold to inverse weighting matrix
                        %--------------------------------------------------
                        try
% TODO: implement threshold for composition
                            LT_act = threshold( linear_transforms{ index_operator }{ index_transform }{ index_config }, options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization.threshold );
                        catch
                            errorStruct.message = sprintf( 'Could not apply threshold to linear_transforms{ %d }{ %d }{ %d }!', index_operator, index_transform, index_config );
                            errorStruct.identifier = 'lq_minimization:UnknownOptionsClass';
                            error( errorStruct );
                        end

                    else

                        %--------------------------------------------------
                        % c) unknown spatial anti-aliasing filter
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of options{ %d }{ %d }{ %d }( %d ).normalization is unknown!', index_operator, index_transform, index_config, index_options );
                        errorStruct.identifier = 'lq_minimization:UnknownOptionsClass';
                        error( errorStruct );

                    end % if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization, 'optimization.options_normalization_off' )

                    % define anonymous function for sensing matrix
                    op_A_bar = @( x, mode ) combined_quick( operators_born_config( index_config ), x, mode, LT_act );

                    % create start vector
% TODO: not valid if normalization changes!
                    if index_options > 1 && isequal( options{ index_operator }{ index_transform }{ index_config }( index_options ).q, options{ index_operator }{ index_transform }{ index_config }( index_options - 1 ).q ) && options{ index_operator }{ index_transform }{ index_config }( index_options ).rel_RMSE < options{ index_operator }{ index_transform }{ index_config }( index_options - 1 ).rel_RMSE
                        x_0 = theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options - 1 };
                        tau = info{ index_operator }{ index_transform }{ index_config }{ index_options - 1 }.tau;
                    else
                        x_0 = [];
                        tau = [];
                    end

                    %------------------------------------------------------
                    % ii.) l2- or l1-minimization (convex)
                    %------------------------------------------------------
                    % call SPGL1 with sensing matrix
                    [ theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }, u_M_res, ~, info{ index_operator }{ index_transform }{ index_config }{ index_options } ] ...
                        = spgl1( op_A_bar, u_M_vect_normed, tau, options{ index_operator }{ index_transform }{ index_config }( index_options ).rel_RMSE, x_0, spgl_opts );

                    %------------------------------------------------------
                    % iii.) optional lq-minimization ( 0 <= q < 1, nonconvex, Foucart's algorithm )
                    %------------------------------------------------------
                    if options{ index_operator }{ index_transform }{ index_config }( index_options ).q < 1

                        % determine problem size
                        size_A = op_A_bar( [], 0 );
                        N_samples = size_A( 1 );

                        % numer of iterations
                        N_iterations = numel( options{ index_operator }{ index_transform }{ index_config }( index_options ).epsilon_n );

                        % allocate memory for results
                        theta_n = zeros( N_samples, N_iterations + 1 );
                        theta_n( :, 1 ) = theta_recon_normed{ index_operator }{ index_options };	% specify start vector ( minimizer of P_{(1, eta)} )

                        info.info_inner = cell(1, N_iterations);
                        info.N_prod_A = 0;
                        info.N_prod_A_adj = 0;
                        info.N_iter = 0;

                        % iterate reweighted l1-minimization problems
                        for index_iter = 1:N_iterations

                            % specify diagonal weighting matrix
                            weights_act = ( abs( theta_n( :, index_iter ) ) + options{ index_operator }{ index_transform }{ index_config }( index_options ).epsilon_n( index_iter ) ).^( 1 - options{ index_operator }{ index_transform }{ index_config }( index_options ).q );
                            LT_weighting_act = linear_transforms.weighting( weights_act );

                            % solve weighted l1-minimization problem using reformulation

                                % define normalized sensing operator
                                op_A_norm_n = @( x, mode ) op_A_norm( x, weights_act, mode );

                                % solve reweighted l1-minimization problem
                                [ temp, r, g, spgl1_info ] = spgl1( op_A_norm_n, u_M_normed, [], options{ index_operator }{ index_transform }{ index_config }( index_options ).rel_RMSE, [], spgl_opts );

                            % remove normalization
                            theta_n( :, index_iter + 1 ) = temp .* weights_act;

                            % statistics for computational costs
                            info.info_inner{ index_iter } = spgl1_info;
                            info.N_prod_A = info.N_prod_A + spgl1_info.nProdA;
                            info.N_prod_A_adj = info.N_prod_A_adj + spgl1_info.nProdAt;
                            info.N_iter = info.N_iter + spgl1_info.iter;

                        end % for index_iter = 1:N_iterations

                        theta_recon_normed{ index_operator }{ index_options } = theta_n( :, end );

                    end % if options{ index_operator }{ index_transform }{ index_config }( index_options ).q < 1

                    %------------------------------------------------------
                    % iv.) invert normalization and apply adjoint linear transform
                    %------------------------------------------------------
                    gamma_recon{ index_operator }{ index_transform }{ index_config }{ index_options } ...
                        = operator_transform( LT_act, theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }, 2 ) * u_M_vect_norm;

                    % display result
                    figure( index_options );
                    imagesc( squeeze( reshape( illustration.dB( gamma_recon{ index_operator }{ index_transform }{ index_config }{ index_options }, 20 ), operators_born( index_operator ).discretization.spatial.grid_FOV.N_points_axis ) )', [ -60, 0 ] );
                    colormap gray;

                end % for index_options = 1:numel( options{ index_operator }{ index_transform }{ index_config } )

                % avoid cell arrays for single options{ index_operator }{ index_transform }{ index_config }
                if isscalar( options{ index_operator }{ index_transform }{ index_config } )
                    gamma_recon{ index_operator }{ index_transform }{ index_config } = gamma_recon{ index_operator }{ index_transform }{ index_config }{ 1 };
                    theta_recon_normed{ index_operator }{ index_transform }{ index_config } = theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ 1 };
                    info{ index_operator }{ index_transform }{ index_config } = info{ index_operator }{ index_transform }{ index_config }{ 1 };
                end

            end % for index_config = 1:numel( operators_born_config )

            % avoid cell arrays for single operators_born_config
            if isscalar( operators_born_config )
                gamma_recon{ index_operator }{ index_transform } = gamma_recon{ index_operator }{ index_transform }{ 1 };
                theta_recon_normed{ index_operator }{ index_transform } = theta_recon_normed{ index_operator }{ index_transform }{ 1 };
                info{ index_operator }{ index_transform } = info{ index_operator }{ index_transform }{ 1 };
            end

        end % for index_transform = 1:numel( linear_transforms{ index_operator } )

        % avoid cell arrays for single linear_transforms{ index_operator }
        if isscalar( linear_transforms{ index_operator } )
            gamma_recon{ index_operator } = gamma_recon{ index_operator }{ 1 };
            theta_recon_normed{ index_operator } = theta_recon_normed{ index_operator }{ 1 };
            info{ index_operator } = info{ index_operator }{ 1 };
        end

    end % for index_operator = 1:numel( operators_born )

	% avoid cell arrays for single operators_born
	if isscalar( operators_born )
        gamma_recon = gamma_recon{ 1 };
        theta_recon_normed = theta_recon_normed{ 1 };
        info = info{ 1 };
    end

	% infer and print elapsed time
	time_elapsed = toc( time_start );
	fprintf( 'done! (%f s)\n', time_elapsed );

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