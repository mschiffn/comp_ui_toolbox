function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators_born, u_M, options )
%
% minimize the lq-norm to recover a sparse coefficient vector
%
% author: Martin F. Schiffner
% date: 2015-06-01
% modified: 2020-01-03
%

	% print status
	time_start = tic;
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( '\t %s: minimizing lq-norms...\n', str_date_time );

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

	% ensure nonempty options
	if isempty( options )
        options = cell( size( operators_born ) );
        for index_operator = 1:numel( operators_born )
            options{ index_operator } = regularization.options.common;
        end
    end

	% ensure cell array for options
	if ~iscell( options )
        options = { options };
    end

	% multiple operators_born / single options
	if ~isscalar( operators_born ) && isscalar( options )
        options = repmat( options, size( operators_born ) );
    end

	% single operators_born / multiple options
	if isscalar( operators_born ) && ~isscalar( options )
        operators_born = repmat( operators_born, size( options ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( operators_born, u_M, options );

	%----------------------------------------------------------------------
	% 2.) process scattering operators
	%----------------------------------------------------------------------
	% specify cell arrays
	gamma_recon = cell( size( operators_born ) );
	theta_recon_normed = cell( size( operators_born ) );
    u_M_res = cell( size( operators_born ) );
	info = cell( size( operators_born ) );

	% iterate scattering operators
	for index_operator = 1:numel( operators_born )

        %------------------------------------------------------------------
        % a) check arguments
        %------------------------------------------------------------------
        % ensure class regularization.options.common
        if ~isa( options{ index_operator }, 'regularization.options.common' )
            errorStruct.message = sprintf( 'options{ %d } must be regularization.options.common!', index_operator );
            errorStruct.identifier = 'adjoint:NoCommonOptions';
            error( errorStruct );
        end

        % ensure class discretizations.signal_matrix
        if ~isa( u_M{ index_operator }, 'discretizations.signal_matrix' )
            errorStruct.message = sprintf( 'u_M{ %d } must be discretizations.signal_matrix!', index_operator );
            errorStruct.identifier = 'adjoint:NoSignalMatrices';
            error( errorStruct );
        end

        %------------------------------------------------------------------
        % b) process options
        %------------------------------------------------------------------
        % name for temporary file
        str_filename = sprintf( 'data/%s/lq_minimization_temp.mat', operators_born( index_operator ).sequence.setup.str_name );

        % get name of directory
        [ str_name_dir, ~, ~ ] = fileparts( str_filename );

        % ensure existence of folder str_name_dir
        [ success, errorStruct.message, errorStruct.identifier ] = mkdir( str_name_dir );
        if ~success
            error( errorStruct );
        end

        % specify cell arrays
        gamma_recon{ index_operator } = cell( size( options{ index_operator } ) );
        theta_recon_normed{ index_operator } = cell( size( options{ index_operator } ) );
        u_M_res{ index_operator } = cell( size( options{ index_operator } ) );
        info{ index_operator } = cell( size( options{ index_operator } ) );

        % iterate options
        for index_options = 1:numel( options{ index_operator } )

            %--------------------------------------------------------------
            % i.) create configuration
            %--------------------------------------------------------------
            [ operator_born_act, LT_tgc, LT_act ] = get_configs( operators_born( index_operator ), options{ index_operator }( index_options ) );

            %--------------------------------------------------------------
            % ii.) create mixed voltage signals
            %--------------------------------------------------------------
            % extract relevant mixed voltage signals
% TODO: detect configuration changes first and avoid step if necessary
            u_M_act = u_M{ index_operator }( operator_born_act.indices_measurement_sel );

            % apply TGC and normalize mixed voltage signals
            u_M_act_vect = return_vector( u_M_act );
            u_M_act_vect_tgc = forward_transform( LT_tgc, u_M_act_vect );
            u_M_act_vect_tgc_norm = norm( u_M_act_vect_tgc );
            u_M_act_vect_tgc_normed = u_M_act_vect_tgc / u_M_act_vect_tgc_norm;

            %--------------------------------------------------------------
            % iii.) define anonymous function for sensing matrix
            %--------------------------------------------------------------
            op_A_bar = @( x, mode ) combined_quick( operator_born_act, mode, x, LT_act, LT_tgc );

            %--------------------------------------------------------------
            % iv.) execute algorithm
            %--------------------------------------------------------------
            if isa( options{ index_operator }( index_options ).algorithm, 'regularization.options.algorithm_spgl1' )

                %----------------------------------------------------------
                % a) SPGL1: l2- or l1-minimization (convex)
                %----------------------------------------------------------
                % create SPGL1 options structure
                spgl_opts = spgSetParms( 'verbosity', 1, 'optTol', 1e-4, 'iterations', options{ index_operator }( index_options ).algorithm.N_iterations_max );

                % alternative projection methods for l2-minimization
                if options{ index_operator }( index_options ).algorithm.q == 2
                    spgl_opts.project = @( x, weight, tau ) regularization.NormL2_project( x, weight, tau );
                    spgl_opts.primal_norm = @( x, weight ) regularization.NormL2_primal( x, weight );
                    spgl_opts.dual_norm = @( x, weight ) regularization.NormL2_dual( x, weight );
                end

                % specify start vector
                indicator_q = index_options > 1 && isequal( options{ index_operator }( index_options ).algorithm.q, options{ index_operator }( index_options - 1 ).algorithm.q );
                indicator_rel_RMSE = index_options > 1 && options{ index_operator }( index_options ).algorithm.rel_RMSE < options{ index_operator }( index_options - 1 ).algorithm.rel_RMSE;

                if ~indicator_q || ~indicator_rel_RMSE || isa( options{ index_operator }( index_options ).warm_start, 'regularization.options.warm_start_off' )

                    % A) inactive or impossible warm start
                    x_0 = [];
                    tau = [];

                elseif isa( options{ index_operator }( index_options ).warm_start, 'regularization.options.warm_start_previous' )

% TODO: might cause problems if normalization changes!
                    % B) use result for previous options for warm start
                    x_0 = theta_recon_normed{ index_operator }{ index_options - 1 };
                    tau = info{ index_operator }{ index_options - 1 }.tau;

                else

                    % C) invalid warm start settings
                    errorStruct.message = sprintf( 'Options{ %d }( %d ).warm_start is invalid for SPGL1!', index_operator, index_options );
                    errorStruct.identifier = 'lq_minimization:InvalidWarmStartSetting';
                    error( errorStruct );

                end % if ~indicator_q || ~indicator_rel_RMSE || isa( options{ index_operator }( index_options ).warm_start, 'regularization.options.warm_start_off' )

                % call SPGL1
                [ theta_recon_normed{ index_operator }{ index_options }, ...
                  u_M_act_vect_tgc_normed_res, ~, ...
                  info{ index_operator }{ index_options } ] ...
                = spgl1( op_A_bar, u_M_act_vect_tgc_normed, tau, options{ index_operator }( index_options ).algorithm.rel_RMSE, x_0, spgl_opts );

            elseif isa( options{ index_operator }( index_options ).algorithm, 'regularization.options.algorithm_omp' )

                %----------------------------------------------------------
                % b) OMP: l0-minimization (nonconvex)
                %----------------------------------------------------------
                % specify start vector
                if isa( options{ index_operator }( index_options ).warm_start, 'regularization.options.warm_start_off' )

                    % A) inactive or impossible warm start
                    x_0 = [];
                    atoms = [];

                else

                    % B) invalid warm start settings
                    errorStruct.message = sprintf( 'Options{ %d }( %d ).warm_start is invalid for OMP!', index_operator, index_options );
                    errorStruct.identifier = 'lq_minimization:InvalidWarmStartSetting';
                    error( errorStruct );

                end % if isa( options{ index_operator }( index_options ).warm_start, 'regularization.options.warm_start_off' )

                % call OMP
% TODO: start vector x_0, atoms
                [ theta_recon_normed{ index_operator }{ index_options }, ...
                  u_M_act_vect_tgc_normed_res, ...
                  info{ index_operator }{ index_options } ] ...
                = regularization.omp( op_A_bar, u_M_act_vect_tgc_normed, options{ index_operator }( index_options ).algorithm );

            elseif isa( options{ index_operator }( index_options ).algorithm, 'regularization.options.algorithm_cosamp' )

                %----------------------------------------------------------
                % c) CoSaMP: l0-minimization (nonconvex)
                %----------------------------------------------------------
                [ theta_recon_normed{ index_operator }{ index_options }, ...
                  u_M_act_vect_tgc_normed_res, ...
                  info{ index_operator }{ index_options } ] ...
                = regularization.cosamp( op_A_bar, u_M_act_vect_tgc_normed, options{ index_operator }( index_options ).algorithm );

            else

                %----------------------------------------------------------
                % d) unknown algorithm
                %----------------------------------------------------------
                errorStruct.message = sprintf( 'Class of options{ %d }( %d ).algorithm is unknown!', index_operator, index_options );
                errorStruct.identifier = 'lq_minimization:UnknownOptionsClass';
                error( errorStruct );

            end % if isa( options{ index_operator }( index_options ).algorithm, 'regularization.options.algorithm_spgl1' )

            %--------------------------------------------------------------
            % v.) format residual mixed RF voltage signals
            %--------------------------------------------------------------
            u_M_res{ index_operator }{ index_options } = format_voltages( operator_born_act, u_M_act_vect_tgc_normed_res * u_M_act_vect_tgc_norm );

            %--------------------------------------------------------------
            % vi.) optional reweighting (Foucart's algorithm, nonconvex)
            %--------------------------------------------------------------
            if isa( options{ index_operator }( index_options ).reweighting, 'regularization.options.reweighting_sequence' )

                %----------------------------------------------------------
                % a) sequential reweighting
                %----------------------------------------------------------
                % exponent, sequence, and number of iterations
                exponent_act = options{ index_operator }( index_options ).reweighting.q;
                epsilon_n_act = options{ index_operator }( index_options ).reweighting.epsilon_n;
                N_iterations = numel( epsilon_n_act );

                % allocate memory for results and specify start vector ( minimizer of P_{(1, eta)} )
                theta_n = zeros( LT_act.N_coefficients, N_iterations + 1 );
                theta_n( :, 1 ) = theta_recon_normed{ index_operator }{ index_options };
% TODO: residual voltages
                % statistics
                info{ index_operator }{ index_transform }{ index_config }{ index_options }.info_reweighting = cell(1, N_iterations);

                % iterate reweighted problems
                for index_iter = 1:N_iterations

                    % specify diagonal weighting matrix
                    weights_act = ( abs( theta_n( :, index_iter ) ) + epsilon_n_act( index_iter ) ).^( 1 - exponent_act );
                    LT_act_n = linear_transforms.composition( linear_transforms.weighting( weights_act ), LT_act );

                    % define anonymous function for reweighted sensing matrix
                    op_A_bar_n = @( x, mode ) combined_quick( operator_born_act, mode, x, LT_act_n, LT_tgc );

                    % solve reweighted problem
                    if isa( options{ index_operator }( index_options ).algorithm, 'regularization.options.algorithm_spgl1' )

                        [ temp, ~, ~, info_reweighting ] = spgl1( op_A_bar_n, u_M_vect_tgc_normed, [], options{ index_operator }( index_options ).algorithm.rel_RMSE, [], spgl_opts );

                    elseif isa( options{ index_operator }( index_options ).algorithm, 'regularization.options.algorithm_omp' )

                        [ temp, ~, info_reweighting ] = regularization.omp( op_A_bar_n, u_M_vect_tgc_normed, options{ index_operator }( index_options ).algorithm );

                    else

                        errorStruct.message = sprintf( 'Options{ %d }{ %d }{ %d }( %d ).algorithm is invalid for reweighting!', index_operator, index_transform, index_config, index_options );
                        errorStruct.identifier = 'lq_minimization:InvalidAlgorithmSetting';
                        error( errorStruct );

                    end

                    % remove reweighting
                    theta_n( :, index_iter + 1 ) = temp .* weights_act;

                    % statistics for computational costs
                    info{ index_operator }{ index_transform }{ index_config }{ index_options }.info_reweighting{ index_iter } = info_reweighting;

                end % for index_iter = 1:N_iterations

                % store results for entire reweighting sequence
                theta_recon_normed{ index_operator }{ index_options } = theta_n;

            end % if isa( options{ index_operator }( index_options ).reweighting, 'regularization.options.reweighting_sequence' )

            %--------------------------------------------------------------
            % vii.) invert normalization and apply adjoint linear transform
            %--------------------------------------------------------------
            gamma_recon{ index_operator }{ index_options } ...
            = operator_transform( LT_act, theta_recon_normed{ index_operator }{ index_options }, 2 ) * u_M_act_vect_tgc_norm;

            %--------------------------------------------------------------
            % save results to temporary file
            %--------------------------------------------------------------
            save( str_filename, 'gamma_recon', 'theta_recon_normed', 'u_M_res', 'info' );

            % display result
            figure( index_options );
            temp_1 = squeeze( reshape( theta_recon_normed{ index_operator }{ index_options }( :, end ), operators_born( index_operator ).sequence.setup.FOV.shape.grid.N_points_axis ) );
            temp_2 = squeeze( reshape( gamma_recon{ index_operator }{ index_options }( :, end ), operators_born( index_operator ).sequence.setup.FOV.shape.grid.N_points_axis ) );
            if ismatrix( temp_1 )
                subplot( 1, 2, 1 );
                imagesc( illustration.dB( temp_1, 20 )', [ -60, 0 ] );
                subplot( 1, 2, 2 );
                imagesc( illustration.dB( temp_2, 20 )', [ -60, 0 ] );
            else
                subplot( 1, 2, 1 );
                imagesc( illustration.dB( squeeze( temp_1( :, 5, : ) ), 20 )', [ -60, 0 ] );
                subplot( 1, 2, 2 );
                imagesc( illustration.dB( squeeze( temp_2( :, 5, : ) ), 20 )', [ -60, 0 ] );
            end
            colormap gray;

        end % for index_options = 1:numel( options{ index_operator } )

        %------------------------------------------------------------------
        % c) create images and signal matrices
        %------------------------------------------------------------------
        gamma_recon{ index_operator } ...
        = discretizations.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, gamma_recon{ index_operator } );
        theta_recon_normed{ index_operator } ...
        = discretizations.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, theta_recon_normed{ index_operator } );

        % avoid cell arrays for single options{ index_operator }
        if isscalar( options{ index_operator } )
            u_M_res{ index_operator } = u_M_res{ index_operator }{ 1 };
            info{ index_operator } = info{ index_operator }{ 1 };
        end

    end % for index_operator = 1:numel( operators_born )

	% avoid cell arrays for single operators_born
	if isscalar( operators_born )
        gamma_recon = gamma_recon{ 1 };
        theta_recon_normed = theta_recon_normed{ 1 };
        u_M_res = u_M_res{ 1 };
        info = info{ 1 };
    end

	% infer and print elapsed time
	time_elapsed = toc( time_start );
	fprintf( 'done! (%f s)\n', time_elapsed );

end % function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators_born, u_M, options )
