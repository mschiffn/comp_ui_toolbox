function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators_born, u_M, options, varargin )
%
% minimize the lq-norm to recover a sparse coefficient vector
%
% author: Martin F. Schiffner
% date: 2015-06-01
% modified: 2019-09-18
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
	if ~iscell( u_M ) || all( cellfun( @( x ) ~iscell( x ), u_M ) )
        u_M = { u_M };
	end

	% ensure cell array for options
	if ~iscell( options )
        options = { options };
	end

	% ensure nonempty LTs
	if nargin >= 4 && ~isempty( varargin{ 1 } )
        LTs = varargin{ 1 };
    else
        % empty linear_transform is identity
        LTs = cell( size( operators_born ) );
    end

	% ensure cell array for LTs
	if ~iscell( LTs ) || all( cellfun( @( x ) ~iscell( x ), LTs ) )
        LTs = { LTs };
    end

	% multiple operators_born / single u_M
	if ~isscalar( operators_born ) && isscalar( u_M )
        u_M = repmat( u_M, size( operators_born ) );
    end

	% multiple operators_born / single options
	if ~isscalar( operators_born ) && isscalar( options )
        options = repmat( options, size( operators_born ) );
    end

	% multiple operators_born / single LTs
	if ~isscalar( operators_born ) && isscalar( LTs )
        LTs = repmat( LTs, size( operators_born ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( operators_born, u_M, options, LTs );

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
        % ensure cell array for LTs{ index_operator }
        if ~iscell( LTs{ index_operator } )
            LTs{ index_operator } = LTs( index_operator );
        end

        %------------------------------------------------------------------
        % b) process linear transforms
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
        gamma_recon{ index_operator } = cell( size( LTs{ index_operator } ) );
        theta_recon_normed{ index_operator } = cell( size( LTs{ index_operator } ) );
        u_M_res{ index_operator } = cell( size( LTs{ index_operator } ) );
        info{ index_operator } = cell( size( LTs{ index_operator } ) );

        % iterate linear transforms
        for index_transform = 1:numel( LTs{ index_operator } )

            %--------------------------------------------------------------
            % i.) check arguments
            %--------------------------------------------------------------
            % set momentary scattering operator options
            operators_born_config = set_properties_momentary( operators_born( index_operator ), varargin{ 2:end } );

            % ensure class linear_transforms.linear_transform
            if ~isa( LTs{ index_operator }{ index_transform }, 'linear_transforms.linear_transform' )
                errorStruct.message = sprintf( 'LTs{ %d }{ %d } must be linear_transforms.linear_transform!', index_operator, index_transform );
                errorStruct.identifier = 'lq_minimization:NoLinearTransforms';
                error( errorStruct );
            end

            % multiple operators_born_config / single LTs{ index_operator }{ index_transform }
            if ~isscalar( operators_born_config ) && isscalar( LTs{ index_operator }{ index_transform } )
                LTs{ index_operator }{ index_transform } = repmat( LTs{ index_operator }{ index_transform }, size( operators_born_config ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators_born_config, LTs{ index_operator }{ index_transform } );

            %--------------------------------------------------------------
            % ii.) process configurations
            %--------------------------------------------------------------
            % numbers of transform coefficients
            N_coefficients = reshape( [ LTs{ index_operator }{ index_transform }.N_coefficients ], size( LTs{ index_operator }{ index_transform } ) );

            % ensure identical numbers of transform coefficients
            if any( N_coefficients( : ) ~= N_coefficients( 1 ) )
                errorStruct.message = sprintf( 'LTs{ %d }{ %d } must have identical numbers of transform coefficients!', index_operator, index_transform );
                errorStruct.identifier = 'adjoint:InvalidNumbersOfCoefficients';
                error( errorStruct );
            end

            % specify cell arrays
            gamma_recon{ index_operator }{ index_transform } = cell( size( operators_born_config ) );
            theta_recon_normed{ index_operator }{ index_transform } = cell( size( operators_born_config ) );
            u_M_res{ index_operator }{ index_transform } = cell( size( operators_born_config ) );
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

                % ensure class optimization.options
                if ~isa( options{ index_operator }{ index_transform }{ index_config }, 'optimization.options' )
                    errorStruct.message = sprintf( 'options{ %d }{ %d }{ %d } must be optimization.options!', index_operator, index_transform, index_config );
                    errorStruct.identifier = 'lq_minimization:NoOptions';
                    error( errorStruct );
                end

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
                u_M_res{ index_operator }{ index_transform }{ index_config } = cell( size( options{ index_operator }{ index_transform }{ index_config } ) );
                info{ index_operator }{ index_transform }{ index_config } = cell( size( options{ index_operator }{ index_transform }{ index_config } ) );

                % iterate optimization options
                for index_options = 1:numel( options{ index_operator }{ index_transform }{ index_config } )

                    %------------------------------------------------------
                    % i.) apply normalization settings
                    %------------------------------------------------------
                    if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization, 'optimization.options.normalization_off' )

                        %--------------------------------------------------
                        % a) complete normalization w/o threshold
                        %--------------------------------------------------
                        % copy linear transform
                        LT_act = LTs{ index_operator }{ index_transform }( index_config );

                    elseif isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization, 'optimization.options.normalization_threshold' )

                        %--------------------------------------------------
                        % b) apply threshold to inverse weighting matrix
                        %--------------------------------------------------
                        try
                            LT_act = threshold( LTs{ index_operator }{ index_transform }( index_config ), options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization.threshold );
                        catch
                            errorStruct.message = sprintf( 'Could not apply threshold to LTs{ %d }{ %d }( %d )!', index_operator, index_transform, index_config );
                            errorStruct.identifier = 'lq_minimization:ThresholdError';
                            error( errorStruct );
                        end

                    else

                        %--------------------------------------------------
                        % c) unknown normalization settings
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of options{ %d }{ %d }{ %d }( %d ).normalization is unknown!', index_operator, index_transform, index_config, index_options );
                        errorStruct.identifier = 'lq_minimization:UnknownOptionsClass';
                        error( errorStruct );

                    end % if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).normalization, 'optimization.options.normalization_off' )

                    %------------------------------------------------------
                    % ii.) define anonymous function for sensing matrix
                    %------------------------------------------------------
                    op_A_bar = @( x, mode ) combined_quick( operators_born_config( index_config ), x, mode, LT_act );

                    %------------------------------------------------------
                    % iii.) execute algorithm
                    %------------------------------------------------------
                    if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm, 'optimization.options.algorithm_spgl1' )

                        %--------------------------------------------------
                        % a) SPGL1: l2- or l1-minimization (convex)
                        %--------------------------------------------------
                        % create SPGL1 options structure
                        spgl_opts = spgSetParms( 'verbosity', 1, 'optTol', 1e-4, 'iterations', options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm.N_iterations_max );

                        % alternative projection methods for l2-minimization
                        if options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm.q == 2
                            spgl_opts.project = @( x, weight, tau ) optimization.NormL2_project( x, weight, tau );
                            spgl_opts.primal_norm = @( x, weight ) optimization.NormL2_primal( x, weight );
                            spgl_opts.dual_norm = @( x, weight ) optimization.NormL2_dual( x, weight );
                        end

                        % specify start vector
                        indicator_q = index_options > 1 && isequal( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm.q, options{ index_operator }{ index_transform }{ index_config }( index_options - 1 ).algorithm.q );
                        indicator_rel_RMSE = index_options > 1 && options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm.rel_RMSE < options{ index_operator }{ index_transform }{ index_config }( index_options - 1 ).algorithm.rel_RMSE;

                        if ~indicator_q || ~indicator_rel_RMSE || isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).warm_start, 'optimization.options.warm_start_off' )

                            % A) inactive or impossible warm start
                            x_0 = [];
                            tau = [];

                        elseif isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).warm_start, 'optimization.options.warm_start_previous' )

% TODO: might cause problems if normalization changes!
                            % B) use result for previous options for warm start
                            x_0 = theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options - 1 };
                            tau = info{ index_operator }{ index_transform }{ index_config }{ index_options - 1 }.tau;

                        else

                            % C) invalid warm start settings
                            errorStruct.message = sprintf( 'Options{ %d }{ %d }{ %d }( %d ).warm_start is invalid for SPGL1!', index_operator, index_transform, index_config, index_options );
                            errorStruct.identifier = 'lq_minimization:InvalidWarmStartSetting';
                            error( errorStruct );

                        end % if ~indicator_q || ~indicator_rel_RMSE || isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).warm_start, 'optimization.options.warm_start_off' )

                        % call SPGL1
                        [ theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }, ...
                        u_M_vect_normed_res, ~, ...
                        info{ index_operator }{ index_transform }{ index_config }{ index_options } ] ...
                        = spgl1( op_A_bar, u_M_vect_normed, tau, options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm.rel_RMSE, x_0, spgl_opts );

                    elseif isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm, 'optimization.options.algorithm_omp' )

                        %--------------------------------------------------
                        % b) OMP: l0-minimization (nonconvex)
                        %--------------------------------------------------
                        % specify start vector
                        if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).warm_start, 'optimization.options.warm_start_off' )

                            % A) inactive or impossible warm start
                            x_0 = [];
                            atoms = [];

                        else

                            % B) invalid warm start settings
                            errorStruct.message = sprintf( 'Options{ %d }{ %d }{ %d }( %d ).warm_start is invalid for OMP!', index_operator, index_transform, index_config, index_options );
                            errorStruct.identifier = 'lq_minimization:InvalidWarmStartSetting';
                            error( errorStruct );

                        end % if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).warm_start, 'optimization.options.warm_start_off' )

                        % call OMP
% TODO: start vector x_0, atoms
                        [ theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }, ...
                          u_M_vect_normed_res, ...
                          info{ index_operator }{ index_transform }{ index_config }{ index_options } ] ...
                        = optimization.omp( op_A_bar, u_M_vect_normed, options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm );

                    elseif isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm, 'optimization.options.algorithm_cosamp' )

                        %--------------------------------------------------
                        % c) CoSaMP: l0-minimization (nonconvex)
                        %--------------------------------------------------
                        [ theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }, ...
                          u_M_vect_normed_res, ...
                          info{ index_operator }{ index_transform }{ index_config }{ index_options } ] ...
                        = optimization.cosamp( op_A_bar, u_M_vect_normed, options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm );

                    else

                        %--------------------------------------------------
                        % d) unknown algorithm
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of options{ %d }{ %d }{ %d }( %d ).algorithm is unknown!', index_operator, index_transform, index_config, index_options );
                        errorStruct.identifier = 'lq_minimization:UnknownOptionsClass';
                        error( errorStruct );

                    end % if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm, 'optimization.options.algorithm_spgl1' )

                    %------------------------------------------------------
                    % iv.) reshape residual mixed RF voltage signals
                    %------------------------------------------------------
                    axes_f = reshape( [ u_M{ index_operator }{ index_config }.axis ], size( u_M{ index_operator }{ index_config } ) );
                    [ N_samples_f, N_signals ] = cellfun( @( x ) size( x ), reshape( { u_M{ index_operator }{ index_config }.samples }, size( u_M{ index_operator }{ index_config } ) ) );
                    u_M_vect_res = mat2cell( u_M_vect_normed_res * u_M_vect_norm, N_samples_f .* N_signals, 1 );
                    for index_matrix = 1:numel( u_M{ index_operator }{ index_config } )
                        u_M_vect_res{ index_matrix } = reshape( u_M_vect_res{ index_matrix }, [ N_samples_f( index_matrix ), N_signals( index_matrix ) ] );
                    end
                    u_M_res{ index_operator }{ index_transform }{ index_config }{ index_options } = discretizations.signal_matrix( axes_f, u_M_vect_res );

                    %------------------------------------------------------
                    % v.) optional reweighting (Foucart's algorithm, nonconvex)
                    %------------------------------------------------------
                    if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).reweighting, 'optimization.options.reweighting_sequence' )

                        %--------------------------------------------------
                        % a) sequential reweighting
                        %--------------------------------------------------
                        % exponent, sequence, and number of iterations
                        exponent_act = options{ index_operator }{ index_transform }{ index_config }( index_options ).reweighting.q;
                        epsilon_n_act = options{ index_operator }{ index_transform }{ index_config }( index_options ).reweighting.epsilon_n;
                        N_iterations = numel( epsilon_n_act );

                        % allocate memory for results and specify start vector ( minimizer of P_{(1, eta)} )
                        theta_n = zeros( LT_act.N_coefficients, N_iterations + 1 );
                        theta_n( :, 1 ) = theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options };
% TODO: residual voltages
                        % statistics
                        info{ index_operator }{ index_transform }{ index_config }{ index_options }.info_reweighting = cell(1, N_iterations);

                        % iterate reweighted problems
                        for index_iter = 1:N_iterations

                            % specify diagonal weighting matrix
                            weights_act = ( abs( theta_n( :, index_iter ) ) + epsilon_n_act( index_iter ) ).^( 1 - exponent_act );
                            LT_act_n = linear_transforms.composition( linear_transforms.weighting( weights_act ), LT_act );

                            % define anonymous function for reweighted sensing matrix
                            op_A_bar_n = @( x, mode ) combined_quick( operators_born_config( index_config ), x, mode, LT_act_n );

                            % solve reweighted problem
                            if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm, 'optimization.options.algorithm_spgl1' )

                                [ temp, ~, ~, info_reweighting ] = spgl1( op_A_bar_n, u_M_vect_normed, [], options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm.rel_RMSE, [], spgl_opts );

                            elseif isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm, 'optimization.options.algorithm_omp' )

                                [ temp, ~, info_reweighting ] = optimization.omp( op_A_bar_n, u_M_vect_normed, options{ index_operator }{ index_transform }{ index_config }( index_options ).algorithm );

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
                        theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options } = theta_n;

                    end % if isa( options{ index_operator }{ index_transform }{ index_config }( index_options ).reweighting, 'optimization.options.reweighting_sequence' )

                    %------------------------------------------------------
                    % vi.) invert normalization and apply adjoint linear transform
                    %------------------------------------------------------
                    gamma_recon{ index_operator }{ index_transform }{ index_config }{ index_options } ...
                    = operator_transform( LT_act, theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }, 2 ) * u_M_vect_norm;

                    %------------------------------------------------------
                    % save results to temporary file
                    %------------------------------------------------------
                    save( str_filename, 'gamma_recon', 'theta_recon_normed', 'u_M_res', 'info' );

                    % display result
                    figure( index_options );
                    temp_1 = squeeze( reshape( theta_recon_normed{ index_operator }{ index_transform }{ index_config }{ index_options }( :, end ), operators_born( index_operator ).sequence.setup.FOV.shape.grid.N_points_axis ) );
                    temp_2 = squeeze( reshape( gamma_recon{ index_operator }{ index_transform }{ index_config }{ index_options }( :, end ), operators_born( index_operator ).sequence.setup.FOV.shape.grid.N_points_axis ) );
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

                end % for index_options = 1:numel( options{ index_operator }{ index_transform }{ index_config } )

                %----------------------------------------------------------
                % D) create images and signal matrices
                %----------------------------------------------------------
                gamma_recon{ index_operator }{ index_transform }{ index_config } ...
                    = discretizations.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, gamma_recon{ index_operator }{ index_transform }{ index_config } );
                theta_recon_normed{ index_operator }{ index_transform }{ index_config } ...
                    = discretizations.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, theta_recon_normed{ index_operator }{ index_transform }{ index_config } );

                % avoid cell arrays for single options{ index_operator }{ index_transform }{ index_config }
                if isscalar( options{ index_operator }{ index_transform }{ index_config } )
                    u_M_res{ index_operator }{ index_transform }{ index_config } = u_M_res{ index_operator }{ index_transform }{ index_config }{ 1 };
                    info{ index_operator }{ index_transform }{ index_config } = info{ index_operator }{ index_transform }{ index_config }{ 1 };
                end

            end % for index_config = 1:numel( operators_born_config )

            % avoid cell arrays for single operators_born_config
            if isscalar( operators_born_config )
                gamma_recon{ index_operator }{ index_transform } = gamma_recon{ index_operator }{ index_transform }{ 1 };
                theta_recon_normed{ index_operator }{ index_transform } = theta_recon_normed{ index_operator }{ index_transform }{ 1 };
                u_M_res{ index_operator }{ index_transform } = u_M_res{ index_operator }{ index_transform }{ 1 };
                info{ index_operator }{ index_transform } = info{ index_operator }{ index_transform }{ 1 };
            end

        end % for index_transform = 1:numel( LTs{ index_operator } )

        % avoid cell arrays for single LTs{ index_operator }
        if isscalar( LTs{ index_operator } )
            gamma_recon{ index_operator } = gamma_recon{ index_operator }{ 1 };
            theta_recon_normed{ index_operator } = theta_recon_normed{ index_operator }{ 1 };
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

end % function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators_born, u_M, options, varargin )
