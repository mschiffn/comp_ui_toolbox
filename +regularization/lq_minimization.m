function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators_born, u_M, options )
%
% minimize the lq-norm to recover a sparse coefficient vector
%
% author: Martin F. Schiffner
% date: 2015-06-01
% modified: 2020-02-18
%
% TODO: move to regularization.options.lq_minimization!
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
	if nargin < 3 || isempty( options )
        options = regularization.options.lq_minimization;
    end

	% ensure cell array for options
	if ~iscell( options )
        options = { options };
    end

	% method get_configs ensures class regularization.options.common for options

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
        % ensure class regularization.options.lq_minimization
        if ~isa( options{ index_operator }, 'regularization.options.lq_minimization' )
            errorStruct.message = sprintf( 'options{ %d } must be regularization.options.lq_minimization!', index_operator );
            errorStruct.identifier = 'lq_minimization:NoLqMinimizationOptions';
            error( errorStruct );
        end

        % ensure class processing.signal_matrix
        if ~isa( u_M{ index_operator }, 'processing.signal_matrix' )
            errorStruct.message = sprintf( 'u_M{ %d } must be processing.signal_matrix!', index_operator );
            errorStruct.identifier = 'lq_minimization:NoSignalMatrices';
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

            % display options
            show( options{ index_operator }( index_options ) );

            %--------------------------------------------------------------
            % i.) create configuration
            %--------------------------------------------------------------
            [ operator_born_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators_born( index_operator ) );

            %--------------------------------------------------------------
            % ii.) create mixed voltage signals
            %--------------------------------------------------------------
            % extract relevant mixed voltage signals
% TODO: detect configuration changes first and avoid step if necessary
            u_M_act = u_M{ index_operator }( operator_born_act.indices_measurement_sel );

            % apply TGC and normalize mixed voltage signals
            u_M_act_vect = return_vector( u_M_act );
            u_M_act_vect_tgc = forward_transform( LT_tgc_act, u_M_act_vect );
            u_M_act_vect_tgc_norm = norm( u_M_act_vect_tgc );
            u_M_act_vect_tgc_normed = u_M_act_vect_tgc / u_M_act_vect_tgc_norm;

            %--------------------------------------------------------------
            % iii.) define anonymous function for sensing matrix
            %--------------------------------------------------------------
            op_A_bar = @( x, mode ) combined_quick( operator_born_act, mode, x, LT_dict_act, LT_tgc_act );

            %--------------------------------------------------------------
            % iv.) recover transform coefficients and material fluctuations
            %--------------------------------------------------------------
            % execute algorithm
            [ theta_recon_normed{ index_operator }{ index_options }, ...
              u_M_act_vect_tgc_normed_res, ...
              info{ index_operator }{ index_options } ] ...
            = execute( options{ index_operator }( index_options ).algorithm, op_A_bar, u_M_act_vect_tgc_normed );

            % invert normalization and apply adjoint linear transform
            gamma_recon{ index_operator }{ index_options } = adjoint_transform( LT_dict_act, theta_recon_normed{ index_operator }{ index_options } ) * u_M_act_vect_tgc_norm;

            % format residual mixed RF voltage signals
            u_M_res{ index_operator }{ index_options } = format_voltages( operator_born_act, u_M_act_vect_tgc_normed_res * u_M_act_vect_tgc_norm );

            %--------------------------------------------------------------
            % v.) optional steps
            %--------------------------------------------------------------
            % save results to temporary file
            if options{ index_operator }( index_options ).save_progress
                save( str_filename, 'gamma_recon', 'theta_recon_normed', 'u_M_res', 'info' );
            end

            % display result
            if options{ index_operator }( index_options ).display

                figure( index_options );
% TODO: reshape is invalid for transform coefficients! method format_coefficients in linear transform?
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

            end % if options{ index_operator }( index_options ).display

        end % for index_options = 1:numel( options{ index_operator } )

        %------------------------------------------------------------------
        % c) create images
        %------------------------------------------------------------------
        gamma_recon{ index_operator } = processing.image( operators_born( index_operator ).sequence.setup.FOV.shape.grid, gamma_recon{ index_operator } );

        % avoid cell arrays for single options{ index_operator }
        if isscalar( options{ index_operator } )
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

end % function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators_born, u_M, options )
