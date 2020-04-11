%
% superclass for all scattering operators
%
% author: Martin F. Schiffner
% date: 2019-02-14
% modified: 2020-04-10
%
classdef (Abstract) operator

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence %( 1, 1 ) scattering.sequences.sequence        % pulse-echo measurement sequence
        options ( 1, 1 ) scattering.options                     % scattering operator options

        % dependent properties
        incident_waves ( :, 1 ) scattering.sequences.syntheses.incident_wave             % incident waves
        indices_measurement_sel ( :, 1 ) double { mustBePositive, mustBeInteger } % indices of selected sequential pulse-echo measurements

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = operator( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'operator:NoSequences';
                error( errorStruct );
            end

            % ensure class scattering.options
            if ~isa( options, 'scattering.options' )
                errorStruct.message = 'options must be scattering.options!';
                errorStruct.identifier = 'operator:NoOptions';
                error( errorStruct );
            end

            % multiple sequences / single options
            if ~isscalar( sequences ) && isscalar( options )
                options = repmat( options, size( sequences ) );
            end

            % single sequences / multiple options
            if isscalar( sequences ) && ~isscalar( options )
                sequences = repmat( sequences, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) create scattering operators
            %--------------------------------------------------------------
            % repeat default scattering operator
            objects = repmat( objects, size( sequences ) );

            % iterate scattering operators
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).sequence = sequences( index_object );
                objects( index_object ).options = options( index_object );

                %----------------------------------------------------------
                % b) spatiospectral discretization of the sequence
                %----------------------------------------------------------
                objects( index_object ).sequence = discretize( objects( index_object ).sequence, objects( index_object ).options.static.discretization );

                %----------------------------------------------------------
                % c) incident acoustic fields (unique frequencies)
                %----------------------------------------------------------
%               p_incident = compute_p_in( objects( index_object ).sequence.setup, objects( index_object ).sequence.settings, objects( index_object ).options.momentary.anti_aliasing.tx );
                p_incident = compute_p_in( objects( index_object ).sequence, [], objects( index_object ).options.momentary.anti_aliasing.tx );
                objects( index_object ).incident_waves = scattering.sequences.syntheses.incident_wave( p_incident );

                %----------------------------------------------------------
                % d) apply spatial anti-aliasing filter
                %----------------------------------------------------------
                objects( index_object ).sequence = update_transfer_function( objects( index_object ).sequence, objects( index_object ).options.momentary.anti_aliasing.rx );

% TODO: use update function
                % update indices of selected sequential pulse-echo measurements
                if isa( objects( index_object ).options.momentary.sequence, 'scattering.options.sequence_full' )
                    objects( index_object ).indices_measurement_sel = 1:numel( objects( index_object ).sequence.settings );
                else
                    objects( index_object ).indices_measurement_sel = objects( index_object ).options.momentary.sequence.indices;
                end

            end % for index_object = 1:numel( objects )

        end % function objects = operator( sequences, options )

        %------------------------------------------------------------------
        % forward scattering
        %------------------------------------------------------------------
        function u_M = forward( operators, coefficients, options )

            % print status
            time_start = tic;
            auxiliary.print_header( "forward scattering" );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'forward:NoScatteringOperators';
                error( errorStruct );
            end

            % ensure cell array for coefficients
            if ~iscell( coefficients )
                coefficients = { coefficients };
            end

            % ensure nonempty options
            if nargin < 3 || isempty( options )
% TODO: ensure usage of current momentary options
                options_momentary = reshape( [ operators.options.momentary ], size( operators ) );
                options_energy = regularization.options.energy_rx( options_momentary );
                options = regularization.options.common( options_energy );
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % multiple operators / single options
            if ~isscalar( operators ) && isscalar( options )
                options = repmat( options, size( operators ) );
            end

            % single operators / multiple options
            if isscalar( operators ) && ~isscalar( options )
                operators = repmat( operators, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, coefficients, options );

            %--------------------------------------------------------------
            % 2.) compute mixed voltage signals
            %--------------------------------------------------------------
            % specify cell array for u_M
            u_M = cell( size( operators ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class regularization.options.common
                if ~isa( options{ index_operator }, 'regularization.options.common' )
                    errorStruct.message = sprintf( 'options{ %d } must be regularization.options.common!', index_operator );
                    errorStruct.identifier = 'forward:NoCommonRegularizationOptions';
                    error( errorStruct );
                end

                % ensure numeric matrix
                if ~( isnumeric( coefficients{ index_operator } ) && ismatrix( coefficients{ index_operator } ) )
                    errorStruct.message = sprintf( 'coefficients{ %d } must be a numeric matrix!', index_operator );
                    errorStruct.identifier = 'forward:NoNumericMatrix';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell array for u_M{ index_operator }
                u_M{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    % display options
                    show( options{ index_operator }( index_options ) );

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators( index_operator ) );

                    %------------------------------------------------------
                    % ii.) forward scattering (scalar)
                    %------------------------------------------------------
                    u_M{ index_operator }{ index_options } = forward_scalar( operator_act, coefficients{ index_operator }, LT_dict_act, LT_tgc_act );
                    u_M{ index_operator }{ index_options } = physical_values.volt( u_M{ index_operator }{ index_options } );

                    % create signals or signal matrices
                    u_M{ index_operator }{ index_options } = format_voltages( operator_act, u_M{ index_operator }{ index_options } );

                end % for index_options = 1:numel( options{ index_operator } )

                % avoid cell array for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    u_M{ index_operator } = u_M{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators )

            % avoid cell array for single operators
            if isscalar( operators )
                u_M = u_M{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function u_M = forward( operators, coefficients, options )

        %------------------------------------------------------------------
        % adjoint scattering
        %------------------------------------------------------------------
        function [ gamma_hat, theta_hat, rel_RMSE ] = adjoint( operators, u_M, options )

            % print status
            time_start = tic;
            auxiliary.print_header( "adjoint scattering" );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'adjoint:NoScatteringOperators';
                error( errorStruct );
            end

            % ensure cell array for u_M
            if ~iscell( u_M )
                u_M = { u_M };
            end

            % ensure nonempty options
            if nargin < 3 || isempty( options )
                options = regularization.options.common;
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % method get_configs ensures class regularization.options.common for options

            % multiple operators / single options
            if ~isscalar( operators ) && isscalar( options )
                options = repmat( options, size( operators ) );
            end

            % single operators / multiple options
            if isscalar( operators ) && ~isscalar( options )
                operators = repmat( operators, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, u_M, options );

            %--------------------------------------------------------------
            % 2.) compute adjoint scattering
            %--------------------------------------------------------------
            % specify cell arrays
            gamma_hat = cell( size( operators ) );
            theta_hat = cell( size( operators ) );
            rel_RMSE = cell( size( operators ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class processing.signal_matrix
                if ~isa( u_M{ index_operator }, 'processing.signal_matrix' )
                    errorStruct.message = sprintf( 'u_M{ %d } must be processing.signal_matrix!', index_operator );
                    errorStruct.identifier = 'adjoint:NoSignalMatrices';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell arrays
                gamma_hat{ index_operator } = cell( size( options{ index_operator } ) );
                theta_hat{ index_operator } = cell( size( options{ index_operator } ) );
                rel_RMSE{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    % display options
                    show( options{ index_operator }( index_options ) );

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators( index_operator ) );

                    %------------------------------------------------------
                    % ii.) create mixed voltage signals
                    %------------------------------------------------------
                    % extract relevant mixed voltage signals
% TODO: detect configuration changes first and avoid step if necessary
                    u_M_act = u_M{ index_operator }( operator_act.indices_measurement_sel );

                    % apply TGC and normalize mixed voltage signals
                    u_M_act_vect = return_vector( u_M_act );
                    u_M_act_vect_tgc = forward_transform( LT_tgc_act, u_M_act_vect );
                    u_M_act_vect_tgc_norm = norm( u_M_act_vect_tgc );
                    u_M_act_vect_tgc_normed = u_M_act_vect_tgc / u_M_act_vect_tgc_norm;

                    %------------------------------------------------------
                    % iii.) adjoint scattering (scalar)
                    %------------------------------------------------------
                    theta_hat{ index_operator }{ index_options } = adjoint_scalar( operator_act, u_M_act_vect_tgc_normed, LT_dict_act, LT_tgc_act );

                    %------------------------------------------------------
                    % iv.) apply adjoint linear transform
                    %------------------------------------------------------
                    gamma_hat{ index_operator }{ index_options } = adjoint_transform( LT_dict_act, theta_hat{ index_operator }{ index_options } );

                    %------------------------------------------------------
                    % v.) relative RMSEs by quick forward scattering
                    %------------------------------------------------------
                    if nargout >= 3

                        % estimate normalized mixed voltage signals by quick forward scattering
                        u_M_act_vect_tgc_normed_est = forward_scalar( operator_act, theta_hat{ index_operator }{ index_options }, LT_dict_act, LT_tgc_act );

                        % compute relative RMSE
                        u_M_act_vect_tgc_normed_res = u_M_act_vect_tgc_normed - u_M_act_vect_tgc_normed_est;
                        rel_RMSE{ index_operator }{ index_options } = norm( u_M_act_vect_tgc_normed_res( : ), 2 );

                    end % if nargout >= 3

                end % for index_options = 1:numel( options{ index_operator } )

                %----------------------------------------------------------
                % c) create images
                %----------------------------------------------------------
                gamma_hat{ index_operator } = processing.image( operators( index_operator ).sequence.setup.FOV.shape.grid, gamma_hat{ index_operator } );

                % avoid cell arrays for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    theta_hat{ index_operator } = theta_hat{ index_operator }{ 1 };
                    rel_RMSE{ index_operator } = rel_RMSE{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators )

            % avoid cell arrays for single operators
            if isscalar( operators )
                gamma_hat = gamma_hat{ 1 };
                theta_hat = theta_hat{ 1 };
                rel_RMSE = rel_RMSE{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function [ gamma_hat, theta_hat, rel_RMSE ] = adjoint( operators, u_M, options )

        %------------------------------------------------------------------
        % received energy
        %------------------------------------------------------------------
        function E_M = energy_rx( operators, options )

            % print status
            time_start = tic;
            auxiliary.print_header( "computing received energies" );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'energy_rx:NoScatteringOperators';
                error( errorStruct );
            end

            % ensure nonempty options
            if nargin < 2 || isempty( options )
                options = regularization.options.energy_rx;
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % multiple operators / single options
            if ~isscalar( operators ) && isscalar( options )
                options = repmat( options, size( operators ) );
            end

            % single operators / multiple options
            if isscalar( operators ) && ~isscalar( options )
                operators = repmat( operators, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, options );

            %--------------------------------------------------------------
            % 2.) compute received energies
            %--------------------------------------------------------------
            % specify cell array for E_M
            E_M = cell( size( operators ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class regularization.options.energy_rx
                if ~isa( options{ index_operator }, 'regularization.options.energy_rx' )
                    errorStruct.message = sprintf( 'options{ %d } must be regularization.options.energy_rx!', index_operator );
                    errorStruct.identifier = 'energy_rx:NoEnergyOptions';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell array for E_M{ index_operator }
                E_M{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    %------------------------------------------------------
                    % i.) create configuration (deactivate normalization)
                    %------------------------------------------------------
                    [ operator_act, LT_dict_act, ~, LTs_tgc_measurement ] = get_configs( options{ index_operator }( index_options ), operators( index_operator ) );

                    %------------------------------------------------------
                    % ii.) call received energy (scalar; decomposition)
                    %------------------------------------------------------
                    E_M{ index_operator }{ index_options } = energy_rx_scalar( operator_act, LT_dict_act, LTs_tgc_measurement );

                end % for index_options = 1:numel( options{ index_operator } )

                % avoid cell array for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    E_M{ index_operator } = E_M{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators )

            % avoid cell array for single operators
            if isscalar( operators )
                E_M = E_M{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function E_M = energy_rx( operators, options )

        %------------------------------------------------------------------
        % transform point spread function
        %------------------------------------------------------------------
        function [ gamma_tpsf, theta_tpsf, E_M, adjointness ] = tpsf( operators, options )

            % print status
            time_start = tic;
            auxiliary.print_header( "transform point spread functions (TPSFs)" );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator_born
            if ~isa( operators, 'scattering.operator_born' )
                errorStruct.message = 'operators must be scattering.operator_born!';
                errorStruct.identifier = 'tpsf:NoScatteringOperators';
                error( errorStruct );
            end

            % ensure nonempty options
            if nargin < 2 || isempty( options )
                options = regularization.options.tpsf;
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % multiple operators / single options
            if ~isscalar( operators ) && isscalar( options )
                options = repmat( options, size( operators ) );
            end

            % single operators / multiple options
            if isscalar( operators ) && ~isscalar( options )
                operators = repmat( operators, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, options );

            %--------------------------------------------------------------
            % 2.) compute transform point spread functions (TPSFs)
            %--------------------------------------------------------------
            % specify cell arrays
            gamma_tpsf = cell( size( operators ) );
            theta_tpsf = cell( size( operators ) );
            E_M = cell( size( operators ) );
            adjointness = cell( size( operators ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class regularization.options.tpsf
                if ~isa( options{ index_operator }, 'regularization.options.tpsf' )
                    errorStruct.message = sprintf( 'options{ %d } must be regularization.options.tpsf!', index_operator );
                    errorStruct.identifier = 'tpsf:NoOptionsTPSF';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % specify cell arrays
                gamma_tpsf{ index_operator } = cell( size( options{ index_operator } ) );
                theta_tpsf{ index_operator } = cell( size( options{ index_operator } ) );
                E_M{ index_operator } = cell( size( options{ index_operator } ) );
                adjointness{ index_operator } = cell( size( options{ index_operator } ) );

                % iterate options
                for index_options = 1:numel( options{ index_operator } )

                    % display options
                    show( options{ index_operator }( index_options ) );

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators( index_operator ) );

                    %------------------------------------------------------
                    % ii.) create coefficient vectors
                    %------------------------------------------------------
                    % a) number of TPSFs
                    N_tpsf = numel( options{ index_operator }( index_options ).indices );

                    % b) indices of coefficients
                    indices_tpsf = ( 0:( N_tpsf - 1 ) ) * LT_dict_act.N_coefficients + options{ index_operator }( index_options ).indices';

                    % c) initialize coefficient vectors
                    theta = zeros( LT_dict_act.N_coefficients, N_tpsf );
                    theta( indices_tpsf ) = 1;

                    %------------------------------------------------------
                    % iii.) quick forward scattering and received energies
                    %------------------------------------------------------
                    u_M = forward_scalar( operator_act, theta, LT_dict_act, LT_tgc_act );
                    E_M{ index_operator }{ index_options } = vecnorm( u_M, 2, 1 ).^2;

                    %------------------------------------------------------
                    % iv.) quick adjoint scattering and test for adjointness
                    %------------------------------------------------------
                    theta_tpsf{ index_operator }{ index_options } = adjoint_scalar( operator_act, u_M, LT_dict_act, LT_tgc_act );
                    adjointness{ index_operator }{ index_options } = E_M{ index_operator }{ index_options } - theta_tpsf{ index_operator }{ index_options }( indices_tpsf );

                    %------------------------------------------------------
                    % iv.) apply adjoint linear transform
                    %------------------------------------------------------
                    gamma_tpsf{ index_operator }{ index_options } = adjoint_transform( LT_dict_act, theta_tpsf{ index_operator }{ index_options } );

                end % for index_options = 1:numel( options{ index_operator } )

                %----------------------------------------------------------
                % c) create images
                %----------------------------------------------------------
                gamma_tpsf{ index_operator } = processing.image( operators( index_operator ).sequence.setup.FOV.shape.grid, gamma_tpsf{ index_operator } );

                % avoid cell array for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    theta_tpsf{ index_operator } = theta_tpsf{ index_operator }{ 1 };
                    E_M{ index_operator } = E_M{ index_operator }{ 1 };
                    adjointness{ index_operator } = adjointness{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators )

            % avoid cell array for single operators
            if isscalar( operators )
                theta_tpsf = theta_tpsf{ 1 };
                E_M = E_M{ 1 };
                adjointness = adjointness{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function [ gamma_tpsf, theta_tpsf, E_M, adjointness ] = tpsf( operators, options )

        %------------------------------------------------------------------
        % solve lq-minimization problems
        %------------------------------------------------------------------
        function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators, u_M, options )

            % print status
            time_start = tic;
            auxiliary.print_header( "lq-minimization" );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'lq_minimization:NoScatteringOperators';
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

            % multiple operators / single options
            if ~isscalar( operators ) && isscalar( options )
                options = repmat( options, size( operators ) );
            end

            % single operators / multiple options
            if isscalar( operators ) && ~isscalar( options )
                operators = repmat( operators, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, u_M, options );

            %--------------------------------------------------------------
            % 2.) process scattering operators
            %--------------------------------------------------------------
            % specify cell arrays
            gamma_recon = cell( size( operators ) );
            theta_recon_normed = cell( size( operators ) );
            u_M_res = cell( size( operators ) );
            info = cell( size( operators ) );

            % iterate scattering operators
            for index_operator = 1:numel( operators )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
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

                %----------------------------------------------------------
                % b) process options
                %----------------------------------------------------------
                % name for temporary file
                str_filename = sprintf( 'data/%s/lq_minimization_temp.mat', operators( index_operator ).sequence.setup.str_name );

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

                    %------------------------------------------------------
                    % i.) create configuration
                    %------------------------------------------------------
                    [ operator_act, LT_dict_act, LT_tgc_act ] = get_configs( options{ index_operator }( index_options ), operators( index_operator ) );

                    %------------------------------------------------------
                    % ii.) create mixed voltage signals
                    %------------------------------------------------------
                    % extract relevant mixed voltage signals
% TODO: detect configuration changes first and avoid step if necessary
                    u_M_act = u_M{ index_operator }( operator_act.indices_measurement_sel );

                    % apply TGC and normalize mixed voltage signals
                    u_M_act_vect = return_vector( u_M_act );
                    u_M_act_vect_tgc = forward_transform( LT_tgc_act, u_M_act_vect );
                    u_M_act_vect_tgc_norm = norm( u_M_act_vect_tgc );
                    u_M_act_vect_tgc_normed = u_M_act_vect_tgc / u_M_act_vect_tgc_norm;

                    %------------------------------------------------------
                    % iii.) define anonymous function for sensing matrix
                    %------------------------------------------------------
                    op_A_bar = @( x, mode ) combined_quick( operator_act, mode, x, LT_dict_act, LT_tgc_act );

                    %------------------------------------------------------
                    % iv.) recover transform coefficients and material fluctuations
                    %------------------------------------------------------
                    % execute algorithm
                    [ theta_recon_normed{ index_operator }{ index_options }, ...
                      u_M_act_vect_tgc_normed_res, ...
                      info{ index_operator }{ index_options } ] ...
                    = execute( options{ index_operator }( index_options ).algorithm, op_A_bar, u_M_act_vect_tgc_normed );

                    % invert normalization and apply adjoint linear transform
                    gamma_recon{ index_operator }{ index_options } = adjoint_transform( LT_dict_act, theta_recon_normed{ index_operator }{ index_options } ) * u_M_act_vect_tgc_norm;

                    % format residual mixed RF voltage signals
                    u_M_res{ index_operator }{ index_options } = format_voltages( operator_act, u_M_act_vect_tgc_normed_res * u_M_act_vect_tgc_norm );

                    %------------------------------------------------------
                    % v.) optional steps
                    %------------------------------------------------------
                    % save results to temporary file
                    if options{ index_operator }( index_options ).save_progress
                        save( str_filename, 'gamma_recon', 'theta_recon_normed', 'u_M_res', 'info' );
                    end

                    % display result
                    if options{ index_operator }( index_options ).display

                        figure( index_options );
% TODO: reshape is invalid for transform coefficients! method format_coefficients in linear transform?
%                         temp_1 = squeeze( reshape( theta_recon_normed{ index_operator }{ index_options }( :, end ), operators( index_operator ).sequence.setup.FOV.shape.grid.N_points_axis ) );
                        temp_2 = squeeze( reshape( gamma_recon{ index_operator }{ index_options }( :, end ), operators( index_operator ).sequence.setup.FOV.shape.grid.N_points_axis ) );
                        if ismatrix( temp_2 )
                            subplot( 1, 2, 1 );
%                             imagesc( illustration.dB( temp_1, 20 )', [ -60, 0 ] );
                            subplot( 1, 2, 2 );
                            imagesc( illustration.dB( temp_2, 20 )', [ -60, 0 ] );
                        else
                            subplot( 1, 2, 1 );
%                             imagesc( illustration.dB( squeeze( temp_1( :, 5, : ) ), 20 )', [ -60, 0 ] );
                            subplot( 1, 2, 2 );
                            imagesc( illustration.dB( squeeze( temp_2( :, 5, : ) ), 20 )', [ -60, 0 ] );
                        end
                        colormap gray;

                    end % if options{ index_operator }( index_options ).display

                end % for index_options = 1:numel( options{ index_operator } )

                %----------------------------------------------------------
                % c) create images
                %----------------------------------------------------------
                gamma_recon{ index_operator } = processing.image( operators( index_operator ).sequence.setup.FOV.shape.grid, gamma_recon{ index_operator } );

                % avoid cell arrays for single options{ index_operator }
                if isscalar( options{ index_operator } )
                    theta_recon_normed{ index_operator } = theta_recon_normed{ index_operator }{ 1 };
                    u_M_res{ index_operator } = u_M_res{ index_operator }{ 1 };
                    info{ index_operator } = info{ index_operator }{ 1 };
                end

            end % for index_operator = 1:numel( operators )

            % avoid cell arrays for single operators
            if isscalar( operators )
                gamma_recon = gamma_recon{ 1 };
                theta_recon_normed = theta_recon_normed{ 1 };
                u_M_res = u_M_res{ 1 };
                info = info{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function [ gamma_recon, theta_recon_normed, u_M_res, info ] = lq_minimization( operators, u_M, options )

        %------------------------------------------------------------------
        % set properties of momentary scattering operator options
        %------------------------------------------------------------------
        function operators = set_options_momentary( operators, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'set_options_momentary:NoOperators';
                error( errorStruct );
            end

            % ensure class scattering.options.momentary
            if ~isa( options, 'scattering.options.momentary' )
                errorStruct.message = 'options must be scattering.options.momentary!';
                errorStruct.identifier = 'set_options_momentary:NoMomentaryOptions';
                error( errorStruct );
            end

            % multiple operators / single options
            if ~isscalar( operators ) && isscalar( options )
                options = repmat( options, size( operators ) );
            end

            % single operators / multiple options
            if isscalar( operators ) && ~isscalar( options )
                operators = repmat( operators, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, options );

            %--------------------------------------------------------------
            % 2.) set momentary scattering operator options
            %--------------------------------------------------------------
            % save momentary scattering options
            options_old = reshape( [ operators.options ], size( operators ) );

            % iterate scattering operators
            for index_object = 1:numel( operators )

                %----------------------------------------------------------
                % a) set current momentary scattering options
                %----------------------------------------------------------
                operators( index_object ).options = set_options_momentary( operators( index_object ).options, options( index_object ) );

                %----------------------------------------------------------
                % b) detect changes and update data structures
                %----------------------------------------------------------
                % indices_measurement_sel
                if ~isequal( operators( index_object ).options.momentary.sequence, options_old( index_object ).momentary.sequence )

                    %------------------------------------------------------
                    % i.) sequence options
                    %------------------------------------------------------
                    % update indices of selected sequential pulse-echo measurements
                    if isa( operators( index_object ).options.momentary.sequence, 'scattering.options.sequence_full' )

                        % select all sequential pulse-echo measurements
                        operators( index_object ).indices_measurement_sel = 1:numel( operators( index_object ).sequence.settings );

                    else

                        % ensure valid indices
                        if any( operators( index_object ).options.momentary.sequence.indices > numel( operators( index_object ).sequence.settings ) )
                            errorStruct.message = sprintf( 'operators( %d ).options.momentary.sequence.indices must not exceed %d!', index_object, numel( operators( index_object ).sequence.settings ) );
                            errorStruct.identifier = 'set_options_momentary:InvalidSequenceIndices';
                            error( errorStruct );
                        end

                        % set indices of selected sequential pulse-echo measurements
                        operators( index_object ).indices_measurement_sel = operators( index_object ).options.momentary.sequence.indices;

                    end % if isa( operators( index_object ).options.momentary.sequence, 'scattering.options.sequence_full' )

                end % if ~isequal( operators( index_object ).options.momentary.sequence, options_old( index_object ).momentary.sequence )

                % incident waves
                if ~isequal( operators( index_object ).options.momentary.anti_aliasing.tx, options_old( index_object ).momentary.anti_aliasing.tx )

                    %------------------------------------------------------
                    % ii.) spatial anti-aliasing filter (tx)
                    %------------------------------------------------------
                    operators( index_object ).incident_waves = scattering.sequences.syntheses.incident_wave( operators( index_object ).sequence, operators( index_object ).options.momentary.anti_aliasing.tx );

                end % if ~isequal( operators( index_object ).options.momentary.anti_aliasing.tx, options_old( index_object ).momentary.anti_aliasing.tx )

                % reference spatial transfer function
                if ~isequal( operators( index_object ).options.momentary.anti_aliasing.rx, options_old( index_object ).momentary.anti_aliasing.rx )

                    %------------------------------------------------------
                    % iii.) spatial anti-aliasing filter (rx)
                    %------------------------------------------------------
                    % update reference spatial transfer function w/ anti-aliasing filter
                    operators( index_object ).sequence = update_transfer_function( operators( index_object ).sequence, operators( index_object ).options.momentary.anti_aliasing.rx );

                end % if ~isequal( operators( index_object ).options.momentary.anti_aliasing.rx, options_old( index_object ).momentary.anti_aliasing.rx )

            end % for index_object = 1:numel( operators )

        end % function operators = set_options_momentary( operators, varargin )

        %------------------------------------------------------------------
        % format voltages
        %------------------------------------------------------------------
        function u_M = format_voltages( operators, u_M )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'format_voltages:NoScatteringOperators';
                error( errorStruct );
            end

            % ensure cell array for u_M
            if ~iscell( u_M )
                u_M = { u_M };
            end

            % multiple operators / single u_M
            if ~isscalar( operators ) && isscalar( u_M )
                u_M = repmat( u_M, size( operators ) );
            end

            % single operators / multiple u_M
            if isscalar( operators ) && ~isscalar( u_M )
                operators = repmat( operators, size( u_M ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, u_M );

            %--------------------------------------------------------------
            % 2.) format voltages
            %--------------------------------------------------------------
            % iterate scattering operators
            for index_object = 1:numel( operators )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure numeric matrix
                if ~( isnumeric( u_M{ index_object } ) && ismatrix( u_M{ index_object } ) )
                    errorStruct.message = sprintf( 'u_M{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'format_voltages:NoNumericMatrix';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) create signals or signal matrices
                %----------------------------------------------------------
                % number of columns
                N_columns = size( u_M{ index_object }, 2 );

                % partition numeric matrix into cell arrays
                N_observations = cellfun( @( x ) sum( x( : ) ), { operators( index_object ).sequence.settings( operators( index_object ).indices_measurement_sel ).N_observations } );
                u_M{ index_object } = mat2cell( u_M{ index_object }, N_observations, N_columns );

                % iterate selected sequential pulse-echo measurements
                for index_measurement_sel = 1:numel( operators( index_object ).indices_measurement_sel )

                    % index of sequential pulse-echo measurement
                    index_measurement = operators( index_object ).indices_measurement_sel( index_measurement_sel );

                    % map unique frequencies of pulse-echo measurement to global unique frequencies
                    indices_f_measurement_to_global = operators( index_object ).sequence.indices_f_to_unique{ index_measurement };

                    % map frequencies of mixed voltage signals to unique frequencies of pulse-echo measurement
                    indices_f_mix_to_measurement = operators( index_object ).sequence.settings( index_measurement ).indices_f_to_unique;

                    % partition matrix into cell arrays
                    u_M{ index_object }{ index_measurement_sel } = mat2cell( u_M{ index_object }{ index_measurement_sel }, operators( index_object ).sequence.settings( index_measurement ).N_observations, ones( 1, N_columns ) );

                    % subsample global unique frequencies to get unique frequencies of pulse-echo measurement
                    axis_f_measurement_unique = subsample( operators( index_object ).sequence.axis_f_unique, indices_f_measurement_to_global );

                    % subsample unique frequencies of pulse-echo measurement to get frequencies of mixed voltage signals
                    axes_f_mix = reshape( subsample( axis_f_measurement_unique, indices_f_mix_to_measurement ), [ size( u_M{ index_object }{ index_measurement_sel }, 1 ), 1 ] );

                    % iterate objects
                    signals = cell( 1, N_columns );
                    for index_column = 1:N_columns

                        % create mixed voltage signals
                        signals{ index_column } = processing.signal( axes_f_mix, u_M{ index_object }{ index_measurement_sel }( :, index_column ) );

                        % try to merge mixed voltage signals
                        try
                            signals{ index_column } = merge( signals{ index_column } );
                        catch
                        end

                    end % for index_column = 1:N_columns

                    % store results in cell array
                    u_M{ index_object }{ index_measurement_sel } = signals;

                    % create array of signal matrices
                    if all( cellfun( @( x ) strcmp( class( x ), 'processing.signal_matrix' ), u_M{ index_object }{ index_measurement_sel } ) )
                        u_M{ index_object }{ index_measurement_sel } = cat( 2, u_M{ index_object }{ index_measurement_sel }{ : } );
                    end

                end % for index_measurement_sel = 1:numel( operators( index_object ).indices_measurement_sel )

                % create array of signal matrices
                if all( cellfun( @( x ) strcmp( class( x ), 'processing.signal_matrix' ), u_M{ index_object } ) )
                    u_M{ index_object } = cat( 1, u_M{ index_object }{ : } );
                end

            end % for index_object = 1:numel( operators )

            % avoid cell array for single operators
            if isscalar( operators )
                u_M = u_M{ 1 };
            end

        end % function u_M = format_voltages( operators, u_M )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Hidden)

        %------------------------------------------------------------------
        % received energy (scalar)
        %------------------------------------------------------------------
        E_M = energy_rx_scalar( operator, LT_dict, LTs_tgc_measurement )

	end % methods (Abstract, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % forward scattering (scalar)
        %------------------------------------------------------------------
        u_M_vect = forward_scalar( operator, coefficients, LT_dict, LT_tgc )

        %------------------------------------------------------------------
        % adjoint scattering (scalar)
        %------------------------------------------------------------------
        theta_hat = adjoint_scalar( operator, u_M_vect, LT_dict, LT_tgc )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) operator
