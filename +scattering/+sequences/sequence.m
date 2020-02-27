%
% superclass for all sequences of pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2020-02-26
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        setup %( 1, 1 ) scattering.sequences.setups.setup           % pulse-echo measurement setup
        settings %( :, : ) scattering.sequences.settings.setting	% pulse-echo measurement settings

        % dependent properties
        interval_hull_t ( 1, 1 ) math.interval          % hull of all recording time intervals
        interval_hull_f ( 1, 1 ) math.interval          % hull of all frequency intervals

        % dependent properties
        axis_f_unique ( 1, 1 ) math.sequence_increasing	% axis of global unique frequencies
        indices_f_to_unique                             % cell array mapping unique frequencies of each pulse-echo measurement to global unique frequencies
        prefactors                                      % prefactors for scattering (local frequencies)
        size ( 1, : ) double                            % size of the discretization

        % optional properties
        h_ref ( :, 1 ) processing.field            % reference spatial transfer function w/ anti-aliasing filter (unique frequencies)
        h_ref_grad ( :, 1 ) processing.field       % spatial gradient of the reference spatial transfer function w/ anti-aliasing filter (unique frequencies)

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods ( Access = public )

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence( setups, settings )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message     = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier	= 'sequence:NoSetup';
                error( errorStruct );
            end

            % ensure cell array for settings
            if ~iscell( settings )
                settings = { settings };
            end

            % multiple setups / single settings
            if ~isscalar( setups ) && isscalar( settings )
                settings = repmat( settings, size( setups ) );
            end

            % single setups / multiple settings
            if isscalar( setups ) && ~isscalar( settings )
                setups = repmat( setups, size( settings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, settings );

            %--------------------------------------------------------------
            % 2.) create sequences of pulse-echo measurements
            %--------------------------------------------------------------
            % repeat default sequence of pulse-echo measurements
            objects = repmat( objects, size( setups ) );

            % iterate sequences of pulse-echo measurements
            for index_object = 1:numel( objects )

                % ensure class scattering.sequences.settings.setting
                if ~isa( settings{ index_object }, 'scattering.sequences.settings.setting' )
                    errorStruct.message = 'settings must be scattering.sequences.settings.setting!';
                    errorStruct.identifier = 'sequence:NoSetting';
                    error( errorStruct );
                end

% TODO: ensure that settings are compatible w/ setup

                % set independent properties
                objects( index_object ).setup = setups( index_object );
                objects( index_object ).settings = settings{ index_object };

                % set dependent properties
                [ objects( index_object ).interval_hull_t, objects( index_object ).interval_hull_f ] = hulls( objects( index_object ).settings );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence( setups, settings )

        %------------------------------------------------------------------
        % apply windows to mixed voltage signals
        %------------------------------------------------------------------
        function u_M_tilde_window = apply_windows( sequences, u_M_tilde, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'apply_windows:NoSequences';
                error( errorStruct );
            end

            % ensure cell array for u_M_tilde
            if ~iscell( u_M_tilde ) || all( cellfun( @( x ) isa( x, 'processing.signal_matrix' ), u_M_tilde ) )
                u_M_tilde = { u_M_tilde };
            end

            % ensure nonempty setting_window
            if nargin >= 3 && isa( varargin{ 1 }, 'auxiliary.setting_window' )
                setting_window = varargin{ 1 };
            else
                setting_window = repmat( auxiliary.setting_window, size( sequences ) );
            end

            % ensure nonempty indices_measurement
            if nargin >= 4 && ~isempty( varargin{ 2 } )
                indices_measurement = varargin{ 2 };
            else
                indices_measurement = cell( size( sequences ) );
                for index_sequence = 1:numel( sequences )
                    indices_measurement{ index_sequence } = { ( 1:numel( sequences( index_sequence ).settings ) ) };
                end
            end

            % ensure cell array for indices_measurement
            if ~iscell( indices_measurement ) || all( cellfun( @( x ) isnumeric( x ), indices_measurement ) )
                indices_measurement = { indices_measurement };
            end

% TODO: multiple sequences / single u_M_tilde

            % multiple sequences / single setting_window
            if ~isscalar( sequences ) && isscalar( setting_window )
                setting_window = repmat( setting_window, size( sequences ) );
            end

            % multiple sequences / single indices_measurement
            if ~isscalar( sequences ) && isscalar( indices_measurement )
                indices_measurement = repmat( indices_measurement, size( sequences ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, u_M_tilde, setting_window, indices_measurement );

            %--------------------------------------------------------------
            % 2.) apply window functions
            %--------------------------------------------------------------
            % specify cell array for u_M_tilde_window
            u_M_tilde_window = cell( size( sequences ) );

            % iterate sequential pulse-echo measurements
            for index_sequence = 1:numel( sequences )

                % convert array of signal matrices into cell array
                if isa( u_M_tilde{ index_sequence }, 'processing.signal_matrix' )
                    u_M_tilde{ index_sequence } = num2cell( u_M_tilde{ index_sequence } );
                end

                % ensure cell array for indices_measurement{ index_sequence }
                if ~iscell( indices_measurement{ index_sequence } )
                    indices_measurement{ index_sequence } = { indices_measurement{ index_sequence } };
                end

                % specify cell array for u_M_tilde_window{ index_sequence }
                u_M_tilde_window{ index_sequence } = cell( size( indices_measurement{ index_sequence } ) );

                % iterate configurations
                for index_config = 1:numel( indices_measurement{ index_sequence } )

                    % specify cell array for u_M_tilde_window{ index_sequence }{ index_config }
                    u_M_tilde_window{ index_sequence }{ index_config } = cell( size( indices_measurement{ index_sequence }{ index_config } ) );

                    % iterate selected pulse-echo measurement settings
                    for index_measurement_sel = 1:numel( indices_measurement{ index_sequence }{ index_config } )

                        % index of pulse-echo measurement setting
                        index_measurement = indices_measurement{ index_sequence }{ index_config }( index_measurement_sel );

                        % extract recording time intervals for all mixed voltage signals
                        intervals_t = reshape( [ sequences( index_sequence ).settings( index_measurement ).rx.interval_t ], size( sequences( index_sequence ).settings( index_measurement ).rx ) );

                        % extract lower and upper bounds on the recording time intervals
                        lbs = reshape( [ intervals_t.lb ], size( intervals_t ) );
                        ubs = reshape( [ intervals_t.ub ], size( intervals_t ) );

                        % check data structure
                        if isa( u_M_tilde{ index_sequence }{ index_measurement }, 'processing.signal' )

                            %----------------------------------------------
                            % a) single signals
                            %----------------------------------------------
                            % extract axes of all mixed voltage signals
                            axes = reshape( [ u_M_tilde{ index_measurement }.axis ], size( u_M_tilde{ index_measurement } ) );
                            samples_win = cell( size( u_M_tilde{ index_measurement } ) );

                            % iterate mixed voltage signals
                            for index_mix = 1:numel( sequences( index_sequence ).settings( index_measurement ).rx )

                                % determine lower and upper bounds on the windows
                                lbs_max = max( lbs( index_mix ), u_M_tilde{ index_measurement }( index_mix ).axis.members( 1 ) );
                                ubs_min = min( ubs( index_mix ), u_M_tilde{ index_measurement }( index_mix ).axis.members( end ) );

                                % number of samples in the windows
                                indicator_lb = u_M_tilde{ index_measurement }( index_mix ).axis.members >= lbs_max;
                                indicator_ub = u_M_tilde{ index_measurement }( index_mix ).axis.members <= ubs_min;
                                indicator = indicator_lb & indicator_ub;
                                N_samples_window = sum( indicator, 2 );

                                % generate and apply window functions
                                test = window( setting_window( index_sequence ).handle, N_samples_window, setting_window( index_sequence ).parameters{ : } )';
                                samples_win = u_M_tilde{ index_measurement }( index_mix ).samples;
                                samples_win( ~indicator ) = 0;
                                samples_win( indicator ) = samples_win( indicator ) .* test;

                            end % for index_mix = 1:numel( sequences( index_sequence ).settings( index_measurement ).rx )

                            % create signals
                            u_M_tilde{ index_measurement } = processing.signal( axes, samples_win );

                        elseif isa( u_M_tilde{ index_sequence }{ index_measurement }, 'processing.signal_matrix' )

                            %----------------------------------------------
                            % b) signal matrix
                            %----------------------------------------------
                            % determine lower and upper bounds on all windows
                            lbs_max = max( lbs, u_M_tilde{ index_sequence }{ index_measurement }.axis.members( 1 ) );
                            ubs_min = min( ubs, u_M_tilde{ index_sequence }{ index_measurement }.axis.members( end ) );

                            % determine lower and upper bounds on axis
                            lbs_max_min = min( lbs_max, [], 'all' );
                            ubs_min_max = max( ubs_min, [], 'all' );

                            % cut out from signal matrix
                            u_M_tilde_window{ index_sequence }{ index_config }{ index_measurement_sel } = cut_out( u_M_tilde{ index_sequence }{ index_measurement }, lbs_max_min, ubs_min_max );

                            % numbers of samples in all windows
                            indicator_lb = repmat( u_M_tilde_window{ index_sequence }{ index_config }{ index_measurement_sel }.axis.members, [ 1, numel( intervals_t ) ] ) >= lbs_max(:).';
                            indicator_ub = repmat( u_M_tilde_window{ index_sequence }{ index_config }{ index_measurement_sel }.axis.members, [ 1, numel( intervals_t ) ] ) <= ubs_min(:).';
                            indicator = indicator_lb & indicator_ub;
                            N_samples_window = sum( indicator, 1 );

                            % generate and apply window functions
                            samples = u_M_tilde_window{ index_sequence }{ index_config }{ index_measurement_sel }.samples;
                            samples( ~indicator ) = 0;

                            % iterate mixed voltage signals
                            for index_mix = 1:numel( sequences( index_sequence ).settings( index_measurement ).rx )

                                % window function gateway
                                samples_window = window( setting_window( index_sequence ).handle, N_samples_window( index_mix ), setting_window( index_sequence ).parameters{ : } );

                                % apply window function
                                samples( indicator( :, index_mix ), index_mix ) = samples( indicator( :, index_mix ), index_mix ) .* samples_window;

                            end % for index_mix = 1:numel( sequences( index_sequence ).settings( index_measurement ).rx )

                            % periodicity renders last sample redundant
% TODO: not always!
                            axis = remove_last( u_M_tilde_window{ index_sequence }{ index_config }{ index_measurement_sel }.axis );
                            samples = samples( 1:( end - 1 ), : );

                            % create signal matrix
                            u_M_tilde_window{ index_sequence }{ index_config }{ index_measurement_sel } = processing.signal_matrix( axis, samples );

                        else

                            %----------------------------------------------
                            % c) unknown data structure
                            %----------------------------------------------
                            errorStruct.message = sprintf( 'Class of u_M_tilde{ %d }{ %d } is unknown!', index_sequence, index_measurement );
                            errorStruct.identifier = 'apply_windows:UnknownSignalClass';
                            error( errorStruct );

                        end % if isa( u_M_tilde{ index_sequence }{ index_measurement }, 'processing.signal' )

                    end % for index_measurement_sel = 1:numel( indices_measurement{ index_sequence }{ index_config } )

                    % convert cell array into array of processing.signal_matrix
                    indicator = cellfun( @( x ) ~isa( x, 'processing.signal' ), u_M_tilde_window{ index_sequence }{ index_config } );
                    if all( indicator(:) )
                        u_M_tilde_window{ index_sequence }{ index_config } = cat( 1, u_M_tilde_window{ index_sequence }{ index_config }{ : } );
                    end

                end % for index_config = 1:numel( indices_measurement{ index_sequence } )

                % avoid cell array for single indices_measurement{ index_sequence }
                if isscalar( indices_measurement{ index_sequence } )
                    u_M_tilde_window{ index_sequence } = u_M_tilde_window{ index_sequence }{ 1 };
                end

            end % for index_sequence = 1:numel( sequences )

            % avoid cell array for single sequences
            if isscalar( sequences )
                u_M_tilde_window = u_M_tilde_window{ 1 };
            end

        end % function u_M_tilde_window = apply_windows( sequences, u_M_tilde, varargin )

        %------------------------------------------------------------------
        % synthesize mixed voltage signals
        %------------------------------------------------------------------
        function [ u_M_tilde, u_M ] = synthesize_voltage_signals( sequences, u_SA_tilde, varargin )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: syntheses of mixed voltage signals...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'synthesize_voltage_signals:NoSequences';
                error( errorStruct );
            end

            % ensure cell array for u_SA_tilde
            if ~iscell( u_SA_tilde )
                u_SA_tilde = { u_SA_tilde };
            end

            % multiple sequences / single u_SA_tilde
            if ~isscalar( sequences ) && isscalar( u_SA_tilde )
                u_SA_tilde = repmat( u_SA_tilde, size( sequences ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, u_SA_tilde );

            %--------------------------------------------------------------
            % 2.) synthesize mixed voltage signals
            %--------------------------------------------------------------
            % specify cell arrays
            u_M_tilde = cell( size( sequences ) );
            u_M = cell( size( sequences ) );

            % iterate sequential pulse-echo measurements
            for index_object = 1:numel( sequences )

                % ensure class processing.signal_matrix
                if ~isa( u_SA_tilde{ index_object }, 'processing.signal_matrix' )
                    errorStruct.message = sprintf( 'u_SA_tilde{ %d } must be processing.signal_matrix!', index_object );
                    errorStruct.identifier = 'synthesize_voltage_signals:NoSignalMatrices';
                    error( errorStruct );
                end

                % ensure suitable number of signals
                if numel( u_SA_tilde{ index_object } ) ~= sequences( index_object ).setup.xdc_array.N_elements
                    errorStruct.message = sprintf( 'Number of elements in u_SA_tilde{ %d } must equal the number of array elements!', index_object );
                    errorStruct.identifier = 'synthesize_voltage_signals:InvalidSignalMatrices';
                    error( errorStruct );
                end

% TODO: ensure time axes
% TODO: mixing

                % extract unique deltas from all transducer control settings
                deltas_unique = unique( [ unique_deltas( sequences( index_object ).settings ), u_SA_tilde{ index_object }( 1 ).axis.delta ] );

                % largest delta_unique must be integer multiple of smaller deltas_unique
                delta_unique_max = max( deltas_unique );
                mustBeInteger( delta_unique_max ./ deltas_unique );

                % quantize hull of all recording time intervals using delta_unique_max
%                 interval_hull_t_quantized = quantize( sequences( index_object ).interval_hull_t, delta_unique_max );

                % determine recording time intervals
                N_samples_t_SA = cellfun( @abs, { u_SA_tilde{ index_object }.axis } );
                N_samples_t_IR = cellfun( @( x ) abs( x.impulse_responses.axis ), { sequences( index_object ).settings.tx } );

                % quantize hull of all recording time intervals using delta_unique_max
                interval_t = math.interval_quantized( 0, max( N_samples_t_SA ) + max( N_samples_t_IR ), u_SA_tilde{ index_object }( 1 ).axis.delta );
%                 interval_t = math.interval_quantized( 0, 3415, u_SA_tilde{ index_object }( 1 ).axis.delta );
                interval_hull_t_quantized = quantize( interval_t, delta_unique_max );

                % compute Fourier coefficients
                interval_f = math.interval( physical_values.hertz( 0 ), physical_values.hertz( 19e6 ) );
%                 u_SA = fourier_coefficients( u_SA_tilde{ index_object }, abs( interval_hull_t_quantized ), sequences( index_object ).interval_hull_f );
                u_SA = fourier_coefficients( u_SA_tilde{ index_object }, abs( interval_hull_t_quantized ), interval_f );

                % specify cell array for samples
                samples = cell( size( sequences( index_object ).settings ) );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( sequences( index_object ).settings )

                    % print progress in percent
                    fprintf( '%5.1f %%', ( index_measurement - 1 ) / numel( sequences( index_object ).settings ) * 1e2 );

                    % ensure class processing.delta_matrix
                    if ~isa( sequences( index_object ).settings( index_measurement ).tx.impulse_responses, 'processing.delta_matrix' )
                        errorStruct.message = 'u_SA_tilde must be processing.delta_matrix!';
                        errorStruct.identifier = 'synthesize_voltage_signals:NoDeltaMatrix';
                        error( errorStruct );
                    end

                    % compute fourier transforms
%                     impulse_responses = fourier_transform( sequences( index_object ).settings( index_measurement ).tx.impulse_responses, abs( interval_hull_t_quantized ), sequences( index_object ).interval_hull_f );
                    impulse_responses = fourier_transform( sequences( index_object ).settings( index_measurement ).tx.impulse_responses, abs( interval_hull_t_quantized ), interval_f );

                    % number of active array elements
                    N_elements_active_tx = numel( sequences( index_object ).settings( index_measurement ).tx.indices_active );

                    % initialize samples
                    samples{ index_measurement } = physical_values.volt( zeros( size( u_SA( 1 ).samples ) ) );

                    % iterate active array elements
                    for index_active_tx = 1:N_elements_active_tx

                        % index of active array element
                        index_element_tx = sequences( index_object ).settings( index_measurement ).tx.indices_active( index_active_tx );

                        % compute voltage signals
                        samples{ index_measurement } = samples{ index_measurement } + physical_values.volt( double( u_SA( index_element_tx ).samples .* impulse_responses.samples( :, index_active_tx ) ) );

                    end % for index_active_tx = 1:N_elements_active_tx

                    % erase progress in percent
                    fprintf( '\b\b\b\b\b\b\b' );

                end % for index_measurement = 1:numel( sequences( index_object ).settings )

                % create signal matrices
                u_M{ index_object } = processing.signal_matrix( u_SA( 1 ).axis, samples );

                % compute samples in time domain
                u_M_tilde{ index_object } = signal( u_M{ index_object }, 0, u_SA_tilde{ index_object }( 1 ).axis.delta );

            end % for index_object = 1:numel( sequences )

            % avoid cell arrays for single sequences
            if isscalar( sequences )
                u_M_tilde = u_M_tilde{ 1 };
                u_M = u_M{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function [ u_M_tilde, u_M ] = synthesize_voltage_signals( sequences, u_SA_tilde, varargin )

        %------------------------------------------------------------------
        % spatiospectral discretization
        %------------------------------------------------------------------
        function sequences = discretize( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'discretize:NoSequences';
                error( errorStruct );
            end

            % ensure class scattering.options.discretization
            if ~isa( options, 'scattering.options.discretization' )
                errorStruct.message = 'options must be scattering.options.discretization!';
                errorStruct.identifier = 'discretize:NoDiscretizationOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) spatiospectral discretizations
            %--------------------------------------------------------------
            % iterate sequences of pulse-echo measurements
            for index_object = 1:numel( sequences )

                % spatial discretization
                sequences( index_object ).setup = discretize( sequences( index_object ).setup, options( index_object ).spatial );

                % spectral discretization
                sequences( index_object ).settings = discretize( sequences( index_object ).settings, options( index_object ).spectral );

                % extract unique frequency axis
                v_d_unique = reshape( [ sequences( index_object ).settings.v_d_unique ], size( sequences( index_object ).settings ) );
                [ sequences( index_object ).axis_f_unique, ~, sequences( index_object ).indices_f_to_unique ] = unique( [ v_d_unique.axis ] );

                % compute prefactors for scattering (local frequencies)
% TODO: What is the exact requirement?
                if isa( sequences( index_object ).setup.FOV.shape.grid, 'math.grid_regular' )
                    sequences( index_object ).prefactors = compute_prefactors( sequences( index_object ) );
                end

                % size of the discretization
                sequences( index_object ).size = [ sum( cellfun( @( x ) sum( x( : ) ), { sequences( index_object ).settings.N_observations } ) ), sequences( index_object ).setup.FOV.shape.grid.N_points ];

            end % for index_object = 1:numel( sequences )

        end % function sequences = discretize( sequences, options )

        %------------------------------------------------------------------
        % update reference spatial transfer function
        %------------------------------------------------------------------
        function sequences = update_transfer_function( sequences, filters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'update_transfer_function:NoSequences';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.setup_grid_symmetric
            indicator = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.setup_grid_symmetric' ), { sequences.setup } );
            if any( indicator( : ) )
                errorStruct.message = 'sequences.setup must be scattering.sequences.setups.setup_grid_symmetric!';
                errorStruct.identifier = 'update_transfer_function:NoSymmetricSetup';
                error( errorStruct );
            end

            % ensure nonempty filters
            if nargin < 2 || isempty( filters )
                filters = scattering.options.anti_aliasing_off;
            end

            % ensure class scattering.options.anti_aliasing
            if ~isa( filters, 'scattering.options.anti_aliasing' )
                errorStruct.message = 'filters must be scattering.options.anti_aliasing!';
                errorStruct.identifier = 'update_transfer_function:NoSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % multiple sequences / single filters
            if ~isscalar( sequences ) && isscalar( filters )
                filters = repmat( filters, size( sequences ) );
            end

            % single sequences / multiple filters
            if isscalar( sequences ) && ~isscalar( filters )
                sequences = repmat( sequences, size( filters ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, filters );

            %--------------------------------------------------------------
            % 2.) update reference spatial transfer function
            %--------------------------------------------------------------
            % iterate sequences of pulse-echo measurements
            for index_sequence = 1:numel( sequences )

                % compute reference spatial transfer function (unique frequencies)
                sequences( index_sequence ).h_ref = transfer_function( sequences( index_sequence ).setup, sequences( index_sequence ).axis_f_unique, [], filters( index_sequence ) );

            end % for index_sequence = 1:numel( sequences )

        end % function sequences = update_transfer_function( sequences, filters )

        %------------------------------------------------------------------
        % compute prefactors (local frequencies)
        %------------------------------------------------------------------
        function prefactors = compute_prefactors( sequences )

            % specify cell array for prefactors
            prefactors = cell( size( sequences ) );

            % iterate spatiospectral discretizations
            for index_object = 1:numel( sequences )

                % compute prefactors (global unique frequencies)
                prefactors_unique = compute_prefactors( sequences( index_object ).setup, sequences( index_object ).axis_f_unique );

                % cell array mapping unique frequencies of each pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = sequences( index_object ).indices_f_to_unique;

                % subsample prefactors (unique frequencies of each pulse-echo measurement)
                prefactors_measurement = subsample( prefactors_unique, indices_f_measurement_to_global );

                % specify cell array for prefactors{ index_object }
                prefactors{ index_object } = cell( size( sequences( index_object ).settings ) );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( sequences( index_object ).settings )

                    % cell array mapping frequencies of each mixed voltage signal to unique frequencies of current pulse-echo measurement
                    indices_f_mix_to_measurement = sequences( index_object ).settings( index_measurement ).indices_f_to_unique;

                    % subsample prefactors (frequencies of each mixed voltage signal)
                    prefactors_mix = subsample( prefactors_measurement( index_measurement ), indices_f_mix_to_measurement );

                    % extract impulse responses of mixing channels
                    impulse_responses_rx = reshape( [ sequences( index_object ).settings( index_measurement ).rx.impulse_responses ], size( sequences( index_object ).settings( index_measurement ).rx ) );

                    % compute prefactors (frequencies of each mixed voltage signal)
                    prefactors{ index_object }{ index_measurement } = prefactors_mix .* impulse_responses_rx;

                end % for index_measurement = 1:numel( sequences( index_object ).settings )

            end % for index_object = 1:numel( sequences )

            % avoid cell array for single sequences
            if isscalar( sequences )
                prefactors = prefactors{ 1 };
            end

        end % function prefactors = compute_prefactors( sequences )

        %------------------------------------------------------------------
        % compute incident acoustic pressure fields
        %------------------------------------------------------------------
        function fields = compute_p_in( sequences, indices_incident, filters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'compute_p_in:NoSequences';
                error( errorStruct );
            end
% TODO: fix indices_incident
            % ensure cell array for indices_incident
            if ~iscell( indices_incident )
                indices_incident = { indices_incident };
            end

            % ensure nonempty filters
            if nargin < 3 || isempty( filters )
                filters = scattering.options.anti_aliasing_off;
            end

            % ensure class scattering.options.anti_aliasing
            if ~isa( filters, 'scattering.options.anti_aliasing' )
                errorStruct.message = 'filters must be scattering.options.anti_aliasing!';
                errorStruct.identifier = 'compute_p_in:NoSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % multiple sequences / single indices_incident
            if ~isscalar( sequences ) && isscalar( indices_incident )
                indices_incident = repmat( indices_incident, size( sequences ) );
            end

            % single sequences / multiple indices_incident
            if isscalar( sequences ) && ~isscalar( indices_incident )
                sequences = repmat( sequences, size( indices_incident ) );
            end

            % multiple sequences / single filters
            if ~isscalar( sequences ) && isscalar( filters )
                filters = repmat( filters, size( sequences ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, indices_incident, filters );

            %--------------------------------------------------------------
            % 2.) compute incident acoustic pressure fields
            %--------------------------------------------------------------
            % specify cell arrays
            fields = cell( size( sequences ) );

            % iterate sequences of pulse-echo measurements
            for index_sequence = 1:numel( sequences )

                %----------------------------------------------------------
                % a) check indices indices_incident{ index_sequence }
                %----------------------------------------------------------
                % ensure nonempty indices_incident{ index_sequence }
                if nargin < 2 || isempty( indices_incident{ index_sequence } )
                    indices_incident{ index_sequence } = ( 1:numel( sequences( index_sequence ).settings ) );
                end

                % ensure nonempty positive integers
                mustBeNonempty( indices_incident{ index_sequence } );
                mustBeInteger( indices_incident{ index_sequence } );
                mustBePositive( indices_incident{ index_sequence } );

                % ensure that indices_incident{ index_sequence } do not exceed the number of sequential pulse-echo measurements
                if any( indices_incident{ index_sequence } > numel( sequences( index_sequence ).settings ) )
                    errorStruct.message = sprintf( 'indices_incident{ %d } must not exceed the number of sequential pulse-echo measurements!', index_sequence );
                    errorStruct.identifier = 'compute_p_in:InvalidMeasurement';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute field samples
                %----------------------------------------------------------
                % map unique frequencies of pulse-echo measurement to global unique frequencies
                indices_f_measurement_to_global = sequences( index_sequence ).indices_f_to_unique{ indices_incident{ index_sequence } };

                % subsample global unique frequencies to get unique frequencies of pulse-echo measurements
                axes_f_measurement_unique = subsample( sequences( index_sequence ).axis_f_unique, indices_f_measurement_to_global );

                % specify cell array for fields{ index_sequence }
                fields{ index_sequence } = cell( size( indices_incident{ index_sequence } ) );

                % iterate pulse-echo measurements
                for index_incident_sel = 1:numel( indices_incident{ index_sequence } )

                    % index of selected pulse-echo measurement
                    index_incident = indices_incident{ index_sequence }( index_incident_sel );

                    % create format string for filename
                    str_format = sprintf( 'data/%s/setup_%%s/p_in_indices_active_%%s_v_d_unique_%%s_aliasing_%%s.mat', sequences( index_sequence ).setup.str_name );

                    % load or compute incident acoustic pressure field (scalar)
% TODO: loading and saving optional
                    fields{ index_sequence }{ index_incident_sel } = ...
                    auxiliary.compute_or_load_hash( str_format, @compute_p_in_scalar, [ 4, 5, 6, 3 ], [ 1, 2, 3 ], ...
                                                    sequences( index_sequence ), index_incident, filters( index_sequence ), ...
                                                    { sequences( index_sequence ).setup.xdc_array.aperture, sequences( index_sequence ).setup.homogeneous_fluid, sequences( index_sequence ).setup.FOV, sequences( index_sequence ).setup.str_name }, ...
                                                    sequences( index_sequence ).settings( index_incident ).tx_unique.indices_active, ...
                                                    sequences( index_sequence ).settings( index_incident ).v_d_unique );

                end % for index_incident_sel = 1:numel( indices_incident{ index_sequence } )

                %----------------------------------------------------------
                % c) create fields
                %----------------------------------------------------------
                fields{ index_sequence } = processing.field( axes_f_measurement_unique, sequences( index_sequence ).setup.FOV.shape.grid, fields{ index_sequence } );

            end % for index_sequence = 1:numel( sequences )

            % avoid cell array for single sequences
            if isscalar( sequences )
                fields = fields{ 1 };
            end

        end % function fields = compute_p_in( sequences, indices_incident, filters )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % compute incident acoustic pressure field (scalar)
        %------------------------------------------------------------------
        function p_in_samples = compute_p_in_scalar( sequence, index_incident, filter )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing incident acoustic pressure field (kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.sequence (scalar) for sequence
            % calling function ensures nonempty positive integer that does not exceed the number of sequential pulse-echo measurements for index_incident
            % calling function ensures class scattering.options.anti_aliasing for filter

            %--------------------------------------------------------------
            % 2.) compute incident acoustic pressure field (scalar)
            %--------------------------------------------------------------
            if isa( sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                %----------------------------------------------------------
                % precompute spatial transfer function for symmetric grid
                %----------------------------------------------------------
                % map unique frequencies of selected pulse-echo measurement to global unique frequencies
                indices_f_to_unique_act = sequence.indices_f_to_unique{ index_incident };

                % compute reference spatial transfer function (global unique frequencies)
                h_ref_unique = transfer_function( sequence.setup, sequence.axis_f_unique, [], filter );
                h_ref_unique = double( h_ref_unique.samples( indices_f_to_unique_act, : ) );

            end % if isa( sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

            % extract transducer control settings in synthesis mode (unique frequencies of selected pulse-echo measurement)
            settings_tx_unique = sequence.settings( index_incident ).tx_unique;

            % extract normal velocities (unique frequencies of selected pulse-echo measurement)
            v_d_unique = sequence.settings( index_incident ).v_d_unique;

            % extract frequency axes (unique frequencies of selected pulse-echo measurement)
            axis_f_unique_measurement = v_d_unique.axis;
            N_samples_f = abs( axis_f_unique_measurement );

            % initialize pressure samples with zeros
            p_in_samples = physical_values.pascal( zeros( N_samples_f, sequence.setup.FOV.shape.grid.N_points ) );

            % iterate active array elements
            for index_active = 1:numel( settings_tx_unique.indices_active )

                % index of active array element
                index_element = settings_tx_unique.indices_active( index_active );

                % spatial transfer function of the active array element
                if isa( sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                    %------------------------------------------------------
                    % a) symmetric spatial discretization based on orthogonal regular grids
                    %------------------------------------------------------
                    % shift reference spatial transfer function to infer that of the active array element
                    indices_occupied_act = sequence.setup.indices_grid_FOV_shift( :, index_element );
                    h_tx_unique = h_ref_unique( :, indices_occupied_act );

                else

                    %------------------------------------------------------
                    % b) arbitrary grid
                    %------------------------------------------------------
                    % compute spatial transfer function of the active array element
                    h_tx_unique = transfer_function( sequence.setup, axis_f_unique_measurement, index_element, filter );
                    h_tx_unique = double( h_tx_unique.samples );

                end % if isa( sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                % compute summand for the incident pressure field
                p_in_samples_summand = h_tx_unique .* double( v_d_unique.samples( :, index_active ) );

                % add summand to the incident pressure field
% TODO: correct unit problem
                p_in_samples = p_in_samples + physical_values.pascal( p_in_samples_summand );

                % display result
                figure( index_incident );
                test = squeeze( reshape( p_in_samples( 1, : ), sequence.setup.FOV.shape.grid.N_points_axis ) );
                if ndims( test ) == 2
                    imagesc( abs( double( test ) )' );
                else
                    imagesc( abs( double( squeeze( test( :, 1, : ) ) ) )' );
                end

            end % for index_active = 1:numel( settings_tx_unique.indices_active )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function p_in_samples = compute_p_in_scalar( sequence, index_incident, filter )

	end % methods (Access = private, Hidden)

end % classdef sequence
