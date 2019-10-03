%
% superclass for all pulse-echo measurement settings
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-08-21
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( :, 1 ) controls.setting_tx     % synthesis settings
        rx ( :, 1 ) controls.setting_rx     % mixer settings

        % dependent properties
        interval_hull_t ( 1, 1 ) math.interval	% hull of all recording time intervals
        interval_hull_f ( 1, 1 ) math.interval	% hull of all frequency intervals

        % discretization
        tx_unique ( :, : ) controls.setting_tx                  % synthesis settings (unique frequencies)
        v_d_unique ( :, 1 ) discretizations.signal_matrix       % normal velocities (unique frequencies)
        indices_f_to_unique                                     % cell array mapping frequencies of each mixed voltage signal to unique frequencies of current pulse-echo measurement
        indices_active_rx_unique ( 1, : ) double
        indices_active_rx_to_unique
        N_observations ( :, : ) double                          % numbers of observations in each mixed voltage signal

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting( settings_tx, settings_rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for settings_rx
            if ~iscell( settings_rx )
                settings_rx = { settings_rx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 2.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            % repeat default pulse-echo measurement setting
            objects = repmat( objects, size( settings_tx ) );

            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings_tx )

                % set independent properties
                objects( index_object ).tx = settings_tx( index_object );
                objects( index_object ).rx = settings_rx{ index_object };

                % set dependent properties
                [ objects( index_object ).interval_hull_t, objects( index_object ).interval_hull_f ] = hulls( objects( index_object ).rx );

            end % for index_object = 1:numel( settings_tx )

        end % function objects = setting( settings_tx, settings_rx )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function settings = discretize( settings, options_spectral )
% TODO: various types of discretization (e.g. regular vs irregular)

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setting
            if ~isa( settings, 'pulse_echo_measurements.setting' )
                errorStruct.message = 'settings must be pulse_echo_measurements.setting!';
                errorStruct.identifier = 'discretize:NoSetting';
                error( errorStruct );
            end

            % ensure class discretizations.options_spectral (scalar)
            if ~( isa( options_spectral, 'discretizations.options_spectral' ) && isscalar( options_spectral ) )
                errorStruct.message = 'options_spectral must be a single discretizations.options_spectral!';
                errorStruct.identifier = 'discretize:NoSingleOptionsSpectral';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) discretize frequency intervals
            %--------------------------------------------------------------
            % specify cell arrays for settings_rx and settings_tx
            settings_rx = cell( size( settings ) );
            settings_tx = cell( size( settings ) );

            % check spectral discretization options
            if isa( options_spectral, 'discretizations.options_spectral_signal' )

                %----------------------------------------------------------
                % a) individual frequency axis for each recorded signal
                %----------------------------------------------------------
                % iterate pulse-echo measurement settings
                for index_object = 1:numel( settings )
% TODO: quantize intervals_t?

                    % time and frequency intervals of each recorded signal
                    intervals_t = reshape( [ settings( index_object ).rx.interval_t ], size( settings( index_object ).rx ) );
                    intervals_f = reshape( [ settings( index_object ).rx.interval_f ], size( settings( index_object ).rx ) );

                    % discretize rx and tx settings
                    settings_rx{ index_object } = discretize( settings( index_object ).rx );
                    settings_tx{ index_object } = discretize( settings( index_object ).tx, abs( intervals_t ), intervals_f );

                end % for index_object = 1:numel( settings )

            elseif isa( options_spectral, 'discretizations.options_spectral_setting' )

                %----------------------------------------------------------
                % b) common frequency axis for all recorded signals per setting
                %----------------------------------------------------------
                % iterate pulse-echo measurement settings
                for index_object = 1:numel( settings )

                    % extract unique deltas from current transducer control settings
                    deltas_unique = unique_deltas( settings( index_object ) );

                    % largest delta_unique must be integer multiple of smaller deltas_unique
                    delta_unique_max = max( deltas_unique );
                    if any( abs( delta_unique_max ./ deltas_unique - round( delta_unique_max ./ deltas_unique ) ) > eps( round( delta_unique_max ./ deltas_unique ) ) )
                        errorStruct.message = 'delta_unique_max must be integer multiple of all deltas_unique!';
                        errorStruct.identifier = 'discretize:NoIntegerMultiple';
                        error( errorStruct );
                    end

                    % quantize hull of all recording time intervals using largest delta
                    interval_hull_t_quantized = quantize( settings( index_object ).interval_hull_t, delta_unique_max );

                    % discretize rx and tx settings
                    settings_rx{ index_object } = discretize( settings( index_object ).rx, abs( interval_hull_t_quantized ), settings( index_object ).interval_hull_f );
                    settings_tx{ index_object } = discretize( settings( index_object ).tx, abs( interval_hull_t_quantized ), settings( index_object ).interval_hull_f );

                end % for index_object = 1:numel( settings )

            elseif isa( options_spectral, 'discretizations.options_spectral_sequence' )

                %----------------------------------------------------------
                % c) common frequency axis for all recorded signals
                %----------------------------------------------------------
                % extract unique deltas from all transducer control settings
                deltas_unique = unique_deltas( settings );

                % largest delta_unique must be integer multiple of smaller deltas_unique
                delta_unique_max = max( deltas_unique );
                if any( abs( delta_unique_max ./ deltas_unique - round( delta_unique_max ./ deltas_unique ) ) > eps( round( delta_unique_max ./ deltas_unique ) ) )
                    errorStruct.message = 'delta_unique_max must be integer multiple of all deltas_unique!';
                    errorStruct.identifier = 'discretize:NoIntegerMultiple';
                    error( errorStruct );
                end

                % determine hulls of all time and frequency intervals
                [ interval_hull_t, interval_hull_f ] = hulls( settings );

                % check specification of custom recording time interval
                if isa( options_spectral, 'discretizations.options_spectral_sequence_custom' )

                    % ensure valid recording time interval
                    if ~isequal( hull( [ options_spectral.interval_hull_t, interval_hull_t ] ), options_spectral.interval_hull_t )
                        errorStruct.message = 'options_spectral.interval_hull_t must contain interval_hull_t!';
                        errorStruct.identifier = 'discretize:InvalidCustomRecordingTimeInterval';
                        error( errorStruct );
                    end

                    % use custom recording time interval
                    interval_hull_t = options_spectral.interval_hull_t;

                end % if isa( options_spectral, 'discretizations.options_spectral_sequence_custom' )

                % quantize hull of all recording time intervals using delta_unique_max
                interval_hull_t_quantized = quantize( interval_hull_t, delta_unique_max );

                % iterate pulse-echo measurement settings
                for index_object = 1:numel( settings )

                    % discretize rx and tx settings
                    settings( index_object ).rx = discretize( settings( index_object ).rx, abs( interval_hull_t_quantized ), interval_hull_f );
                    settings( index_object ).tx = discretize( settings( index_object ).tx, abs( interval_hull_t_quantized ), interval_hull_f );

                end % for index_object = 1:numel( settings )

            else

                %----------------------------------------------------------
                % d) unknown spectral discretization options
                %----------------------------------------------------------
                errorStruct.message = 'Class of options_spectral is unknown!';
                errorStruct.identifier = 'discretize:UnknownOptionsClass';
                error( errorStruct );

            end % if isa( options_spectral, 'discretizations.options_spectral_signal' )

            %--------------------------------------------------------------
            % 3.) ensure identical frequency axes
            %--------------------------------------------------------------
            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings )

                % ensure that settings( index_object ).rx and settings( index_object ).tx have identical frequency axes
                IR_rx = [ settings( index_object ).rx.impulse_responses ];
                IR_tx = [ settings( index_object ).tx.impulse_responses ];

                if ~isequal( IR_rx.axis, IR_tx.axis )
                    errorStruct.message = sprintf( 'settings( %d ).rx and settings( %d ).tx must have identical frequency axes!', index_object, index_object );
                    errorStruct.identifier = 'discretize:MismatchFrequencyAxis';
                    error( errorStruct );
                end

            end % for index_object = 1:numel( settings )

            %--------------------------------------------------------------
            % 4.) compute dependent properties
            %--------------------------------------------------------------
            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings )

                % determine tx settings for unique frequencies
                [ settings( index_object ).tx_unique, ~, settings( index_object ).indices_f_to_unique ] = unique( settings( index_object ).tx );
                if isscalar( settings( index_object ).tx )
                    settings( index_object ).indices_f_to_unique = repmat( settings( index_object ).indices_f_to_unique, size( settings( index_object ).rx ) );
                end
%               [ settings( index_object ).indices_active_rx_unique, settings( index_object ).indices_active_rx_to_unique ] = unique_indices_active( settings( index_object ).rx );

                % numbers of observations in each mixed voltage signal
                settings( index_object ).N_observations = compute_N_observations( settings( index_object ).rx );

                % compute normal velocities (unique frequencies)
                settings( index_object ).v_d_unique = compute_normal_velocities( settings( index_object ).tx_unique );

            end % for index_object = 1:numel( settings )

        end % function settings = discretize( settings, options_spectral )

        %------------------------------------------------------------------
        % unique deltas
        %------------------------------------------------------------------
        function deltas_unique = unique_deltas( settings )

            % extract unique deltas from all transducer control settings in synthesis mode
            deltas_unique_tx = unique_deltas( [ settings.tx ] );

            % iterate pulse-echo measurement settings
            deltas_unique_rx = cell( size( settings ) );
            for index_setting = 1:numel( settings )

                % extract unique deltas from all transducer control settings in recording mode
                deltas_unique_rx{ index_setting } = unique_deltas( settings( index_setting ).rx );

            end % for index_setting = 1:numel( settings )

            % ensure equal subclasses of physical_values.physical_quantity
            auxiliary.mustBeEqualSubclasses( 'physical_values.physical_quantity', deltas_unique_tx, deltas_unique_rx{ : } );

            % extract unique deltas from all transducer control settings
            deltas_unique = unique( [ deltas_unique_tx, cat( 2, deltas_unique_rx{ : } ) ] );

        end % function deltas_unique = unique_deltas( settings )

        %------------------------------------------------------------------
        % convex hulls of all intervals
        %------------------------------------------------------------------
        function [ interval_hull_t, interval_hull_f ] = hulls( settings )

            % convex hull of all recording time intervals
            interval_hull_t = hull( [ settings.interval_hull_t ] );

            % convex hull of all frequency intervals
            interval_hull_f = hull( [ settings.interval_hull_f ] );

        end % function [ interval_hull_t, interval_hull_f ] = hulls( settings )

	end % methods

end % classdef setting
