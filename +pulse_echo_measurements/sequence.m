%
% superclass for all sequential pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-04-22
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        setup %( 1, 1 ) pulse_echo_measurements.setup        % pulse-echo measurement setup
        settings %( :, : ) pulse_echo_measurements.setting	% pulse-echo measurement settings

        % dependent properties
        interval_t ( 1, 1 ) math.interval	% hull of all recording time intervals
        interval_f ( 1, 1 ) math.interval	% hull of all frequency intervals

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
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup' )
                errorStruct.message     = 'setups must be pulse_echo_measurements.setup!';
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
            % repeat default sequence
            objects = repmat( objects, size( setups ) );

            % iterate sequences
            for index_object = 1:numel( objects )

                % ensure class pulse_echo_measurements.setting
                if ~isa( settings{ index_object }, 'pulse_echo_measurements.setting' )
                    errorStruct.message     = 'settings must be pulse_echo_measurements.setting!';
                    errorStruct.identifier	= 'sequence:NoSetting';
                    error( errorStruct );
                end

                % TODO: ensure that settings are compatible w/ setup

                % set independent properties
                objects( index_object ).setup = setups( index_object );
                objects( index_object ).settings = settings{ index_object };

                % set dependent properties
                objects( index_object ).interval_t = hull( [ objects( index_object ).settings.interval_t ] );
                objects( index_object ).interval_f = hull( [ objects( index_object ).settings.interval_f ] );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence( setups, settings )

        %------------------------------------------------------------------
        % estimate recording time intervals
        %------------------------------------------------------------------
        function [ intervals_t, hulls ] = determine_interval_t( object )

            %--------------------------------------------------------------
            % 1.) lower and upper bounds on the times-of-flight
            %--------------------------------------------------------------
            tof = times_of_flight( object.setup );

            %--------------------------------------------------------------
            % 2.) estimate support of each mix
            %--------------------------------------------------------------
            N_incident = numel( object.settings );
            intervals_t = cell( N_incident, 1 );
            hulls = repmat( tof( 1, 1 ), [ N_incident, 1 ] );

            for index_incident = 1:N_incident

                % indices of active tx elements
                indices_tx_act = object.settings( index_incident ).tx.indices_active;
                N_elements_tx = numel( indices_tx_act );

                % determine support of each mix
                N_mix = numel( object.settings( index_incident ).mixes );

                % initialize lower and upper bounds on the support
                t_lbs = physical_values.time( zeros( 1, N_mix ) );
                t_ubs = physical_values.time( zeros( 1, N_mix ) );

                for index_mix = 1:N_mix

                    % indices of active rx elements
                    indices_rx_act = object.settings( index_incident ).rx( index_mix ).indices_active;
                    N_elements_rx = numel( indices_rx_act );

                    % allocate memory
                    t_lbs_all = physical_values.time( zeros( N_elements_tx, N_elements_rx ) );
                    t_ubs_all = physical_values.time( zeros( N_elements_tx, N_elements_rx ) );

                    % check all combinations of active tx and rx elements
                    for index_tx = 1:N_elements_tx

                        % index of tx array element
                        index_element_tx = indices_tx_act( index_tx );

                        % support of excitation voltage
                        t_lb_tx_act = object.settings( index_incident ).tx.excitation_voltages( index_tx ).set_t.S( 1 ) + object.settings( index_incident ).tx.time_delays( index_tx );
                        t_ub_tx_act = object.settings( index_incident ).tx.excitation_voltages( index_tx ).set_t.S( end ) + object.settings( index_incident ).tx.time_delays( index_tx );

                        for index_rx = 1:N_elements_rx

                            % index of rx array element
                            index_element_rx = indices_rx_act( index_rx );

                            % support of impulse response
                            t_lb_rx_act = object.settings( index_incident ).rx( index_mix ).impulse_responses( index_rx ).set_t.S( 1 );
                            t_ub_rx_act = object.settings( index_incident ).rx( index_mix ).impulse_responses( index_rx ).set_t.S( end );

                            t_lbs_all( index_tx, index_rx ) = t_lb_tx_act + tof( index_element_tx, index_element_rx ).bounds( 1 ) + t_lb_rx_act;
                            t_ubs_all( index_tx, index_rx ) = t_ub_tx_act + tof( index_element_tx, index_element_rx ).bounds( 2 ) + t_ub_rx_act;

                        end % for index_rx = 1:N_elements_rx
                    end % for index_tx = 1:N_elements_tx

                    t_lbs( index_mix ) = min( t_lbs_all );
                    t_ubs( index_mix ) = max( t_ubs_all );

                end % for index_mix = 1:N_mix

                % create time intervals for all mixes
                intervals_t{ index_incident } = math.interval_time( t_lbs, t_ubs );

                % determine hull of time intervals
                hulls( index_incident ) = hull( intervals_t{ index_incident } );

            end % for index_incident = 1:N_incident

        end % function [ intervals_t, hulls ] = determine_interval_t( object )

        %------------------------------------------------------------------
        % spatiospectral discretizations
        %------------------------------------------------------------------
        function spatiospectrals = discretize( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options
            if ~isa( options, 'discretizations.options' )
                errorStruct.message     = 'options must be discretizations.options!';
                errorStruct.identifier	= 'discretize:NoOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) spatial discretizations
            %--------------------------------------------------------------
            setups = reshape( [ sequences.setup ], size( sequences ) );
            spatials = discretize( setups, [ options.spatial ] );

            %--------------------------------------------------------------
            % 3.) spectral discretizations
            %--------------------------------------------------------------
            % specify cell array for spectrals
            spectrals = cell( size( sequences ) );

            % iterate sequential pulse-echo measurements
            for index_object = 1:numel( sequences )

                % discretize pulse-echo measurement settings
                spectrals{ index_object } = discretize( sequences( index_object ).settings, setups( index_object ).absorption_model, options( index_object ).spectral );

            end % for index_object = 1:numel( sequences )

            %--------------------------------------------------------------
            % 4.) create spatiospectral discretizations
            %--------------------------------------------------------------
            spatiospectrals = discretizations.spatiospectral( spatials, spectrals );

        end % function spatiospectrals = discretize( sequences, options )

	end % methods

end % classdef sequence
