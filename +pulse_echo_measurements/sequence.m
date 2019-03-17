%
% superclass for all sequential pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-02-19
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setup %( 1, 1 ) pulse_echo_measurements.setup        % pulse-echo measurement setup
        settings %( :, : ) pulse_echo_measurements.setting	% pulse-echo measurement settings

        % dependent properties
        interval_t ( 1, 1 ) physical_values.interval_time       % hull of all recording time intervals
        interval_f ( 1, 1 ) physical_values.interval_frequency	% hull of all frequency intervals

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence( setup, settings )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'sequence:NoSingleSetup';
                error( errorStruct );
            end
            % assertion: setup is a single pulse_echo_measurements.setup

            % ensure class pulse_echo_measurements.setting
            if ~isa( settings, 'pulse_echo_measurements.setting' )
                errorStruct.message     = 'settings must be pulse_echo_measurements.setting!';
                errorStruct.identifier	= 'sequence:NoSetting';
                error( errorStruct );
            end
            % assertion: settings is pulse_echo_measurements.setting

            %--------------------------------------------------------------
            % 2.) set independent and dependent properties
            %--------------------------------------------------------------
            % set independent properties
            object.setup = setup;
            object.settings = settings;

            % set dependent properties
            object.interval_t = hull( [ object.settings.interval_t ] );
            object.interval_f = hull( [ object.settings.interval_f ] );

        end % function object = sequence( setup, settings )

        %------------------------------------------------------------------
        % spatiospectral discretization
        %------------------------------------------------------------------
        function spatiospectral = discretize( sequences, options )
            % TODO: single sequence

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options
            if ~isa( options, 'discretizations.options' )
                errorStruct.message     = 'options_spectral must be discretizations.options_spectral!';
                errorStruct.identifier	= 'discretize:NoOptionsFrequency';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) spatial discretization
            %--------------------------------------------------------------
            spatial = discretize( [ sequences.setup ], [ options.spatial ] );

            %--------------------------------------------------------------
            % 3.) spectral discretization
            %--------------------------------------------------------------
            spectral = discretize( [ sequences.settings ], [ options.spectral ] );

            %--------------------------------------------------------------
            % 4.) create spatiospectral discretization
            %--------------------------------------------------------------
            spatiospectral = discretizations.spatiospectral( spatial, spectral );

        end % function spatiospectral = discretize( object, options )

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
                intervals_t{ index_incident } = physical_values.interval_time( t_lbs, t_ubs );

                % determine hull of time intervals
                hulls( index_incident ) = hull( intervals_t{ index_incident } );

            end % for index_incident = 1:N_incident

        end % function intervals_t = determine_interval_t( object )

        %------------------------------------------------------------------
        % compute incident acoustic fields
        %------------------------------------------------------------------
        function p_incident = compute_p_incident( object, spatiospectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            p_incident = syntheses.pressure_incident( object.setup, spatiospectral );

        end % function p_incident = compute_p_incident( object, spatiospectral )

	end % methods

end % classdef sequence
