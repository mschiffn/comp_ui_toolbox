%
% superclass for all sequential pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-01-24
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setup %( 1, 1 ) pulse_echo_measurements.setup                        % pulse-echo measurement setup
        measurements %( :, 1 ) pulse_echo_measurements.measurement             % column vector of sequential pulse-echo measurements
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = sequence( setup, settings_tx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'sequence_QPW:NoSetup';
                error( errorStruct );
            end
            % assertion: setup is a single pulse_echo_measurements.setup

            %TODO: check symmetry of setup and choose class accordingly
            % set independent properties
            obj.setup	 = setup;

            %--------------------------------------------------------------
            % 2.) quantize tx settings
            %--------------------------------------------------------------
            settings_tx_quantized = quantize( settings_tx, 1 / obj.setup.f_clk );

            %--------------------------------------------------------------
            % 3.) create rx settings
            %--------------------------------------------------------------
            % determine recording time intervals
            % TODO: include excitation voltages, impulse responses, etc. (pulse-echo responses)
            T_pulse_echo = 3.3500e-06;
            intervals_t = determine_interval_t( obj, settings_tx_quantized, T_pulse_echo );

            % determine frequency intervals
            intervals_f = determine_interval_f( obj, intervals_t );

            % create pulse-echo measurements
            % TODO: check method to determine Fourier coefficients
            f_s = 20e6;
            obj.measurements = pulse_echo_measurements.measurement_ADC( intervals_t, intervals_f, settings_tx_quantized, [], f_s * ones( size( intervals_t ) ) );

            % TODO: check for identical recording time intervals / identical frequency intervals
            % check for identical frequency axes identical?
%             N_objects = size( measurements, 1 );
%             for index_object = 1:N_objects
%                 measurements( index_object )
%             end

        end

        %------------------------------------------------------------------
        % determine recording time intervals
        %------------------------------------------------------------------
        function intervals_t = determine_interval_t( obj, settings_tx, T_pulse_echo )

            %--------------------------------------------------------------
            % 1. minimum and maximum times of flight (geometric)
            %--------------------------------------------------------------
            t_tof_lb = 2 * obj.setup.FOV.offset_axis(2) / obj.setup.c_avg;
            t_tof_ub = 2 * sqrt( ( obj.setup.FOV.offset_axis(1) + obj.setup.FOV.size_axis(1) + obj.setup.xdc_array.width_axis(1) / 2 )^2 + ( obj.setup.FOV.offset_axis(2) + obj.setup.FOV.size_axis(2) )^2 ) / obj.setup.c_avg;

            %--------------------------------------------------------------
            % 2. maximum time delays in the syntheses
            %--------------------------------------------------------------
            N_incident = size( settings_tx, 1 );
            time_delay_max = zeros( N_incident, 1 );
            for index_incident = 1:N_incident
                time_delay_max( index_incident ) = max( [ settings_tx( index_incident ).time_delays.value ] );
            end

            %--------------------------------------------------------------
            % 3. time duration of pulse-echo response
            %--------------------------------------------------------------
            t_lb = physical_values.time( t_tof_lb );
            t_ub = physical_values.time( t_tof_ub + time_delay_max + T_pulse_echo );
            % assertion: t_lb >= 0, t_ub > t_lb

            % create time intervals
            intervals_t = physical_values.time_interval( [ repmat( t_lb, [ N_incident, 1 ] ), t_ub ] );
        end

        %------------------------------------------------------------------
        % determine frequency intervals
        %------------------------------------------------------------------
        function intervals_f = determine_interval_f( obj, intervals_t )

            %
            N_incident = size( intervals_t, 1 );
            f_lb = physical_values.frequency( 2.6e6 );
            f_ub = physical_values.frequency( 5.4e6 );
            % TODO: assertion: f_lb > 0, f_ub >= f_lb + 1 / T_rec

            % create frequency intervals
            intervals_f = physical_values.frequency_interval( repmat( [ f_lb, f_ub ], [ N_incident, 1 ] ) );
        end

        %------------------------------------------------------------------
        % compute incident acoustic fields
        %------------------------------------------------------------------
        function p_incident = compute_p_incident( obj )

            p_incident = syntheses.pressure_incident( obj.setup, obj.measurements );
        end
	end % methods

end % classdef sequence
