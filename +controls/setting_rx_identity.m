%
% superclass for all pulse-echo measurement settings w/o mixing
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-02-24
%
classdef setting_identity < pulse_echo_measurements.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_identity( setup, settings_tx, interval_t, interval_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'setting_identity:NoSetup';
                error( errorStruct );
            end
            % assertion: setup is a single pulse_echo_measurements.setup

            %--------------------------------------------------------------
            % 1.) lower and upper bounds on the times-of-flight
            %--------------------------------------------------------------
%             intervals_tof = times_of_flight( setup );

            %--------------------------------------------------------------
            % 2.) create mix settings
            %--------------------------------------------------------------
            % construct objects
            N_objects = numel( settings_tx );
            mixes = cell( size( settings_tx ) );

            % TODO: choose sampling rate according to settings
            T_s = physical_values.time( 1/20e6 );
            support = physical_values.interval_time( physical_values.time( 0 ), T_s );
            set_t = discretize( support, T_s );

            % set independent properties
            for index_object = 1:N_objects

                % indices of active tx elements
%                 indices_tx_act = settings_tx( index_object ).indices_active;
%                 N_elements_tx = numel( indices_tx_act );

                % create cell arrays
                indices_active = cell( 1, setup.xdc_array.N_elements );
                impulse_responses = cell( 1, setup.xdc_array.N_elements );
                apodization_weights = cell( 1, setup.xdc_array.N_elements );

                for index_element = 1:setup.xdc_array.N_elements

                    %------------------------------------------------------
                    % a) one array element is active
                    %------------------------------------------------------
                    indices_active{ index_element } = index_element;

                    %------------------------------------------------------
                    % b) delta impulse response
                    %------------------------------------------------------
                    impulse_responses{ index_element } = physical_values.impulse_response( set_t, { physical_values.physical_value( 1 ) } );

                    %------------------------------------------------------
                    % c) unity apodization weight
                    %------------------------------------------------------
                    apodization_weights{ index_element } = physical_values.apodization_weight( 1 );

                end % for index_element = 1:setup.xdc_array.N_elements

                % TODO: introduce control_identity / impulse_response_delta
                % create xdc control object
                settings_rx = transducers.control( indices_active, impulse_responses, apodization_weights );

                % create time intervals
                intervals_t = repmat( interval_t, [ 1, setup.xdc_array.N_elements ] );

                % create frequency intervals
                intervals_f = repmat( interval_f, [ 1, setup.xdc_array.N_elements ] );

                % create mixes
                mixes{ index_object } = pulse_echo_measurements.mix( settings_rx, intervals_t, intervals_f );

            end

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@pulse_echo_measurements.setting( settings_tx, mixes );

        end % function objects = setting_identity( setup, settings_tx )

	end % methods

end % classdef setting_identity < pulse_echo_measurements.setting
