%
% superclass for all sequential measurements based on
% superpositions of randomly-apodized quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-28
% modified: 2019-06-01
%
classdef sequence_rnd_apo < pulse_echo_measurements.sequence

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_rnd_apo( setup, u_tx_tilde, settings_rng, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = controls.setting_tx_rnd_apo( setup, u_tx_tilde, settings_rng );

            %--------------------------------------------------------------
            % 2.) create reception settings
            %--------------------------------------------------------------
            % specify cell array for settings_rx
            settings_rx = cell( size( settings_tx ) );

            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( settings_tx )

                % create reception settings w/o mixing
                settings_rx{ index_object } = controls.setting_rx_identity( setup, settings_tx( index_object ), interval_f );

            end % for index_object = 1:numel( settings_tx )

            %--------------------------------------------------------------
            % 3.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = pulse_echo_measurements.setting( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings );

        end % function object = sequence_rnd_apo( setup, u_tx_tilde, settings_rng, interval_f )

	end % methods

end % classdef sequence_rnd_apo < pulse_echo_measurements.sequence
