%
% superclass for all sequential quasi-(d-1)-spherical wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-05-15
%
classdef sequence_QSW < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_QSW( setup, u_tx_tilde, positions_src, angles, interval_t, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = controls.setting_tx_QSW( setup, u_tx_tilde, positions_src, angles );

            %--------------------------------------------------------------
            % 2.) create reception settings (estimate recording time intervals via options?)
            %--------------------------------------------------------------
            settings_rx = repmat( { controls.setting_rx_identity( setup, interval_t, interval_f ) }, size( settings_tx ) );

            %--------------------------------------------------------------
            % 3.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = pulse_echo_measurements.setting( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings );

        end % function object = sequence_QSW( setup, u_tx_tilde, positions_src, angles, interval_t, interval_f )

	end % methods

end % classdef sequence_QSW < pulse_echo_measurements.sequence
