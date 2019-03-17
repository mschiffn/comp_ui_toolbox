%
% superclass for all sequential quasi-plane wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-03-03
%
classdef sequence_QPW < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_QPW( setup, excitation_voltages_common, e_theta, interval_t, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = controls.setting_tx_QPW( setup, excitation_voltages_common, e_theta );

            %--------------------------------------------------------------
            % 2.) create reception settings (estimate recording time intervals via options?)
            %--------------------------------------------------------------
            settings_rx = repmat( { controls.setting_rx_identity( setup, interval_t, interval_f ) }, size( settings_tx ) );
%             indices = randperm( 128 );
%             indices = indices(1:2);
%             settings_rx = repmat( { temp( indices ) }, size( settings_tx ) );

            % determine frequency intervals
            % TODO: assertion: f_lb > 0, f_ub >= f_lb + 1 / T_rec
%             [ intervals_t, hulls ] = determine_interval_t( object );

            %--------------------------------------------------------------
            % 3.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = pulse_echo_measurements.setting( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings );

        end

	end % methods

end % classdef sequence_QPW < pulse_echo_measurements.sequence
