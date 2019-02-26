%
% superclass for all sequential quasi-plane wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-02-12
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
            settings_tx = syntheses.setting_QPW( setup, excitation_voltages_common, e_theta );

            %--------------------------------------------------------------
            % 2.) quantize tx settings (not necessary when using updated synthesis settings)
            %--------------------------------------------------------------
%             settings_tx_quantized = quantize( [settings.tx], 1 / object.setup.f_clk );
            % determine frequency intervals
            % TODO: assertion: f_lb > 0, f_ub >= f_lb + 1 / T_rec
             %--------------------------------------------------------------
            % 3.) estimate recording time intervals
            %--------------------------------------------------------------
%             [ intervals_t, hulls ] = determine_interval_t( object );

            %--------------------------------------------------------------
            % 2.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = pulse_echo_measurements.setting_identity( setup, settings_tx, interval_t, interval_f );

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings );

        end

	end % methods

end % classdef sequence_QPW
