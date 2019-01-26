%
% superclass for all sequential quasi-plane wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-01-23
%
classdef sequence_QPW < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_QPW( setup, excitation_voltages_common, e_theta )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = syntheses.setting_QPW( setup, excitation_voltages_common, e_theta );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings_tx );
        end
	end % methods
end % classdef sequence_QPW
