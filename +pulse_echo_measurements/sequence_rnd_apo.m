%
% superclass for all sequential measurements based on
% superpositions of randomly-apodized quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-28
% modified: 2019-01-28
%
classdef sequence_rnd_apo < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_rnd_apo( setup, excitation_voltages_common, settings_rng )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = syntheses.setting_rnd_apo( setup, excitation_voltages_common, settings_rng );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings_tx );
        end
	end % methods
end % classdef sequence_rnd_apo
