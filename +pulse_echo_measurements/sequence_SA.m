%
% superclass for all sequential quasi-(d-1)-spherical wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-01-25
%
classdef sequence_SA < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_SA( setup, excitation_voltages_common, angles )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = syntheses.setting_QSW( setup, excitation_voltages_common, setup.xdc_array.grid_ctr.positions, angles );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings_tx );
        end
	end % methods
end % classdef sequence_SA
