%
% superclass for all sequential measurements based on
% superpositions of randomly-apodized and randomly-delayed quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-02-02
% modified: 2019-02-02
%
classdef sequence_rnd_apo_del < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_rnd_apo_del( setup, excitation_voltages_common, e_theta, settings_rng_apo, settings_rng_del )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = syntheses.setting_rnd_apo_del( setup, excitation_voltages_common, e_theta, settings_rng_apo, settings_rng_del );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@pulse_echo_measurements.sequence( setup, settings_tx );
        end

	end % methods

end % classdef sequence_rnd_apo_del
