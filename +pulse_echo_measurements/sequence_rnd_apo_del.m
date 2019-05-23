%
% superclass for all sequential measurements based on
% superpositions of randomly-apodized and randomly-delayed quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-02-02
% modified: 2019-05-15
%
classdef sequence_rnd_apo_del < pulse_echo_measurements.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_rnd_apo_del( setup, u_tx_tilde, e_theta, settings_rng_apo, settings_rng_del, interval_t, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = controls.setting_tx_rnd_apo_del( setup, u_tx_tilde, e_theta, settings_rng_apo, settings_rng_del );

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

        end % function object = sequence_rnd_apo_del( setup, u_tx_tilde, e_theta, settings_rng_apo, settings_rng_del )

	end % methods

end % classdef sequence_rnd_apo_del < pulse_echo_measurements.sequence
