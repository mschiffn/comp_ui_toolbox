%
% superclass for all sequential quasi-(d-1)-spherical wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-05-15
%
classdef sequence_SA < scattering.sequences.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_SA( setup, u_tx_tilde, angles, interval_t, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = scattering.sequences.settings.controls.tx_QSW( setup, u_tx_tilde, centers( setup.xdc_array ), angles );

            %--------------------------------------------------------------
            % 2.) create reception settings (estimate recording time intervals via options?)
            %--------------------------------------------------------------
            settings_rx = repmat( { scattering.sequences.settings.controls.rx_identity( setup, interval_t, interval_f ) }, size( settings_tx ) );

            %--------------------------------------------------------------
            % 3.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = scattering.sequences.settings.setting( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            object@scattering.sequences.sequence( setup, settings );

        end % function object = sequence_SA( setup, u_tx_tilde, angles, interval_t, interval_f )

	end % methods

end % classdef sequence_SA < scattering.sequences.sequence
