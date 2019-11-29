%
% superclass for all sequential quasi-plane wave measurements
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2019-11-28
%
classdef sequence_PW < scattering.sequences.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_PW( setup, u_tx_tilde, e_theta, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = scattering.sequences.settings.controls.tx_PW( setup, u_tx_tilde, e_theta );

            %--------------------------------------------------------------
            % 2.) create reception settings
            %--------------------------------------------------------------
            % specify cell array for settings_rx
            settings_rx = cell( size( settings_tx ) );

            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( settings_tx )

                % create reception settings w/o mixing
                settings_rx{ index_object } = scattering.sequences.settings.controls.rx_identity( setup, settings_tx( index_object ), interval_f );

            end % for index_object = 1:numel( settings_tx )

            %--------------------------------------------------------------
            % 3.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = scattering.sequences.settings.setting( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            object@scattering.sequences.sequence( setup, settings );

        end % function object = sequence_PW( setup, u_tx_tilde, e_theta, interval_f )

	end % methods

end % classdef sequence_PW < scattering.sequences.sequence
