%
% superclass for all sequential quasi-(d-1)-spherical wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-05-27
%
classdef sequence_QSW < scattering.sequences.sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_QSW( setup, u_tx_tilde, positions_src, angles, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = scattering.sequences.settings.controls.tx_QSW( setup, u_tx_tilde, positions_src, angles );

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

        end % function object = sequence_QSW( setup, u_tx_tilde, positions_src, angles, interval_f )

	end % methods

end % classdef sequence_QSW < scattering.sequences.sequence
