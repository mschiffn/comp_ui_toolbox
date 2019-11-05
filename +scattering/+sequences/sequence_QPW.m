%
% superclass for all sequential quasi-plane wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-10-22
%
classdef sequence_QPW < scattering.sequences.sequence

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_QPW( setup, u_tx_tilde, e_theta, interval_f )

            %--------------------------------------------------------------
            % 1.) create synthesis settings
            %--------------------------------------------------------------
            settings_tx = scattering.sequences.settings.controls.setting_tx_QPW( setup, u_tx_tilde, e_theta );

            %--------------------------------------------------------------
            % 2.) create reception settings
            %--------------------------------------------------------------
            % specify cell array for settings_rx
            settings_rx = cell( size( settings_tx ) );

            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( settings_tx )

%                 if isa( options_mixing( index_object ), 'options_mixing_identity' )
                    %------------------------------------------------------
                    % create reception settings w/o mixing
                    %------------------------------------------------------
                    settings_rx{ index_object } = scattering.sequences.settings.controls.setting_rx_identity( setup, settings_tx( index_object ), interval_f );

%                 elseif isa( options_mixing( index_object ), 'options_mixing_random' )

                    %------------------------------------------------------
                    % create reception settings w/ random mixing
                    %------------------------------------------------------
%                     settings_rx{ index_object } = scattering.sequences.settings.controls.setting_rx_random( setup, settings_tx( index_object ), auxiliary.setting_rng, interval_f );

%                   else
%                       error
%                   end
            end % for index_object = 1:numel( settings_tx )

            %--------------------------------------------------------------
            % 3.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            settings = scattering.sequences.settings.setting( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            object@scattering.sequences.sequence( setup, settings );

        end % function object = sequence_QPW( setup, u_tx_tilde, e_theta, interval_f )

	end % methods

end % classdef sequence_QPW < scattering.sequences.sequence
