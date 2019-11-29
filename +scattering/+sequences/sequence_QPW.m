%
% superclass for all sequential quasi-plane wave measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-11-28
%
classdef sequence_QPW < scattering.sequences.sequence

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = sequence_QPW( setup, u_tx_tilde, e_theta, options_mixing )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.mixing
            if ~isa( options_mixing, 'scattering.options.mixing' )
                errorStruct.message = 'options_mixing must be scattering.options.mixing!';
                errorStruct.identifier = 'sequence_QPW:NoMixingOptions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create sequential quasi-plane wave measurements
            %--------------------------------------------------------------
            % create synthesis settings
            settings_tx = scattering.sequences.settings.controls.tx_QPW( setup, u_tx_tilde, e_theta );

            % specify cell array for settings_rx
            settings_rx = cell( size( settings_tx ) );

            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( settings_tx )

                if isa( options_mixing( index_object ), 'scattering.options.mixing_identity' )

                    %------------------------------------------------------
                    % a) create reception settings w/o mixing
                    %------------------------------------------------------
                    settings_rx{ index_object } = scattering.sequences.settings.controls.rx_identity( setup, settings_tx( index_object ), options_mixing( index_object ).interval_f, [], options_mixing( index_object ).impulse_responses );

                elseif isa( options_mixing( index_object ), 'scattering.options.mixing_ir' )

                    %------------------------------------------------------
                    % b) create reception settings w/o mixing
                    %------------------------------------------------------

                elseif isa( options_mixing( index_object ), 'scattering.options.mixing_random' )

                    %------------------------------------------------------
                    % c) create reception settings w/ random mixing
                    %------------------------------------------------------
                    settings_rx{ index_object } = scattering.sequences.settings.controls.rx_random( setup, settings_tx( index_object ), auxiliary.setting_rng, interval_f );

                else

                    %------------------------------------------------------
                    % d) unknown mixing options
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Class of options_mixing( %d ) is unknown!', index_object );
                    errorStruct.identifier = 'sequence_QPW:UnknownOptionsClass';
                    error( errorStruct );

                end % if isa( options_mixing( index_object ), 'scattering.options.mixing_identity' )

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
