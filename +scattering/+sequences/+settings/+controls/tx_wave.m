%
% superclass for all control settings in synthesis mode for
% a common wave type
%
% author: Martin F. Schiffner
% date: 2020-04-12
% modified: 2020-07-14
%
classdef tx_wave < scattering.sequences.settings.controls.tx

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        wave ( 1, 1 ) scattering.sequences.syntheses.wave = scattering.sequences.syntheses.deterministic.qpw

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tx_wave( setup, u_tx_tilde, impulse_responses, waves )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure four arguments
            narginchk( 4, 4 );

            % method compute_excitation_voltages ensures class scattering.sequences.setups.setup for setup
            % method compute_excitation_voltages ensures class processing.signal for u_tx_tilde
            % method compute_excitation_voltages ensures class scattering.sequences.syntheses.wave for waves

            % ensure class processing.signal_matrix
            if ~isa( impulse_responses, 'processing.signal_matrix' )
                errorStruct.message = 'impulse_responses must be processing.signal_matrix!';
                errorStruct.identifier = 'tx_wave:NoSignalArrays';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create control settings in synthesis mode (common wave type)
            %--------------------------------------------------------------
            % compute excitation voltages
            [ excitation_voltages, indices_active ] = compute_excitation_voltages( setup, u_tx_tilde, waves );

            % subsample impulse responses
% TODO: subsampling does not work properly!
            impulse_responses = subsample( impulse_responses, [], indices_active );

            % constructor of superclass
            objects@scattering.sequences.settings.controls.tx( indices_active, num2cell( impulse_responses ), num2cell( excitation_voltages ) );

            % iterate control settings in synthesis mode (common wave type)
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).wave = waves( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = tx_wave( setup, u_tx_tilde, impulse_responses, waves )

	end % methods

end % classdef tx_wave < scattering.sequences.settings.controls.tx
