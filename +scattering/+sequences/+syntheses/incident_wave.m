%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2020-02-26
%
% TODO: change functionality...
classdef incident_wave

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        p_incident %( 1, 1 ) processing.field      % incident acoustic pressure field
        p_incident_grad ( 1, : ) processing.field	% spatial gradient of the incident acoustic pressure field

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = incident_wave( sequence, filter )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence (scalar)
            if ~( isa( sequence, 'scattering.sequences.sequence' ) && isscalar( sequence ) )
                errorStruct.message     = 'sequence must be scattering.sequences.sequence!';
                errorStruct.identifier	= 'incident_wave:NoSequence';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create incident waves
            %--------------------------------------------------------------
            % repeat default incident wave
            objects = repmat( objects, size( sequence.settings ) );

            % iterate incident waves
            for index_object = 1:numel( sequence.settings )

                %----------------------------------------------------------
                % a) check type of wave
                %----------------------------------------------------------
                if isa( sequence.settings( index_object ).tx_unique, 'scattering.sequences.settings.controls.tx_PW' )

                    objects( index_object ) = compute_p_in_pw( objects( index_object ), sequence.setup, sequence.settings( index_object ).tx_unique, sequence.settings( index_object ).v_d_unique );

                else

                    %------------------------------------------------------
                    % b) arbitrary wave
                    %------------------------------------------------------
                    % determine number of active elements
%                     N_elements_active = numel( sequence.settings( index_object ).tx_unique.indices_active );

                    % compute incident acoustic pressure field (unique frequencies)
                    objects( index_object ).p_incident = compute_p_in( sequence, index_object, filter );

                end % if isa( sequence.settings( index_object ).tx_unique, 'scattering.sequences.settings.controls.tx_PW' )

            end % for index_object = 1:numel( sequence.settings )

        end % function objects = incident_wave( sequence, filter )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % steered plane wave
        %------------------------------------------------------------------
        function incident_wave = compute_p_in_pw( incident_wave, setup, settings_tx_pw, v_d_unique )
% TODO: move to class sequence
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.syntheses.incident_wave
            

            %--------------------------------------------------------------
            % 2.) compute acoustic pressure
            %--------------------------------------------------------------
            % compute current complex-valued wavenumbers
            axis_k_tilde = compute_wavenumbers( setup.homogeneous_fluid.absorption_model, v_d_unique.axis );

            % compute incident acoustic pressure
            p_incident = double( v_d_unique.samples( :, settings_tx_pw.index_ref ) ) .* exp( -1j * axis_k_tilde.members * ( settings_tx_pw.e_theta.components * ( setup.FOV.shape.grid.positions - settings_tx_pw.position_ref )' ) );
            p_incident = physical_values.pascal( p_incident );

            % create field
            incident_wave.p_incident = processing.field( v_d_unique.axis, setup.FOV.shape.grid, p_incident );

        end % function incident_wave = compute_p_in_pw( incident_wave, setup, settings_tx_pw )

        %------------------------------------------------------------------
        % arbitrary wave
        %------------------------------------------------------------------
%         function field = compute_p_in
%         end
	end % methods (Access = private, Hidden)

end % classdef incident_wave
