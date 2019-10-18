%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2019-09-25
%
classdef incident_wave

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        p_incident %( 1, 1 ) discretizations.field      % incident acoustic pressure field
        p_incident_grad ( 1, : ) discretizations.field	% spatial gradient of the incident acoustic pressure field

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = incident_wave( sequence )

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
                if isa( sequence.settings( index_object ).tx_unique, 'scattering.sequences.settings.controls.setting_tx_PW' )

                    objects( index_object ) = compute_p_in_pw( objects( index_object ), sequence.setup, sequence.settings( index_object ).tx_unique, sequence.settings( index_object ).v_d_unique );

                else

                    %------------------------------------------------------
                    % b) arbitrary wave
                    %------------------------------------------------------
                    % determine number of active elements
                    N_elements_active = numel( sequence.settings( index_object ).tx_unique.indices_active );

                    % determine if saving is required
                    if N_elements_active >= 2

                        % create format string for filename
                        str_format = sprintf( 'data/%s/setup_%%s/p_in_indices_active_%%s_v_d_unique_%%s.mat', sequence.setup.str_name );

                        % load or compute incident acoustic pressure field (unique frequencies)
                        objects( index_object ).p_incident...
                        = auxiliary.compute_or_load_hash( str_format, @scattering.sequences.syntheses.compute_p_in, [ 3, 4, 5 ], [ 1, 2 ], ...
                            sequence, index_object, ...
                            { sequence.setup.xdc_array.aperture, sequence.setup.homogeneous_fluid, sequence.setup.FOV, sequence.setup.str_name }, ...
                            sequence.settings( index_object ).tx_unique.indices_active, sequence.settings( index_object ).v_d_unique );

                    else

                        % compute incident acoustic pressure field (unique frequencies)
                        objects( index_object ).p_incident = scattering.sequences.syntheses.compute_p_in( sequence, index_object );

                    end % if N_elements_active >= 2

                end % if isa( sequence.settings( index_object ).tx_unique, 'scattering.sequences.settings.controls.setting_tx_PW' )

            end % for index_object = 1:numel( sequence.settings )

        end % function objects = incident_wave( sequence )

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
            incident_wave.p_incident = discretizations.field( v_d_unique.axis, setup.FOV.shape.grid, p_incident );

        end % function incident_wave = compute_p_in_pw( incident_wave, setup, settings_tx_pw )

        %------------------------------------------------------------------
        % arbitrary wave
        %------------------------------------------------------------------
%         function field = compute_p_in
%         end
	end % methods (Access = private, Hidden)

end % classdef incident_wave
