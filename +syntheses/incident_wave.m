%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2019-08-23
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
            % ensure class pulse_echo_measurements.sequence (scalar)
            if ~( isa( sequence, 'pulse_echo_measurements.sequence' ) && isscalar( sequence ) )
                errorStruct.message     = 'sequence must be pulse_echo_measurements.sequence!';
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
                % a) check for and inspect file on disk
                %----------------------------------------------------------
                % determine number of active elements
                N_elements_active = numel( sequence.settings( index_object ).tx_unique.indices_active );

                % determine if saving is required
                if N_elements_active >= 2

                    % create format string for filename
                    str_format = sprintf( 'data/%s/setup_%%s/p_in_indices_active_%%s_v_d_unique_%%s.mat', sequence.setup.str_name );

                    % load or compute incident acoustic pressure field (unique frequencies)
                    objects( index_object ).p_incident...
                    = auxiliary.compute_or_load_hash( str_format, @syntheses.compute_p_in, [ 3, 4, 5 ], [ 1, 2 ],...
                        sequence, index_object,...
                        { sequence.setup.xdc_array.aperture, sequence.setup.homogeneous_fluid, sequence.setup.FOV, sequence.setup.str_name },...
                        sequence.settings( index_object ).tx_unique.indices_active, sequence.settings( index_object ).v_d_unique );

                else

                    % compute incident acoustic pressure field (unique frequencies)
                    objects( index_object ).p_incident = syntheses.compute_p_in( sequence, index_object );

                end % if N_elements_active >= 2

            end % for index_object = 1:numel( sequence.settings )

        end % function objects = incident_wave( sequence )

	end % methods

end % classdef incident_wave
