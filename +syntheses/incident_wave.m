%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2019-05-16
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
        function objects = incident_wave( spatiospectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.spatiospectral
            if ~isa( spatiospectral, 'discretizations.spatiospectral' )
                errorStruct.message     = 'spatiospectral must be discretizations.spatiospectral!';
                errorStruct.identifier	= 'incident_wave:NoSpatiospectral';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create incident waves
            %--------------------------------------------------------------
            % repeat default incident wave
            objects = repmat( objects, size( spatiospectral.spectral ) );

            % iterate incident waves
            for index_object = 1:numel( spatiospectral.spectral )

                %----------------------------------------------------------
                % a) check for and inspect file on disk
                %----------------------------------------------------------
                % determine number of active elements
                N_elements_active = numel( spatiospectral.spectral( index_object ).tx_unique.indices_active );

                % determine if saving is required
                if N_elements_active >= 2

                    % create format string for filename
                    str_format = sprintf( 'data/%s/spatial_%%s/p_in_indices_active_%%s_v_d_unique_%%s.mat', spatiospectral.spatial.str_name );

                    % compute / load and save incident acoustic pressure field (unique frequencies)
                    objects( index_object ).p_incident = auxiliary.compute_or_load_hash( str_format, @syntheses.compute_p_in, [ 3, 4, 5 ], [ 1, 2 ], spatiospectral, index_object, spatiospectral.spatial, spatiospectral.spectral( index_object ).tx_unique.indices_active, spatiospectral.spectral( index_object ).v_d_unique );

                else

                    % compute incident acoustic pressure field (unique frequencies)
                    objects( index_object ).p_incident = syntheses.compute_p_in( spatiospectral, index_object );

                end % if N_elements_active >= 2

            end % for index_object = 1:numel( spatiospectral.spectral )

        end % function objects = incident_wave( spatiospectral )

	end % methods

end % classdef incident_wave
