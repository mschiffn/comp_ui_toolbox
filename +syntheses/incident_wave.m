%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2019-05-11
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
        function objects = incident_wave( setup, spatiospectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup (scalar)
            if ~( isa( setup, 'pulse_echo_measurements.setup' ) && isscalar( setup ) )
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'incident_wave:NoSingleSetup';
                error( errorStruct );
            end

            % ensure class discretizations.spatiospectral
            if ~isa( spatiospectral, 'discretizations.spatiospectral' )
                errorStruct.message     = 'spatiospectral must be discretizations.spatiospectral!';
                errorStruct.identifier	= 'incident_wave:NoSpatiospectral';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) load or compute incident acoustic pressure fields
            %--------------------------------------------------------------
            % repeat default incident wave
            objects = repmat( objects, size( spatiospectral.spectral ) );

            % compute hashes
% TODO: fix hash
            str_hash_setup = hash( setup );
            str_hash_discretization_spatial = hash( spatiospectral.spatial );
            str_name_dir = sprintf( '%s_setup_%s/spatial_%s', setup.str_name, str_hash_setup, str_hash_discretization_spatial );

            % create cell arrays
            N_objects = numel( spatiospectral.spectral );

            % iterate incident waves
            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) check for and inspect file on disk
                %----------------------------------------------------------
                % create filename
                % TODO: hash collision because hash function ignores parts of the object properties
                str_hash_discretization_spectral = hash( spatiospectral.spectral( index_object ).tx_unique );
                str_name_file = sprintf( '%s/p_in_spectral_%s.mat', str_name_dir, str_hash_discretization_spectral );

                % check existence and contents of file
                [ indicator_file_exists, indicator_pressure_exists ] = auxiliary.investigate_file( str_name_file, { 'p_incident' } );

                % check for hash collisions if file exists
                if indicator_file_exists

                    % load configuration
                    temp = load( str_name_file, 'spatial', 'settings_tx_unique' );

                    % ensure equality of configuration
                    if ~( isequal( temp.spatial, spatiospectral.spatial ) && isequal( temp.settings_tx_unique, spatiospectral.spectral( index_object ).tx_unique ) )
                        errorStruct.message = sprintf( 'Hash collision for %s!', str_name_file );
                        errorStruct.identifier	= 'incident_wave:HashCollision';
                        error( errorStruct );
                    end

                end % if indicator_file_exists

                %----------------------------------------------------------
                % b) load or compute incident acoustic pressure field
                %----------------------------------------------------------
                if indicator_pressure_exists

                    %------------------------------------------------------
                    % a) load incident acoustic pressure field
                    %------------------------------------------------------
                    % print status
                    time_start = tic;
                    str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
                    fprintf( '\t %s: loading incident acoustic pressure field (kappa)...', str_date_time );

                    % load p_incident from file
                    temp = load( str_name_file, 'p_incident' );
                    objects( index_object ).p_incident = temp.p_incident;

                    % infer and print elapsed time
                    time_elapsed = toc( time_start );
                    fprintf( 'done! (%f s)\n', time_elapsed );

                else

                    %------------------------------------------------------
                    % b) compute incident acoustic pressure field
                    %------------------------------------------------------
                    objects( index_object ).p_incident = syntheses.compute_p_in( spatiospectral, index_object );

                    %------------------------------------------------------
                    % c) save result to disk
                    %------------------------------------------------------
                    N_elements_active = numel( spatiospectral.spectral( index_object ).tx_unique.indices_active );
                    if N_elements_active >= 2

                        % specify data structures to save
                        p_incident = objects( index_object ).p_incident;
                        spatial = spatiospectral.spatial;
                        settings_tx_unique = spatiospectral.spectral( index_object ).tx_unique;

                        % append or create file
                        if indicator_file_exists

                            % append to existing file
                            save( str_name_file, 'p_incident', '-append' );

                        else

                            % ensure that folder exists
                            [ success, errorStruct.message, errorStruct.identifier ] = mkdir( str_name_dir );
                            if ~success
                                error( errorStruct );
                            end

                            % create new file
                            save( str_name_file, 'p_incident', 'spatial', 'settings_tx_unique', '-v7.3' );

                        end % if indicator_file_exists

                    end % if N_elements_active >= 2

                end % if indicator_pressure_exists

            end % for index_object = 1:N_objects

        end % function objects = incident_wave( setup, spatiospectral )

	end % methods

end % classdef incident_wave
