%
% superclass for all incident waves
%
% author: Martin F. Schiffner
% date: 2019-04-06
% modified: 2019-04-06
%
classdef incident_wave

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        p_incident %( 1, 1 ) discretizations.field       % incident acoustic pressure field
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
            % ensure class pulse_echo_measurements.setup
            if ~( isa( setup, 'pulse_echo_measurements.setup' ) && numel( setup ) == 1 )
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'sequence:NoSingleSetup';
                error( errorStruct );
            end

            % ensure class discretizations.spatiospectral
            if ~isa( spatiospectral, 'discretizations.spatiospectral' )
                errorStruct.message     = 'spatiospectral must be discretizations.spatiospectral!';
                errorStruct.identifier	= 'sequence:NoSpatiospectral';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) load or compute incident acoustic pressure fields
            %--------------------------------------------------------------
            % repeat default incident wave
            objects = repmat( objects, size( spatiospectral.spectral ) );

            % compute hashes
            str_hash_setup = hash( setup );
            str_hash_discretization_spatial = hash( spatiospectral.spatial );
            str_name_dir = sprintf( '%s_setup_%s/spatial_%s', setup.str_name, str_hash_setup, str_hash_discretization_spatial );

            % create cell arrays
            N_objects = numel( spatiospectral.spectral );

            % iterate incident waves
            for index_object = 1:N_objects

                fprintf( 'index_object = %d of %d:\n', index_object, N_objects );

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
                    temp = load( str_name_file, 'setup', 'spatial', 'spectral_points_tx' );

                    % ensure equality of configuration
                    if ~( isequal( temp.setup, setup ) && isequal( temp.spatial, spatiospectral.spatial ) && isequal( temp.spectral_points_tx, spatiospectral.spectral( index_object ).tx_unique ) )
                        errorStruct.message     = sprintf( 'Hash collision for %s!', str_name_file );
                        errorStruct.identifier	= 'pressure_incident:WrongObjects';
                        error( errorStruct );
                    end

                end % if indicator_file_exists

                %----------------------------------------------------------
                % b) load or compute incident acoustic pressure field
                %----------------------------------------------------------
                time_start = tic;
                str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
                if indicator_pressure_exists

                    %------------------------------------------------------
                    % a) load incident acoustic pressure field
                    %------------------------------------------------------
                    fprintf( '\t %s: loading incident acoustic pressure field (kappa, %.2f MiB)...', str_date_time, mebibyte( objects( index_object ).size_bytes ) );
                    temp = load( str_name_file, 'p_incident' );
                    objects( index_object ).p_incident = temp.p_incident;

                else

                    %------------------------------------------------------
                    % b) compute incident acoustic pressure field
                    %------------------------------------------------------
                    fprintf( '\t %s: computing incident acoustic pressure field (kappa, %.2f MiB)...', str_date_time, mebibyte( objects( index_object ).size_bytes ) );
                    objects( index_object ).p_incident = compute_p_in( setup, spatiospectral.spatial, spatiospectral.spectral( index_object ).tx_unique );

                    %------------------------------------------------------
                    % c) save result to disk
                    %------------------------------------------------------
                    N_elements_active = numel( spatiospectral.spectral( index_object ).tx_unique.indices_active );
                    if N_elements_active >= 2

                        % specify data structures to save
                        p_incident = objects( index_object ).p_incident;
                        spatial = spatiospectral.spatial;
                        spectral_points_tx = spatiospectral.spectral( index_object ).tx_unique;

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
                            save( str_name_file, 'p_incident', 'setup', 'spatial', 'spectral_points_tx', '-v7.3' );

                        end % indicator_file_exists
                    end % if N_elements_active >= 2

                end % if indicator_pressure_exists

                time_elapsed = toc( time_start );
                fprintf( 'done! (%f s)\n', time_elapsed );

            end % for index_object = 1:N_objects

        end % function objects = pressure_incident( setup, spatiospectral )

	end % methods

end % classdef incident_wave

%--------------------------------------------------------------------------
% compute incident acoustic pressure field
%--------------------------------------------------------------------------
function object = compute_p_in( setup, spatial_grid, setting_tx )

	%----------------------------------------------------------------------
	% 1.) normal velocities of active elements
	%----------------------------------------------------------------------
	v_d = setting_tx.excitation_voltages .* setting_tx.impulse_responses;

	%----------------------------------------------------------------------
	% 2.) spatial transfer function of the first array element
	%----------------------------------------------------------------------
	if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )
        h_tx_ref = discretizations.spatial_transfer_function( spatial_grid.grids_elements( 1 ), spatial_grid.grid_FOV, setup.absorption_model, setting_tx.excitation_voltages.axis );
        factor_interp_tx = round( setup.xdc_array.element_pitch_axis(1) / spatial_grid.grid_FOV.cell_ref.edge_lengths(1) );
    end

	%----------------------------------------------------------------------
	% 3.) superimpose quasi-(d-1)-spherical waves
	%----------------------------------------------------------------------
    p_incident = physical_values.pascal( zeros() );

	for index_active = 1:numel( setting_tx.indices_active )

        % index of active array element
        index_element = setting_tx.indices_active( index_active );
        % TODO: indices_axis

        % spatial transfer function of the active array element
        if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

            %--------------------------------------------------------------
            % a) symmetric grid
            %--------------------------------------------------------------
            % shift in grid points required for current array element
            delta_lattice_points = ( index_element - 1 ) * factor_interp_tx;

            % compute summand for the incident pressure field
            index_start = spatial_grid.grid_FOV.N_points_axis(1) - ( setup.xdc_array.N_elements - 1 ) * factor_interp_tx + 1;
            index_stop = index_start + delta_lattice_points - 1;
            h_tx = [ h_tx_ref( :, index_stop:-1:index_start, : ), h_tx_ref( :, 1:(end - delta_lattice_points), : ) ];

        else

            %--------------------------------------------------------------
            % b) arbitrary grid
            %--------------------------------------------------------------
            % spatial impulse response of the active array element
            h_tx = discretizations.spatial_transfer_function( spatial_grid.grids_elements( index_element ), spatial_grid.grid_FOV, setup.absorption_model, setting_tx.excitation_voltages.axis );

        end % if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

        % compute summand for the incident pressure field
        p_incident_summand = h_tx.samples .* repmat( reshape( v_d( index_active ).samples, [ ones( 1, spatial_grid.grid_FOV.N_dimensions ), N_samples_f ] ), [ spatial_grid.grid_FOV.N_points_axis, 1 ] );

        p_incident = p_incident + p_incident_summand;

	end % for index_active = 1:numel( setting_tx.indices_active )

end % function object = compute_p_in( setup, spatial_grid, setting_tx )
