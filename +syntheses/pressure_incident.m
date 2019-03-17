%
% superclass for all incident acoustic pressure fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-03-14
%
classdef pressure_incident < syntheses.field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = pressure_incident( setup, spatiospectral )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@syntheses.field( spatiospectral );
            N_objects = numel( objects );
            % assertion: objects are fields initialized by zeros

            %--------------------------------------------------------------
            % 2.) load or compute incident acoustic pressure fields
            %--------------------------------------------------------------
            % compute hashes
            str_hash_setup = hash( setup );
            str_hash_discretization_spatial = hash( spatiospectral.spatial );
            str_name_dir = sprintf( '%s_setup_%s/spatial_%s', setup.str_name, str_hash_setup, str_hash_discretization_spatial );

            % iterate pulse-echo measurements
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
                [ indicator_file_exists, indicator_pressure_exists ] = auxiliary.investigate_file( str_name_file, { 'pressure_incident_act' } );

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
                    temp = load( str_name_file, 'pressure_incident_act' );
                    objects( index_object ) = temp.pressure_incident_act;

                else

                    %------------------------------------------------------
                    % b) compute incident acoustic pressure field
                    %------------------------------------------------------
                    fprintf( '\t %s: computing incident acoustic pressure field (kappa, %.2f MiB)...', str_date_time, mebibyte( objects( index_object ).size_bytes ) );
                    objects( index_object ) = compute_p_in( objects( index_object ), setup, spatiospectral.spatial, spatiospectral.spectral( index_object ).tx_unique );

                    %------------------------------------------------------
                    % c) save result to disk
                    %------------------------------------------------------
                    N_elements_active = numel( spatiospectral.spectral( index_object ).tx_unique.indices_active );
                    if N_elements_active >= 2

                        % specify data structures to save
                        pressure_incident_act = objects( index_object );
                        spatial = spatiospectral.spatial;
                        spectral_points_tx = spatiospectral.spectral( index_object ).tx_unique;

                        % append or create file
                        if indicator_file_exists

                            % append to existing file
                            save( str_name_file, 'pressure_incident_act', '-append' );

                        else

                            % ensure that folder exists
                            [ success, errorStruct.message, errorStruct.identifier ] = mkdir( str_name_dir );
                            if ~success
                                error( errorStruct );
                            end

                            % create new file
                            save( str_name_file, 'pressure_incident_act', 'setup', 'spatial', 'spectral_points_tx', '-v7.3' );

                        end % indicator_file_exists
                    end % if N_elements_active >= 2

                end % if indicator_pressure_exists

                time_elapsed = toc(time_start);
                fprintf('done! (%f s)\n', time_elapsed);
                % assertion: p_incident_theta_act contains incident acoustic pressure

            end % for index_object = 1:N_objects

        end % function objects = pressure_incident( setup, spatiospectral )

        %------------------------------------------------------------------
        % compute incident acoustic pressure field
        %------------------------------------------------------------------
        function object = compute_p_in( object, setup, spatial_grid, spectral_points_tx )

            %--------------------------------------------------------------
            % 1.) compute complex-valued wavenumbers
            %--------------------------------------------------------------
            axis_f = double( spectral_points_tx.excitation_voltages( 1 ).set_f.S );
            N_samples_f = abs( spectral_points_tx.excitation_voltages( 1 ).set_f );
            axis_k_tilde = setup.absorption_model.compute_wavenumbers( axis_f );

            %--------------------------------------------------------------
            % 2.) normal velocities of active elements
            %--------------------------------------------------------------
            v_d = spectral_points_tx.excitation_voltages .* spectral_points_tx.transfer_functions;

            %--------------------------------------------------------------
            % 3.) spatial transfer function of the first array element
            %--------------------------------------------------------------
            if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

                h_tx_ref = spatial_transfer_function( spatial_grid, axis_k_tilde, 1 );
            end

            %--------------------------------------------------------------
            % 4.) superimpose quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            factor_interp_tx = round( setup.xdc_array.element_pitch_axis(1) / spatial_grid.grid_FOV.delta_axis(1) );
            for index_active = 1:numel( spectral_points_tx.indices_active )

                % index of active array element
                index_element = spectral_points_tx.indices_active( index_active );

                % spatial transfer function of the active array element
                if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

                    %------------------------------------------------------
                    % a) symmetric grid
                    %------------------------------------------------------
                    % shift in grid points required for current array element
                    delta_lattice_points = ( index_element - 1 ) * factor_interp_tx;

                    % compute summand for the incident pressure field
                    index_start = spatial_grid.grid_FOV.N_points_axis(1) - ( setup.xdc_array.N_elements - 1 ) * factor_interp_tx + 1;
                    index_stop = index_start + delta_lattice_points - 1;
                    h_tx = [ h_tx_ref( :, index_stop:-1:index_start, : ), h_tx_ref( :, 1:(end - delta_lattice_points), : ) ];

                else

                    %------------------------------------------------------
                    % b) arbitrary grid
                    %------------------------------------------------------
                    % spatial impulse response of the active array element
                    h_tx = spatial_transfer_function( spatial_grid, axis_k_tilde, index_element );

                end

                % compute summand for the incident pressure field
                p_incident_summand = h_tx .* repmat( reshape( v_d( index_active ).coefficients, [ 1, 1, N_samples_f ] ), [ spatial_grid.grid_FOV.N_points_axis(2), spatial_grid.grid_FOV.N_points_axis(1), 1 ] );

                for index_f = 1:N_samples_f
                    object.values{ index_f } = object.values{ index_f } + p_incident_summand( :, :, index_f );
                end

                show( object );

            end % for index_active = 1:numel( spectral_points_tx.indices_active )

        end % function object = compute_p_in( object, setup, spatial_grid, spectral_points )

	end % methods

end % classdef pressure_incident < syntheses.field
