%
% superclass for all incident acoustic pressure fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-02-22
%
classdef pressure_incident < syntheses.field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = pressure_incident( setup, settings, discretization )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@syntheses.field( discretization );
            N_objects = numel( objects );
            % assertion: objects are fields initialized by zeros

            %--------------------------------------------------------------
            % 2.) load or compute incident acoustic pressure fields
            %--------------------------------------------------------------
            str_hash_setup = hash( setup );
            for index_object = 1:N_objects

                fprintf( 'index_object = %d of %d:\n', index_object, N_objects );

                %----------------------------------------------------------
                % check for and inspect file on disk
                %----------------------------------------------------------
                % create filename
                str_filename = sprintf( '%s_setup_%s/p_in_measurement_%s.mat', setup.str_name, str_hash_setup, hash( settings( index_object ) ) );

                % check existence and contents of file
                [ indicator_file_exists, indicator_pressure_exists ] = auxiliary.investigate_file( str_filename, { 'pressure_incident_act' } );

                % check for hash collisions if file exists
                if indicator_file_exists
                    % TODO: load and measurement and compare for equality
                    temp = load( str_filename, 'setup', 'measurement_act' );
                    % isequal returns logical 1 (true) for objects of the same class with equal property values.
                    if ~isequal( temp.setup, setup ) || ~isequal( temp.measurement_act, settings( index_object ) )
                        errorStruct.message     = sprintf( 'Hash collision for %s!', str_filename );
                        errorStruct.identifier	= 'pressure_incident:WrongObjects';
                        error( errorStruct );
                    end
                end

                time_start = tic;
                str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
                if indicator_pressure_exists

                    %------------------------------------------------------
                    % a) load incident acoustic pressure field
                    %------------------------------------------------------
                    fprintf( '\t %s: loading incident acoustic pressure field (kappa, %.2f MiB)...', str_date_time, mebibyte( objects( index_object ).size_bytes ) );
                    temp = load( str_filename, 'pressure_incident_act' );
                    objects( index_object ) = temp.pressure_incident_act;
                else

                    %------------------------------------------------------
                    % b) compute incident acoustic pressure field
                    %------------------------------------------------------
                    fprintf( '\t %s: computing incident acoustic pressure field (kappa, %.2f MiB)...', str_date_time, mebibyte( objects( index_object ).size_bytes ) );
                    if isa( discretization.space, 'discretizations.spatial_grid_symmetric' )
                        %
                        objects( index_object ) = compute_p_in_symmetric( objects( index_object ), setup, settings( index_object ), discretization.space, discretization.frequency( index_object ) );
                    else
                        %
                        objects( index_object ) = compute_p_in( objects( index_object ), setup, settings( index_object ) );
                    end

                    %------------------------------------------------------
                    % c) save result to disk
                    %------------------------------------------------------
                    N_elements_active = numel( settings( index_object ).tx.indices_active );
                    if N_elements_active >= 2
                        pressure_incident_act = objects( index_object );
                        measurement_act = settings( index_object );
                        if indicator_file_exists
                            % append to existing file
                            save( str_filename, 'pressure_incident_act', '-append' );
                        else
                            % create new file
                            save( str_filename, 'pressure_incident_act', 'setup', 'measurement_act', '-v7.3' );
                        end % indicator_file_exists
                    end % if N_elements_active >= 2
                end % if indicator_pressure_exists

                time_elapsed = toc(time_start);
                fprintf('done! (%f s)\n', time_elapsed);
                % assertion: p_incident_theta_act contains incident acoustic pressure

            end % for index_object = 1:N_objects

        end % function objects = pressure_incident( setup, settings, discretization )

        %------------------------------------------------------------------
        % compute incident acoustic pressure field (symmetric grids)
        %------------------------------------------------------------------
        function object = compute_p_in_symmetric( object, setup, setting, spatial_grid_symmetric, set_discrete_frequency )

            %--------------------------------------------------------------
            % 1.) frequency variables
            %--------------------------------------------------------------
            % compute complex-valued wavenumbers
            axis_f = double( [ set_discrete_frequency.S ] );
            axis_omega = 2*pi*axis_f;
            N_samples_f = abs( set_discrete_frequency );
            axis_k_tilde = setup.absorption_model.compute_wavenumbers( axis_f );

            %--------------------------------------------------------------
            % 2.) Fourier coefficients of the excitation voltages
            %--------------------------------------------------------------
            N_elements_active = numel( setting.tx.indices_active );
            % TODO: identical frequency axes?
            excitation_voltages = fourier_coefficients( setting.tx.excitation_voltages, setting.interval_t, setting.interval_f );

            %--------------------------------------------------------------
            % 3.) compute reference fields radiated by the first array element
            %--------------------------------------------------------------
            % TODO: use spatial impulse response
            p_in_ref = zeros( spatial_grid_symmetric.grids_elements( 1 ).N_points, spatial_grid_symmetric.grid_FOV.N_points, N_samples_f );
            for index_f = 1:N_samples_f
                p_in_ref( :, :, index_f ) = 0.25j * excitation_voltages( 1 ).coefficients( index_f ) * besselh( 0, 2, axis_k_tilde( index_f ) * spatial_grid_symmetric.D_ref );
            end

            p_in_ref = reshape( squeeze( sum( p_in_ref, 1 ) ), [ spatial_grid_symmetric.grid_FOV.N_points_axis(2), spatial_grid_symmetric.grid_FOV.N_points_axis(1), N_samples_f ] );

            %--------------------------------------------------------------
            % 2.) compute phase shifts
            %--------------------------------------------------------------
            shift_phase = exp( -1j * axis_omega(:) * [ setting.tx.time_delays.value ] );

            %--------------------------------------------------------------
            % 4.) superimpose quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            factor_interp_tx = round( setup.xdc_array.element_pitch_axis(1) / spatial_grid_symmetric.grid_FOV.delta_axis(1) );
            for index_active = 1:N_elements_active

                % index of active array element
                index_element = setting.tx.indices_active( index_active );

                % shift in grid points required for current array element
                delta_lattice_points = ( index_element - 1 ) * factor_interp_tx;

                % compute summand for the incident pressure field
                index_start = spatial_grid_symmetric.grid_FOV.N_points_axis(1) - ( setup.xdc_array.N_elements - 1 ) * factor_interp_tx + 1;
                index_stop = index_start + delta_lattice_points - 1;
                p_incident_summand = [ p_in_ref( :, index_stop:-1:index_start, : ), p_in_ref( :, 1:(end - delta_lattice_points), : ) ];
                p_incident_summand = p_incident_summand .* repmat( reshape( shift_phase( :, index_active ), [1, 1, N_samples_f] ), [ spatial_grid_symmetric.grid_FOV.N_points_axis(2), spatial_grid_symmetric.grid_FOV.N_points_axis(1), 1 ] );
                p_incident_summand = p_incident_summand * setting.tx.apodization_weights( index_active ).value;

                for index_f = 1:N_samples_f
                    object.values{ index_f } = object.values{ index_f } + p_incident_summand( :, :, index_f );
                end

                show( object )
            end % for index_active = 1:N_elements_active

        end % function object = compute_p_in_symmetric( object, setup, setting, spatial_grid_symmetric, set_discrete_frequency )

        %------------------------------------------------------------------
        % compute incident acoustic pressure field (arbitrary grids)
        %------------------------------------------------------------------
        function object = compute_p_in( object, setup, measurement )

            %--------------------------------------------------------------
            % 1.) frequency variables
            %--------------------------------------------------------------
            % compute complex-valued wavenumbers
            axis_f = [ measurement.set_f.F_BP.value ];
            axis_omega = 2*pi*axis_f;
            N_samples_f = numel( axis_f );
            axis_k_tilde = setup.absorption_model.compute_wavenumbers( axis_f );

            %--------------------------------------------------------------
            % 2.) Fourier coefficients of the excitation voltages
            %--------------------------------------------------------------
            N_elements_active = numel( setting.tx.indices_active );
            excitation_voltages = zeros( N_samples_f, N_elements_active );
            for index_active = 1:N_elements_active

                % number of samples at current sampling rate
                N_samples_t = ( measurement.interval_t.bounds(2).value - measurement.interval_t.bounds(1).value ) * measurement.settings_tx.excitation_voltages( index_active ).f_s.value;
                axis_f_act = (0:(N_samples_t - 1)) * measurement.settings_tx.excitation_voltages( index_active ).f_s.value / N_samples_t;
                indicator = ( axis_f_act >= measurement.interval_f.bounds(1).value ) & ( axis_f_act <= measurement.interval_f.bounds(2).value );

                % compute Fourier coefficients
                temp = sqrt( N_samples_t ) * fft( double( measurement.settings_tx.excitation_voltages( index_active ).u_tx_tilde ), N_samples_t, 1 );
                excitation_voltages( :, index_active ) = temp( indicator );
            end

            %--------------------------------------------------------------
            % 3.) compute phase shifts
            %--------------------------------------------------------------
            shift_phase = exp( -1j * axis_omega(:) * [ measurement.settings_tx.time_delays.value ] );

            
            %--------------------------------------------------------------
            % 3.) compute reference fields radiated by the first array element
            %--------------------------------------------------------------
            p_in_ref = zeros( setup.xdc_array.grid(1).N_points, setup.FOV.grid.N_points, N_samples_f );
            for index_f = 1:N_samples_f
                p_in_ref( :, :, index_f ) = 0.25j * excitation_voltages( index_f, index_active ) * besselh( 0, 2, axis_k_tilde( index_f ) * setup.D_ref );
            end

            p_in_ref = reshape( squeeze( sum( p_in_ref, 1 ) ), [ setup.FOV.grid.N_points_axis(2), setup.FOV.grid.N_points_axis(1), N_samples_f ] );

            

            %--------------------------------------------------------------
            % 4.) superimpose quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            factor_interp_tx = setup.xdc_array.element_pitch_axis(1) / setup.FOV.grid.delta_axis(1);
            for index_active = 1:N_elements_active

                % index of active array element
                index_element = measurement.settings_tx.indices_active( index_active );

                % shift in grid points required for current array element
                delta_lattice_points = ( index_element - 1 ) * factor_interp_tx;

                % compute summand for the incident pressure field
                index_start = setup.FOV.grid.N_points_axis(1) - ( setup.xdc_array.N_elements - 1 ) * factor_interp_tx + 1;
                index_stop = index_start + delta_lattice_points - 1;
                p_incident_summand = [ p_in_ref( :, index_stop:-1:index_start, : ), p_in_ref( :, 1:(end - delta_lattice_points), : ) ];
                p_incident_summand = p_incident_summand .* repmat( reshape( shift_phase( :, index_active ), [1, 1, N_samples_f] ), [ setup.FOV.grid.N_points_axis(2), setup.FOV.grid.N_points_axis(1), 1 ] );
                p_incident_summand = p_incident_summand * measurement.settings_tx.apodization_weights( index_active ).value;

                for index_f = 1:N_samples_f
                    object.values{ index_f } = object.values{ index_f } + p_incident_summand( :, :, index_f );
                end
            end % for index_active = 1:N_elements_active

        end % function object = compute_p_in( object, setup, measurement )

	end % methods

end % classdef pressure_incident
