function [ field, varargout ] = compute_p_in( setup, spatial_grid, setting_tx )
%
% compute incident acoustic pressure field
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-04-09
%

	%----------------------------------------------------------------------
	% 1.) extract frequency axis
	%----------------------------------------------------------------------
	axis_f = setting_tx.excitation_voltages.axis;
	N_samples_f = abs( axis_f );

	%----------------------------------------------------------------------
	% 2.) normal velocities of active elements
	%----------------------------------------------------------------------
	v_d = setting_tx.excitation_voltages .* setting_tx.impulse_responses;

	%----------------------------------------------------------------------
	% 3.) spatial transfer function of the first array element
	%----------------------------------------------------------------------
	if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

        time_start = tic;
        fprintf( '\t \t: computing spatial transfer function of the first array element...' );

        h_tx_ref = discretizations.spatial_transfer_function( setup.xdc_array.parameters, spatial_grid.grids_elements( 1 ), spatial_grid.grid_FOV, setup.absorption_model, axis_f );

        time_elapsed = toc( time_start );
        fprintf( 'done! (%f s)\n', time_elapsed );

    end

	%----------------------------------------------------------------------
	% 4.) superimpose quasi-(d-1)-spherical waves
	%----------------------------------------------------------------------
	% initialize pressure field with zeros
	if isa( spatial_grid.grid_FOV, 'discretizations.grid_regular' )
        p_incident = physical_values.pascal( zeros( [ spatial_grid.grid_FOV.N_points_axis, N_samples_f ] ) );
    else
        p_incident = physical_values.pascal( zeros( [ spatial_grid.grid_FOV.N_points, N_samples_f ] ) );
    end

	% iterate active array elements
	for index_active = 1:numel( setting_tx.indices_active )

        % index of active array element
        index_element = setting_tx.indices_active( index_active );

        % spatial transfer function of the active array element
        if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

            %--------------------------------------------------------------
            % a) symmetric spatial discretization based on orthogonal regular grids
            %--------------------------------------------------------------
            % shift in grid points required for current array element
            index_element_axis = inverse_index_transform( setup.xdc_array, index_element - 1 );
            N_points_shift_axis = index_element_axis .* spatial_grid.N_points_per_pitch_axis;

            % shift reference spatial transfer function to infer that of the active array element
            h_tx = shift( h_tx_ref, spatial_grid, N_points_shift_axis );

        else

            %--------------------------------------------------------------
            % b) arbitrary grid
            %--------------------------------------------------------------
            % spatial transfer function of the active array element
            h_tx = discretizations.spatial_transfer_function( setup.xdc_array.parameters, spatial_grid.grids_elements( index_element ), spatial_grid.grid_FOV, setup.absorption_model, axis_f );

        end % if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )

        % compute summand for the incident pressure field
        p_incident_summand = h_tx.samples .* repmat( reshape( v_d.samples( index_active, : ), [ ones( 1, spatial_grid.grid_FOV.N_dimensions ), N_samples_f ] ), [ spatial_grid.grid_FOV.N_points_axis, 1 ] );
%         p_incident_summand = h_tx.samples .* repmat( reshape( v_d.samples( index_active, : ), [ 1, N_samples_f ] ), [ spatial_grid.grid_FOV.N_points, 1 ] );

        % added contribution to pressure field
        p_incident = p_incident + physical_values.pascal( double( p_incident_summand ) );
        figure(1);imagesc( squeeze( abs( double( p_incident(:,1,:,end) ) ) ) );

	end % for index_active = 1:numel( setting_tx.indices_active )

    %----------------------------------------------------------------------
	% 5.) create field
	%----------------------------------------------------------------------
	field = discretizations.field( axis_f, spatial_grid.grid_FOV, p_incident );

	if isa( spatial_grid, 'discretizations.spatial_grid_symmetric' )
        varargout{ 1 } = h_tx_ref;
    end

end % function [ field, varargout ] = compute_p_in( setup, spatial_grid, setting_tx )
