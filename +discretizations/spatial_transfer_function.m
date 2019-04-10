function objects = spatial_transfer_function( xdc_array.parameters, grids_element, grids_FOV, absorption_models, axes_f )
% spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-04-10

    N_points_max = 6;

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% TODO: check for wavenumbers
	% ensure class transducers.parameters
	if ~isa( parameters, 'transducers.parameters' )
        errorStruct.message     = 'grids_element must be discretizations.grid_regular!';
        errorStruct.identifier	= 'spatial_transfer_function:NoRegularGrid';
        error( errorStruct );
    end

	% ensure class discretizations.grid_regular
	if ~isa( grids_element, 'discretizations.grid_regular' )
        errorStruct.message     = 'grids_element must be discretizations.grid_regular!';
        errorStruct.identifier	= 'spatial_transfer_function:NoRegularGrid';
        error( errorStruct );
    end

    %----------------------------------------------------------------------
	% 2.) compute complex-valued wavenumbers
    %----------------------------------------------------------------------
    axes_k_tilde = compute_wavenumbers( absorption_models, axes_f );

	%----------------------------------------------------------------------
	% 3.) compute spatial transfer functions
	%----------------------------------------------------------------------
	% compute Green's functions for specified pairs of grids
% 	h_tx = discretizations.greens_function( grids_element, grids_FOV, axes_k_tilde );

    % ensure cell array for h_tx
%     if ~iscell( h_tx )
%         h_tx = { h_tx };
%     end

    % specify cell array for h_tx
    h_tx = cell( size( grids_element ) );

    % iterate spatial transfer functions
    for index_object = 1:numel( h_tx )

        %------------------------------------------------------------------
        % 3.) compute apodization weights
        %------------------------------------------------------------------
        % compute normalized relative positions of mathematical array elements
        positions_rel = ( grids_element( index_object ).positions - xdc_array.aperture().pos_center ) ./ xdc_array.parameters.element_width_axis;

        % compute apodization weights
        if isscalar( xdc_array.parameters.apodization )
            apo_weights = xdc_array.parameters.apodization( positions_rel );
        else
            apo_weights = parameters.apodization( grids_element( index_object ).positions( :, 1 ), grids_element( index_object ).positions( :, 2 ) );
        end

        %------------------------------------------------------------------
        % compute phase shifts
        %------------------------------------------------------------------

    axes_k_tilde
    parameters.focus

        % initialize results
% TODO: correct unit
        h_tx{ index_object } = physical_values.unity_per_meter( zeros( grids_FOV( index_object ).N_points, abs( axes_k_tilde ) ) );

        %
        N_batches = floor( grids_element( index_object ).N_points / N_points_max );

        for index_batch = 1:N_batches
            
            % indices of current grid points
            index_start = ( index_batch - 1 ) * N_points_max + 1;
            index_stop = index_start + N_points_max - 1;
            disp( ( index_start:index_stop ) );

            % compute Green's functions for specified pairs of grids
            temp = discretizations.greens_function( grids_element( index_object ), grids_FOV( index_object ), axes_k_tilde, ( index_start:index_stop ) );

            % compute apodization weights
            
            h_tx{ index_object } = h_tx{ index_object } + squeeze( sum( temp, 1 ) );

        end

        % compute Green's functions for specified pairs of grids
        indices_remaining = ( (index_stop + 1):grids_element( index_object ).N_points );
        if numel( indices_remaining ) > 0
            temp = discretizations.greens_function( grids_element( index_object ), grids_FOV( index_object ), axes_k_tilde, indices_remaining );
            h_tx{ index_object } = h_tx{ index_object } + squeeze( sum( temp, 1 ) );
        end

        % reshape result for regular grid
        if isa( grids_FOV( index_object ), 'discretizations.grid_regular' )
            h_tx{ index_object } = reshape( h_tx{ index_object }, [ grids_FOV( index_object ).N_points_axis, abs( axes_k_tilde( index_object ) ) ] );
        end

        h_tx{ index_object } = -2 * grids_element( index_object ).cell_ref.volume * h_tx{ index_object };

    end

	%----------------------------------------------------------------------
	% 4.) create fields
	%----------------------------------------------------------------------
	objects = discretizations.field( axes_f, grids_FOV, h_tx );

end % function objects = spatial_transfer_function( grids_element, grids_FOV, absorption_models, axes_f )
