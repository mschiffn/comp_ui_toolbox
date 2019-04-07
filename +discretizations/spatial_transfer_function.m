function objects = spatial_transfer_function( grids_element, grids_FOV, absorption_models, axes_f )
% spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-04-07

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% TODO: check for wavenumbers
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
	% TODO: prevent swapping for three-dimensional FOVs
    % compute Green's functions for specified pairs of grids
	h_tx = discretizations.greens_function( grids_element, grids_FOV, axes_k_tilde );

    % ensure cell array for h_tx
    if ~iscell( h_tx )
        h_tx = { h_tx };
    end

    for index_object = 1:numel( h_tx )

        h_tx{ index_object } = squeeze( sum( h_tx{ index_object }, 1 ) );

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
