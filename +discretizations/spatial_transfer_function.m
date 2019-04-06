function objects = spatial_transfer_function( axis_k, grid_element, grid_FOV )
% spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-04-01

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% TODO: check for wavenumbers

	% ensure class discretizations.grid_regular
	if ~isa( grid_element, 'discretizations.grid_regular' )
        errorStruct.message     = 'grid_element must be discretizations.grid_regular!';
        errorStruct.identifier	= 'spatial_transfer_function:NoRegularGrid';
        error( errorStruct );
    end

	%----------------------------------------------------------------------
	% 2.) compute spatial transfer functions
	%----------------------------------------------------------------------
	% TODO: prevent swapping for three-dimensional FOVs
	h_tx = discretizations.greens_function( axis_k, grid_element, grid_FOV );
	h_tx = reshape( squeeze( sum( h_tx, 1 ) ), [ grid_FOV.N_points_axis, abs( axis_k ) ] );
	h_tx = -2 * grid_element.cell_ref.volume * h_tx;

	%----------------------------------------------------------------------
	% 3.) create fields
	%----------------------------------------------------------------------
	objects = discretizations.field( axis_k, grid_FOV, h_tx );

end % function objects = spatial_transfer_function( axis_k, grid_element, grid_FOV, varargin )
