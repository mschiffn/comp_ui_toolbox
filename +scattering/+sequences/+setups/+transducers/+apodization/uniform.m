function values = uniform( positions_rel_norm )
% uniform apodization along each coordinate axis
% positions_rel_norm = normalized relative positions of the grid points
%
% author: Martin F. Schiffner
% date: 2019-08-23
% modified: 2019-08-23

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure numeric matrix
    if ~( isnumeric( positions_rel_norm ) && ismatrix( positions_rel_norm ) )
        errorStruct.message = 'shapes must be scattering.sequences.setups.geometry.shape!';
        errorStruct.identifier = 'face:NoShapes';
        error( errorStruct );
    end

    %----------------------------------------------------------------------
	% 2.) compute uniform apodization weights
	%----------------------------------------------------------------------
    values = ones( size( positions_rel_norm, 1 ), 1 );

end % function values = uniform( positions_rel_norm )
