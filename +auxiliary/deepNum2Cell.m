function [ input, depth, uniform ] = deepNum2Cell( input )
%
% recursively traverses
% nested cell arrays and converts
% arrays at
% the deepest nodes into
% cell arrays
%
% author: Martin F. Schiffner
% date: 2019-08-09
% modified: 2019-08-09
%

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% check for quick exit
	if ~iscell( input )

        input = num2cell( input );
        depth = 1;
        uniform = true;

    end

	%----------------------------------------------------------------------
	% 2.) recursively traverse nested cells
	%----------------------------------------------------------------------
	% initialize depth w/ zeros
	depth = zeros( size( input ) );
    uniform = false( size( input ) );

	% iterate cells of cell array input
	for index_cell = 1:numel( input )
        [ input{ index_cell }, depth( index_cell ), uniform( index_cell ) ] = auxiliary.deepNum2Cell( input{ index_cell } );
    end

    % determine uniformity
    if all( depth( : ) == depth( 1 ) )
        uniform = true;
    end

	% increment depth
	depth = max( depth( : ) + 1 );

end % function [ input, depth, uniform ] = deepNum2Cell( input )
