function mustBeEqualSize( varargin )
% ensure equal number of dimensions and sizes of the arguments
%
% author: Martin F. Schiffner
% date: 2019-02-01
% modified: 2019-02-02

    % check number of arguments
	if nargin < 2
        errorStruct.message     = 'At least two arguments are required!';
        errorStruct.identifier	= 'mustBeEqualSize:FewArguments';
        error( errorStruct );
    end
    % assertion: nargin >= 2

	% use number of dimensions and size of first argument as reference
	N_dimensions_ref = ndims( varargin{ 1 } );
	size_ref = size( varargin{ 1 } );

    % compare to additional arguments
	for index_arg = 2:nargin

        % ensure equal number of dimensions and sizes
        if ndims( varargin{ index_arg } ) ~= N_dimensions_ref || ~all( size( varargin{ index_arg } ) == size_ref )
            errorStruct.message     = sprintf( 'Argument %d differs in dimension and/or size from the first argument!', index_arg );
            errorStruct.identifier	= 'mustBeEqualSize:DimensionOrSizeMismatch';
            error( errorStruct );
        end
    end
end
