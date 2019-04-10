function tf = isEqualSize( varargin )
% ensure equal number of dimensions and sizes of the arguments
%
% author: Martin F. Schiffner
% date: 2019-02-01
% modified: 2019-04-10

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% check number of arguments
	if nargin < 1
        errorStruct.message     = 'At least one argument is required!';
        errorStruct.identifier	= 'isEqualSize:FewArguments';
        error( errorStruct );
    end
    % assertion: nargin >= 1

	%----------------------------------------------------------------------
	% 2.) compare number of dimensions and sizes
	%----------------------------------------------------------------------
	% use number of dimensions and size of first argument as reference
	N_dimensions_ref = ndims( varargin{ 1 } );
	size_ref = size( varargin{ 1 } );

    % initialize return value
    tf = true;

    % compare to additional arguments
	for index_arg = 2:nargin

        % ensure equal number of dimensions and sizes
        if ndims( varargin{ index_arg } ) ~= N_dimensions_ref || ~all( size( varargin{ index_arg } ) == size_ref )
            tf = false;
            return;
        end

    end % for index_arg = 2:nargin

end % function tf = isEqualSize( varargin )
