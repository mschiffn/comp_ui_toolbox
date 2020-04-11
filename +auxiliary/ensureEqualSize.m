function varargout = ensureEqualSize( varargin )
%
% ensure equal number of dimensions and sizes of the input arguments
%
% author: Martin F. Schiffner
% date: 2020-04-09
% modified: 2020-04-09
%

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% check number of arguments
	if nargin < 1
        errorStruct.message = 'At least one input argument is required!';
        errorStruct.identifier = 'ensureEqualSize:FewArguments';
        error( errorStruct );
    end

	%----------------------------------------------------------------------
	% 2.) compare number of dimensions and sizes
	%----------------------------------------------------------------------
    % copy input arguments
    varargout = varargin;

	% iterate input arguments
	for index_arg = 1:nargin

        % compare to additional arguments
        for index_arg_cmp = (index_arg + 1):nargin

            % multiple varargout{ index_arg } / single varargout{ index_arg_cmp }
            if ~isscalar( varargout{ index_arg } ) && isscalar( varargout{ index_arg_cmp } )
                varargout{ index_arg_cmp } = repmat( varargout{ index_arg_cmp }, size( varargout{ index_arg } ) );
            end

            % single varargout{ index_arg } / multiple varargout{ index_arg_cmp }
            if isscalar( varargout{ index_arg } ) && ~isscalar( varargout{ index_arg_cmp } )
                varargout{ index_arg } = repmat( varargout{ index_arg }, size( varargout{ index_arg_cmp } ) );
            end

            % ensure equal number of dimensions and sizes
            if ~isequal( size( varargout{ index_arg } ), size( varargout{ index_arg_cmp } ) )
                errorStruct.message = sprintf( 'Input argument %d differs in dimension and/or size from input argument %d!', index_arg, index_arg_cmp );
                errorStruct.identifier = 'ensureEqualSize:DimensionOrSizeMismatch';
                error( errorStruct );
            end

        end % for index_arg_cmp = (index_arg + 1):nargin

	end % for index_arg = 1:nargin

end % function varargout = ensureEqualSize( varargin )
