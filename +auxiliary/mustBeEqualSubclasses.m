function mustBeEqualSubclasses( str_superclass, varargin )
% ensure equal subclasses with the common superclass str_superclass
%
% author: Martin F. Schiffner
% date: 2019-02-12
% modified: 2019-02-12

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% check number of arguments
	if nargin < 2
        errorStruct.message     = 'At least two arguments are required!';
        errorStruct.identifier	= 'mustBeEqualSubclasses:FewArguments';
        error( errorStruct );
    end
    % assertion: nargin >= 2

	%----------------------------------------------------------------------
	% 2.) compare subclasses
	%----------------------------------------------------------------------
	% check first argument and use its class as reference
	if ~isa( varargin{ 1 }, str_superclass )
        errorStruct.message     = sprintf( 'Argument %d is not %s!', 2, str_superclass );
        errorStruct.identifier	= 'mustBeEqualSubclasses:SuperclassMismatch';
        error( errorStruct );
    end
	% assertion: varargin{ 1 } is str_superclass
	str_class_ref = class( varargin{ 1 } );

	% compare to additional arguments
	for index_arg = 2:(nargin - 1)

        % ensure class str_class_ref with the common superclass str_superclass
        if ~strcmp( class( varargin{ index_arg } ), str_class_ref )
            errorStruct.message     = sprintf( 'Argument %d must be %s but is %s!', index_arg + 1, str_class_ref, class( varargin{ index_arg } ) );
            errorStruct.identifier	= 'mustBeEqualSubclasses:ClassMismatch';
            error( errorStruct );
        end

	end % for index_arg = 2:(nargin - 1)

end
