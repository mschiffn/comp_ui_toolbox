function mustBeEqualSize( objects_1, objects_2 )
% inspect the contents of a *.mat file
%
% author: Martin F. Schiffner
% date: 2019-02-01
% modified: 2019-02-01

    % ensure equal number of dimensions and sizes
	if ndims( objects_1 ) ~= ndims( objects_2 ) || sum( size( objects_1 ) ~= size( objects_2 ) )
        errorStruct.message     = 'objects_1 and objects_2 must have the same number of dimensions and size!';
        errorStruct.identifier	= 'mustBeEqualSize:DimensionOrSizeMismatch';
        error( errorStruct );
    end

end
