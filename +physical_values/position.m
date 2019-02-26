%
% superclass for all positions
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-02-11
%
classdef position < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = position( values )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'position:Arguments';
                error( errorStruct );
            end

            % constructor of superclass
            objects@physical_values.physical_value( values );

        end

	end % methods

end % classdef position
