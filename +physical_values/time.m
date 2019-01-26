%
% superclass for all times
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-01-21
%
classdef time < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = time( values )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'time:Arguments';
                error( errorStruct );
            end

            % check argument
            mustBeNonnegative( values );

            % constructor of superclass
            obj@physical_values.physical_value( values );
        end
	end % methods
end % classdef time
