%
% superclass for all frequencies
%
% author: Martin F. Schiffner
% date: 2019-01-15
% modified: 2019-01-21
%
classdef frequency < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = frequency( values )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'frequency:Arguments';
                error( errorStruct );
            end

            % check argument
            mustBePositive( values );

            % constructor of superclass
            obj@physical_values.physical_value( values );
        end
	end % methods
end % classdef frequency
