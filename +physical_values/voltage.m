%
% superclass for all voltages
%
% author: Martin F. Schiffner
% date: 2019-01-15
% modified: 2019-01-23
%
classdef voltage < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = voltage( values )

            % check number of arguments
%             if nargin ~= 1
%                 errorStruct.message     = 'The number of arguments must equal unity!';
%                 errorStruct.identifier	= 'voltage:Arguments';
%                 error( errorStruct );
%             end

            % constructor of superclass
            obj@physical_values.physical_value( values );
        end
	end % methods
end % classdef voltage
